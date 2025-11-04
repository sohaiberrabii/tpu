from dataclasses import asdict
from pathlib import Path
import json

import os
os.environ["GPI_LOG_LEVEL"] = os.getenv("GPI_LOG_LEVEL", "WARNING")
os.environ["COCOTB_LOG_LEVEL"] = os.getenv("COCOTB_LOG_LEVEL", "WARNING")
import cocotb

from amaranth.back.verilog import convert
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tpu.sw import *
from tpu.tpu import IntType, TPUConfig, TPU
from test.cocotb_test.helpers import CocotbModel, TPUAxiInterface, cocotb_run

ASSETDIR = Path(__file__).parent / "assets"
ASSETDIR.mkdir(exist_ok=True)
COCOTB = int(os.getenv("COCOTB", 0))

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, 18),
            nn.ReLU(),
            nn.Linear(18, 10)
        )
    def forward(self, x):
        return self.layers(x)

def train(model, loader, optimizer, criterion, device):
    model.train()
    for data, target in loader:
        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

@torch.no_grad
def test(model, loader, device):
    model.eval()
    correct = 0
    for data, target in loader:
        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return correct

def loader(x, y, bs=64):
    for i in range(-(-len(x) // bs)):
        batch = x[i * bs:(i + 1) * bs]
        batch = batch.astype(np.float32) / 255.0
        batch = resize_bilinear(batch.reshape(-1, 28, 28), 12, 12)
        yield batch, y[i * bs:(i + 1) * bs]

@torch.no_grad
def quantize(model, act_stats, config):
    qlayers = []
    params_blob = bytearray()
    offset = 0
    for i, module in enumerate(model.layers):
        if isinstance(module, nn.Linear):
            sa, za = qparams(act_stats[f"layer.{i}"]["input"], config.act_dtype)
            sd, zd = qparams(act_stats[f"layer.{i}"]["output"], config.act_dtype)
            b, c = module.weight.T.cpu().numpy(), module.bias.cpu().numpy()
            sb, _ = qparams((b.min().item(), b.max().item()), config.weight_dtype, symmetric=True)
            bq, cq, qmul, shamt = quantize_matmul(b, c, sa, za, sb, sd, config)

            params_blob.extend(weight_bytes := pack_weights(bq, config))
            params_blob.extend(bias_bytes := pack_bias(cq, config))

            qlayers.append({
                "op": "fully_connected",
                "qparams": {
                    "output_zp": zd[0].item(),
                    "qmul": qmul[0].item(),
                    "shamt": shamt[0].item(),
                },
                "args": {
                    "weight": {
                        "offset": offset,
                        "shape": list(bq.shape),
                        "size": len(weight_bytes),
                        "dtype": asdict(config.weight_dtype),
                    },
                    "bias": {
                        "offset": offset + len(weight_bytes), 
                        "shape": list(cq.shape),
                        "size": len(bias_bytes),
                        "dtype": asdict(config.acc_dtype),
                    }
                },
                "act": "NONE",
            })
            offset += len(weight_bytes) + len(bias_bytes)

        elif isinstance(module, nn.ReLU):
            qlayers[-1]["act"] = "RELU"
        elif isinstance(module, nn.Flatten):
            qlayers.append({"op": "flatten", "args": [module.start_dim, module.end_dim]})

    return qlayers, params_blob

RESETC = "\033[39m\033[49m"
GREEN = "\033[38;2;44;160;44m"
RED = "\033[38;2;214;39;40m"

def display_image(dat, size=(28, 28)):
    """Expects uint8 data"""
    def bg(gs): return f"\033[48;2;{gs};{gs};{gs}m"
    def fg(gs): return f"\033[38;2;{gs};{gs};{gs}m"
    num_rows, num_cols = size
    for i in range(0, num_rows, 2):
        print("".join(f"{fg(int(dat[i * num_cols + j]))}{bg(int(dat[(i + 1) * num_cols + j]))}▀" for j in range(num_cols)) + RESETC)

def display_outputs(output, expected, min, max):
    def bar(x, color=""):
        return color + "▄" * round(20 * (x.astype(np.int32) - min) / (max - min)) + RESETC + f" {x}"
    actual = np.argmax(output)
    for i, x in enumerate(output):
        print(f"{i}▕ {bar(x, color=GREEN if i == expected else (RED if i == actual else ''))}")

def get_act_stats(model, xcalib):
    import torch
    assert hasattr(model, "layers")
    def activation_observer(name, act_stats, relu):
        def hook(module, input, output):
            inmn, inmx = input[0].min().item(), input[0].max().item()
            outmn, outmx = output.min().item(), output.max().item()
            outmn = max(0, outmn) if relu else outmn

            if name not in act_stats:
                act_stats[name] = {"input": (inmn, inmx), "output": (outmn, outmx)}
            else:
                act_stats[name]["input"] = (min(act_stats[name]["input"][0], inmn), max(act_stats[name]["input"][1], inmx))
                act_stats[name]["output"] = (min(act_stats[name]["output"][0], outmn), max(act_stats[name]["output"][1], outmx))
        return hook
    hooks, act_stats = [], {}
    for i, module in enumerate(model.layers):
        relu_next = isinstance(model.layers[i + 1], nn.ReLU) if i + 1 < len(model.layers) else False
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(activation_observer(f"layer.{i}", act_stats, relu=relu_next)))

    with torch.no_grad():
        model(xcalib)

    for h in hooks:
        h.remove()
    return act_stats

@cocotb.test()
async def mnist_tb(dut):
    with open(Path(__file__).parent / "build" / "config.json") as f:
        config = TPUConfig.fromdict(json.load(f))

    tpu_axi = TPUAxiInterface(dut)
    await tpu_axi.init()
    await tpu_axi.reset()
    tb_model = CocotbModel(tpu_axi, config, ASSETDIR / "model.json", ASSETDIR / "tensors.bin")

    *_, x, y = fetch_mnist()
    qmin, qmax = dtype_to_bounds(config.act_dtype)
    s, z = 1.0 / (qmax - qmin), qmin
    correct, total = 0, 0
    pbar = tqdm(total=10000)
    for xb, yb in loader(x, y, bs=64):
        xq = (xb / s + z).astype(config.act_dtype.numpy)
        # display_image(xb[0].reshape(-1).astype(np.int32) + 128, size=(12, 12))
        tb_y = await tb_model(xq)
        # display_outputs(tb_y[0], yb[0], qmin, qmax)
        correct += sum(tb_y.argmax(axis=-1) == yb)
        total += len(xb)
        pbar.desc = f"COCOTB: {correct}/{total} correct predictions"
        pbar.update(len(xb), close=total==len(x))

    print(f"COCOTB: {correct}/{len(x)} correct predictions, {100.0 * correct / len(x):.2f}% accuracy")


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = fetch_mnist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Training")
    best_acc = 0.
    for epoch in (pbar := tqdm(range(1, 51))):
        train(model, loader(x_train, y_train, bs=64), optimizer, criterion, device)
        correct = test(model, loader(x_test, y_test, bs=10000), device)
        acc = 100.0 * correct / len(x_test)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ASSETDIR / "model.pth")
        pbar.desc = f"Epoch: {epoch}, Test accuracy: {acc:.2f}%"

    state_dict = torch.load(ASSETDIR / "model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    correct = test(model, loader(x_test, y_test, bs=10000), device)
    print(f"FP32: {correct}/{len(x_test)} correct predictions, {100.0 * correct / len(x_test):.2f}% accuracy")

    # quantization and export
    config = TPUConfig(rows=8, cols=8, act_mem_depth=32, acc_mem_depth=32, weight_fifo_depth=32, instr_fifo_depth=16,
        act_dtype=IntType(width=8, signed=True), weight_dtype=IntType(width=8, signed=True), acc_dtype=IntType(width=32, signed=True))
    xcalib, _ = next(iter(loader(x_train, y_train, bs=10000)))
    act_stats = get_act_stats(model, torch.tensor(xcalib).to(device))
    qlayers, tensorsbin = quantize(model, act_stats, config)
    with open(ASSETDIR / "model.json", "w") as fm, open(ASSETDIR / "tensors.bin", "wb") as ft:
        ft.write(tensorsbin)
        json.dump(qlayers, fm, indent=2)

    # numpy quantized baseline
    npmodel = NumpyModel(ASSETDIR / "model.json", ASSETDIR / "tensors.bin", config)
    x, _ = next(iter(loader(x_test, y_test, bs=10000)))
    sa, za = qparams(act_stats["layer.1"]["input"], config.act_dtype)
    xq = (x / sa + za).astype(config.act_dtype.numpy)
    y = npmodel(xq)
    correct = sum(y.argmax(axis=-1) == y_test)
    print(f"INT{config.act_dtype.width}: {correct}/{len(x_test)} correct predictions, {100.0 * correct / len(x_test):.2f}% accuracy")

    # generate the verilog and run cocotb simulation
    if COCOTB:
        dut = TPU(config)
        builddir = Path(__file__).parent / "build"
        builddir.mkdir(exist_ok=True)
        with open(builddir / "tpu.v", 'w') as f:
            f.write(convert(dut, name="TPU", emit_src=False, strip_internal_attrs=True))
        with open(builddir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=4)
        cocotb_run([builddir / "tpu.v"], "TPU", waves=True, sim="verilator", module="mnist")
