import functools
import json

import numpy as np
import pynq as pq
from pynq.overlay import Overlay

from tpu.isa import Activation
from tpu.tpu import TPUConfig
from tpu.sw import *


class PynqDevice:
    def __init__(self, ip_config, ip_name="TPU_0", bitstream="tpu.bit"):
        self.config = ip_config
        self.ol = Overlay(bitstream, download=False)
        self.csr_mmio = pq.MMIO(self.ol.ip_dict[ip_name]["phys_addr"], self.ol.ip_dict[ip_name]["addr_range"])

    def download(self):
        pq.pl.PL.reset()
        self.ol.download()
        self.ol.reset()

    def _start_tpu(self, instrs):
        instr_words = repack([self.config.isa_layout.const(instr).as_bits() for instr in instrs],
            self.config.isa_layout.size, self.config.host_data_width, aligned=True)
        instr_bytes = [byte for word in instr_words for byte in unpacked(word, self.config.host_data_width, 8)]
        instr_buf = pq.allocate(len(instr_bytes), dtype=np.uint8)
        instr_buf[:] = instr_bytes

        xfer_size = min(self.config.instr_fifo_depth, self.config.max_reps)
        ins_xfers = -(-len(instrs) // xfer_size)
        for i in range(ins_xfers):
            nins = xfer_size if i < ins_xfers - 1 else len(instrs) - i * xfer_size
            addr_offset = i * xfer_size * aligned_size(self.config.isa_layout.size, self.config.host_data_width) // 8
            self.csr_mmio.write(self.config.csr_offsets["insadr"], instr_buf.physical_address + addr_offset)
            self.csr_mmio.write(self.config.csr_offsets["nins"], nins)
            self.csr_mmio.write(self.config.csr_offsets["tpus"], 1)
            while not self.csr_mmio.read(self.config.csr_offsets["tpur"]): pass

    def qmatmul(self, a, bbuf, cbuf, n, zd, qmul, shamt, actfn=Activation.RELU):
        m, k = a.shape
        a_bytes = pack_activations(a, self.config)
        abuf = pq.allocate(len(a_bytes), dtype=np.uint8)
        abuf[:] = a_bytes

        a_tile_row_size = aligned_size(self.config.act_dtype.width * self.config.rows, self.config.host_data_width) // 8
        resbuf = pq.allocate(m * ceildiv(n, self.config.cols) * a_tile_row_size, dtype=np.uint8)

        instrs = tpu_matmul(m, k, n, self.config, zd.item(), shamt.item(), qmul.item(), actfn=actfn,
            a_haddr=abuf.physical_address, c_haddr=cbuf.physical_address, b_haddr=bbuf.physical_address, d_haddr=resbuf.physical_address)
        self._start_tpu(instrs)
        return unpack_activations(resbuf, m, n, self.config)

class PynqModel:
    def __init__(self, modelfn, tensorfn, device: PynqDevice):
        self.dev = device
        with open(modelfn) as fm, open(tensorfn, 'rb') as ft:
            self.ops = json.load(fm)
            self.tensors = ft.read()
        self.layers = [self._parse_op(op) for op in self.ops]

    def __call__(self, x):
        return functools.reduce(lambda x, f: f(x), self.layers, x)

    def _parse_op(self, op):
        match op["op"]:
            case "fully_connected":
                qparams = {k: np.array(v) for k, v in op["qparams"].items()}
                tensors = {k: self._extract_tensor(v)  for k, v in op["args"].items()}
                bbuf = pq.allocate(len(tensors["weight"]), dtype=np.uint8)
                bbuf[:] = tensors["weight"]
                cbuf = pq.allocate(len(tensors["bias"]), dtype=np.uint8)
                cbuf[:] = tensors["bias"]
                n = op["args"]["weight"]["shape"][1]
                return functools.partial(self.dev.qmatmul, n=n, bbuf=bbuf, cbuf=cbuf,
                    zd=qparams["output_zp"], qmul=qparams["qmul"], shamt=qparams["shamt"], actfn=Activation[op["act"]])

            case "flatten":
                def flatten(x):
                    start_dim, end_dim = op["args"]
                    end_dim = len(x.shape) + end_dim if end_dim < 0 else end_dim
                    return x.reshape(*x.shape[:start_dim], -1, *x.shape[end_dim + 1:])
                return flatten

    def _extract_tensor(self, info):
        return np.frombuffer(self.tensors[info["offset"]:info["offset"] + info["size"]], dtype=np.uint8)


if __name__ == '__main__':
    with open("config.json") as f:
        config = TPUConfig.fromdict(json.load(f))
    print("Downloading bitstream")
    dev = PynqDevice(config, bitstream="tpu.bit")
    dev.download()
    print("Done")
    model = PynqModel("model.json", "tensors.bin", dev)

    *_, x_test, y_test = fetch_mnist()
    x = x_test.astype(np.float32) / 255.0
    x = resize_bilinear(x.reshape(-1, 28, 28), 12, 12)
    qmin, qmax = dtype_to_bounds(config.act_dtype)
    s, z = 1.0 / (qmax - qmin), qmin
    x = (x / s + z).astype(config.act_dtype.numpy)

    print("Some examples on PL")
    y = model(x[1000:1003])
    for i in range(3):
        display_image(x[1000+i].reshape(-1).astype(np.int32) + 128, size=(12, 12))
        display_outputs(y[i], y_test[1000+i], qmin, qmax)

    print("Accuracy Check between PS (numpy) and PL (tpu)")
    # numpy
    npmodel = NumpyModel("model.json", "tensors.bin", config)
    npy = npmodel(x)
    correct = sum(npy.argmax(axis=-1) == y_test)
    print(f"PS: {correct}/{len(x_test)} correct predictions, {100.0 * correct / len(x_test):.2f}% accuracy")

    # fpga
    pbar = tqdm(total=10000)
    correct, total = 0, 0
    for xb, yb in loader(x_test, y_test, bs=16):
        xq = (xb / s + z).astype(config.act_dtype.numpy)
        y = model(xq)
        correct += sum(y.argmax(axis=-1) == yb)
        total += len(xb)
        pbar.desc = f"PL: {correct}/{total} correct predictions"
        pbar.update(len(xb), close=total==len(x_test))
    print(f"PL: {correct}/{len(x_test)} correct predictions, {100.0 * correct / len(x_test):.2f}% accuracy")
