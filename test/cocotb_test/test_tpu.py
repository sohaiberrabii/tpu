# test generated verilog with cocotb

import os
import random
from pathlib import Path
from dataclasses import asdict
import json
import pytest

os.environ["GPI_LOG_LEVEL"] = os.getenv("GPI_LOG_LEVEL", "WARNING")
os.environ["COCOTB_LOG_LEVEL"] = os.getenv("COCOTB_LOG_LEVEL", "WARNING")
import cocotb

from amaranth.back.verilog import convert

from tpu.isa import Activation
from tpu.tpu import TPU, TPUConfig
from test.cocotb_test.helpers import cocotb_run, TPUAxiInterface, run_tpu
from test.helpers import *

#TODO: tests could be much faster we don't regen verilog and rebuild with verilator

@pytest.mark.parametrize("data_width", [64])
@pytest.mark.parametrize("max_reps", [15])
@pytest.mark.parametrize("instr_fifo_depth", [32])
@pytest.mark.parametrize(
    "dim, acc_mem_depth, act_mem_depth, weight_fifo_depth", [
    (8, 32, 32, 32),
])
def test_tpu_standalone(dim, act_mem_depth, acc_mem_depth, weight_fifo_depth, instr_fifo_depth, data_width, max_reps):
    config = TPUConfig(rows=dim, cols=dim, instr_fifo_depth=instr_fifo_depth, weight_fifo_depth=weight_fifo_depth,
        acc_mem_depth=acc_mem_depth, act_mem_depth=act_mem_depth, host_data_width=data_width, max_reps=max_reps)
    dut = TPU(config)

    builddir = Path(__file__).parent / "build"
    builddir.mkdir(exist_ok=True)
    with open(builddir / "tpu.v", 'w') as f:
        f.write(convert(dut, name="TPU", emit_src=False))
    with open(builddir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)

    #FIXME:icarus has some issue with generated always @* + casez
    cocotb_run([builddir / "tpu.v"], "TPU", waves=True, sim="verilator")

@cocotb.test()
async def test_matmul(dut):
    tpu_axi = TPUAxiInterface(dut)
    await tpu_axi.init()
    await tpu_axi.reset()
    with open(Path(__file__).parent / "build" / "config.json") as f:
        config = TPUConfig.fromdict(json.load(f))

    for i in range(3):
        m, k, n = [random.randint(8, 65) for _ in range(3)]
        actfn = random.choice(list(Activation))
        print(f"Run {i} with dims (M, K, N)={m, k, n}")
        actbuf, wbuf, biasbuf, actfn, zd, shamt, qmul, expected = matmul_case(m, k, n, config)
        data_bytes = wbuf + biasbuf + actbuf
        tpu_axi.memory[:len(data_bytes)] = data_bytes

        act_offset = len(wbuf) + len(biasbuf)
        res_offset = len(data_bytes)
        instrs = tpu_matmul(m, k, n, config, zd, shamt.item(), qmul.item(), actfn=actfn,
            a_haddr=act_offset, c_haddr=len(wbuf), d_haddr=res_offset)

        #FIXME: encoding is slow for large number of instrs
        encoded_instrs = [config.isa_layout.const(instr).as_bits() for instr in instrs]
        ibuf = repack(encoded_instrs, config.isa_layout.size, config.host_data_width, aligned=True)
        instr_bytes = [byte for word in ibuf for byte in unpacked(word, config.host_data_width, 8)]
        instr_baseaddr = len(tpu_axi.memory) - len(instr_bytes)
        tpu_axi.memory[instr_baseaddr:] = instr_bytes

        result_size = m * -(-n // config.cols) * -(-config.act_dtype.width * config.rows // config.host_data_width) * config.host_data_width // 8
        await run_tpu(tpu_axi, config, len(instrs), instr_baseaddr)
        result = unpack_activations(tpu_axi.memory[res_offset:res_offset + result_size], m, n, config)
        np.testing.assert_array_equal(result, expected)
