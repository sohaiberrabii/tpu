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
from tpu.sw import *
from test.cocotb_test.helpers import cocotb_run, TPUAxiInterface, run_tpu
from test.helpers import matmul_case

#TODO: tests could be much faster we don't regen verilog and rebuild with verilator

@pytest.mark.parametrize("data_width", [64])
@pytest.mark.parametrize("max_reps", [15])
@pytest.mark.parametrize("instr_fifo_depth", [16])
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
    tpu_axi = TPUAxiInterface(dut, rand=True)
    await tpu_axi.init()
    await tpu_axi.reset()
    with open(Path(__file__).parent / "build" / "config.json") as f:
        config = TPUConfig.fromdict(json.load(f))

    for i in range(10):
        m, k, n = [random.randint(8, 180) for _ in range(3)]
        print(f"Run {i} with dims (M, K, N)={m, k, n}")
        (instr_baseaddr, ninstrs), (res_offset, result_size), expected = matmul_case(m, k, n, config, tpu_axi.memory)
        await run_tpu(tpu_axi, config, ninstrs, instr_baseaddr)
        result = unpack_activations(tpu_axi.memory[res_offset:res_offset + result_size], m, n, config)
        np.testing.assert_array_equal(result, expected)
