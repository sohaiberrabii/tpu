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
from cocotb.handle import Immediate
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer, RisingEdge

from amaranth.back.verilog import convert

from naccel.isa import Activation, ISALayout
from naccel.tpu import TPU, TPUConfig
from test.cocotb_test.helpers import cocotb_run, TPUAxiInterface
from test.helpers import *

#TODO: tests could be much faster we don't regen verilog and rebuild with verilator

@pytest.mark.parametrize("data_width", [64])
@pytest.mark.parametrize("max_reps", [15])
@pytest.mark.parametrize("instr_fifo_depth", [32])
@pytest.mark.parametrize(
    "dim, acc_mem_depth, act_mem_depth, weight_fifo_depth", [
    (8, 32, 32, 16),
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
async def tb(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start(start_high=False))
    tpu_axi = TPUAxiInterface(dut)
    await tpu_axi.init()
    dut.rst.set(Immediate(1))
    await Timer(15, unit="ns")
    dut.rst.set(Immediate(0))

    with open(Path(__file__).parent / "build" / "config.json") as f:
        config = TPUConfig.fromdict(json.load(f))

    m, k, n = [random.randint(4, 4) for _ in range(3)]
    actfn = random.choice(list(Activation))
    actbuf, wbuf, expected = matmul_case(m, k, n, config, actfn=actfn)
    data_bytes = [byte for word in wbuf + actbuf for byte in unpacked(word, config.host_data_width, 8)]
    tpu_axi.memory[:len(data_bytes)] = data_bytes

    act_baseaddr = len(wbuf) * config.host_data_width // 8
    result_baseaddr = len(data_bytes)
    instrs = tpu_matmul(m, k, n, config, actfn=actfn, weight_haddr=0, act_haddr=act_baseaddr, d_haddr=result_baseaddr)
    ibuf = repack([config.isa_layout.const(instr).as_bits() for instr in instrs], config.isa_layout.size, config.host_data_width, aligned=True)
    instr_bytes = [byte for word in ibuf for byte in unpacked(word, config.host_data_width, 8)] 
    instr_baseaddr = len(tpu_axi.memory) - len(instr_bytes) 
    tpu_axi.memory[instr_baseaddr:] = instr_bytes

    result_size = m * -(-n // config.cols) * -(-config.act_dtype.width * config.rows // config.host_data_width) * config.host_data_width // 8
    assert len(data_bytes) + len(instr_bytes) + result_size < len(tpu_axi.memory)

    await run_tpu(tpu_axi, config, len(instrs), instr_baseaddr)
    await ClockCycles(dut.clk, 100, rising=True)
    expected_bytes = [byte for word in expected for byte in unpacked(word, config.host_data_width, 8)]
    np.array_equal(tpu_axi.memory[result_baseaddr:result_baseaddr + result_size], expected_bytes)

async def run_tpu(axi, config, ninstrs, instr_baseaddr):
    xfer_size = min(config.instr_fifo_depth, config.max_reps)
    ins_xfers = -(-ninstrs // xfer_size)
    for i in range(ins_xfers):
        await RisingEdge(axi.dut.clk)
        nins = xfer_size if i < ins_xfers - 1 else ninstrs - i * xfer_size
        insaddr = instr_baseaddr + i * xfer_size * -(-config.isa_layout.size // config.host_data_width) * config.host_data_width // 8
        if await axi.read_csr(config.csr_offsets["tpur"]) == 1:
            await axi.write_csr(config.csr_offsets["nins"], nins)
            await axi.write_csr(config.csr_offsets["insadr"], insaddr)
            await axi.write_csr(config.csr_offsets["tpus"], 1)
