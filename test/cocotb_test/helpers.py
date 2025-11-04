import inspect
from pathlib import Path

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, ReadWrite, Timer
from cocotb.handle import Immediate
from cocotb.queue import Queue
from cocotb_tools.runner import get_runner

from tpu.isa import Activation
from tpu.sw import *
from tpu.bus import RespType


def cocotb_run(srcs, top, module=None, sim="verilator", always=False, waves=False, timescale=("1ns", "1ps"),
    test_args=[], build_args=[], logtest=False):
    if waves and sim == "verilator":
        build_args += ["--Wno-fatal", "--trace-fst", "--quiet"]
    module = inspect.getmodule(inspect.stack()[1].frame).__name__ if module is None else module
    runner = get_runner(sim)
    simdir = Path(__file__).parent / "sim_build"
    runner.build(sources=srcs, hdl_toplevel=top, build_args=build_args, always=always, waves=waves, build_dir=simdir, timescale=timescale,
        log_file=simdir / "build.log")
    runner.test(hdl_toplevel=top, test_module=module, waves=waves, test_args=test_args, build_dir=simdir, test_dir=simdir,
        log_file=simdir / "test.log" if logtest else None)

async def run_tpu(axi, config, ninstrs, instr_baseaddr):
    assert await axi.read_csr(config.csr_offsets["tpur"])
    xfer_size = min(config.instr_fifo_depth, config.max_reps)
    ins_xfers = -(-ninstrs // xfer_size)
    for i in range(ins_xfers):
        nins = xfer_size if i < ins_xfers - 1 else ninstrs - i * xfer_size
        insaddr = instr_baseaddr + i * xfer_size * aligned_size(config.isa_layout.size, config.host_data_width) // 8
        await axi.write_csr(config.csr_offsets["nins"], nins)
        await axi.write_csr(config.csr_offsets["insadr"], insaddr)
        await axi.write_csr(config.csr_offsets["tpus"], 1)
        while True:
            await RisingEdge(axi.dut.clk)
            if await axi.read_csr(config.csr_offsets["tpur"]):
                break

async def tb_qmatmul(tpu_axi, config, a, n, b_bytes, c_bytes, zd, qmul, shamt, actfn=Activation.RELU):
    m, k = a.shape
    a_bytes = np.array(pack_activations(a, config), dtype=np.uint8)
    data_bytes = np.concatenate([b_bytes, c_bytes, a_bytes])
    tpu_axi.memory[:len(data_bytes)] = data_bytes
    res_offset = len(data_bytes)
    instrs = tpu_matmul(m, k, n, config, zd, shamt.item(), qmul.item(), actfn=actfn,
        a_haddr=len(b_bytes) + len(c_bytes), c_haddr=len(b_bytes), d_haddr=res_offset)

    ibuf = repack([config.isa_layout.const(instr).as_bits() for instr in instrs], config.isa_layout.size, config.host_data_width, aligned=True)
    instr_bytes = [byte for word in ibuf for byte in unpacked(word, config.host_data_width, 8)]
    instr_baseaddr = len(tpu_axi.memory) - len(instr_bytes)
    tpu_axi.memory[instr_baseaddr:] = instr_bytes

    result_size = m * -(-n // config.cols) * -(-config.act_dtype.width * config.rows // config.host_data_width) * config.host_data_width // 8
    await run_tpu(tpu_axi, config, len(instrs), instr_baseaddr)
    return unpack_activations(tpu_axi.memory[res_offset:res_offset + result_size], m, n, config)

class CocotbModel:
    def __init__(self, tpu_axi, config, modelfn, tensorfn):
        self.config = config
        self.tpu_axi = tpu_axi
        with open(modelfn) as fm, open(tensorfn, 'rb') as ft:
            self.ops = json.load(fm)
            self.tensors = ft.read()
        self.layers = [self._parse_op(op) for op in self.ops]

    async def __call__(self, x: np.ndarray):
        for l in self.layers:
            x = await l(x) if inspect.iscoroutinefunction(l) else l(x)
        return x

    def _parse_op(self, op):
        match op["op"]:
            case "fully_connected":
                qparams = {k: np.array(v) for k, v in op["qparams"].items()}
                tensors = {k: self._extract_tensor(v)  for k, v in op["args"].items()}
                n = op["args"]["weight"]["shape"][1]
                return functools.partial(tb_qmatmul, self.tpu_axi, self.config, n=n, b_bytes=tensors["weight"], c_bytes=tensors["bias"],
                    zd=qparams["output_zp"], qmul=qparams["qmul"], shamt=qparams["shamt"], actfn=Activation[op["act"]])

            case "flatten":
                def flatten(x):
                    start_dim, end_dim = op["args"]
                    end_dim = len(x.shape) + end_dim if end_dim < 0 else end_dim
                    return x.reshape(*x.shape[:start_dim], -1, *x.shape[end_dim + 1:])
                return flatten

    def _extract_tensor(self, info):
        return np.frombuffer(self.tensors[info["offset"]:info["offset"] + info["size"]], dtype=np.uint8)

#TODO: refactor or maybe make it generic
class TPUAxiInterface:
    def __init__(self, dut, mem_size=4 * 1024 * 1024):
        self.dut = dut
        self.clock = Clock(dut.clk, 10, unit="ns")

        self.csr_arfifo = Queue()
        self.csr_rfifo  = Queue()
        self.csr_awfifo = Queue()
        self.csr_wfifo  = Queue()
        self.csr_bfifo  = Queue()

        self.bus_arfifo = Queue()
        self.bus_rfifo  = Queue()
        self.bus_awfifo = Queue()
        self.bus_wfifo  = Queue()
        self.bus_bfifo  = Queue()

        self.memory = np.zeros(mem_size, dtype=np.uint8)

    async def reset(self):
        self.dut.rst.set(Immediate(1))
        await Timer(100, unit="ns")
        self.dut.rst.set(Immediate(0))

    async def init(self):
        cocotb.start_soon(self.clock.start(start_high=False))
        cocotb.start_soon(self.csr_arprocess())
        cocotb.start_soon(self.csr_rprocess())
        cocotb.start_soon(self.csr_awprocess())
        cocotb.start_soon(self.csr_wprocess())
        cocotb.start_soon(self.csr_bprocess())

        cocotb.start_soon(self.bus_arprocess())
        cocotb.start_soon(self.memory_read_process())
        cocotb.start_soon(self.bus_rprocess())

        cocotb.start_soon(self.bus_awprocess())
        cocotb.start_soon(self.bus_wprocess())
        cocotb.start_soon(self.memory_write_process())
        cocotb.start_soon(self.bus_bprocess())

    async def read_csr(self, addr):
        await self.csr_arfifo.put(addr)
        return await self.csr_rfifo.get()

    async def write_csr(self, addr, data):
        await self.csr_awfifo.put(addr)
        await self.csr_wfifo.put(data)
        assert await self.csr_bfifo.get() == RespType.OKAY.value

    async def bus_arprocess(self): 
        self.dut.bus__arready.value = 1
        while True:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            if self.dut.bus__arvalid.value:
                await self.bus_arfifo.put({
                    "addr": self.dut.bus__araddr.value.to_unsigned(),
                    "len": self.dut.bus__arlen.value.to_unsigned(),
                    "size": self.dut.bus__arsize.value.to_unsigned(),
                })

    async def memory_read_process(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.bus_arfifo.qsize():
                req = await self.bus_arfifo.get()
                addr = req["addr"]
                size = 1 << req["size"]
                assert addr >= 0 and addr + size * (req["len"] + 1) <= len(self.memory)
                for i in range(0, req["len"] + 1):
                    resp = {"data": packed(self.memory[addr + size * i:addr + size * (i + 1)].tolist(), 8), "last": int(i == req["len"])}
                    await self.bus_rfifo.put(resp)

    async def bus_rprocess(self, timeout=10): 
        self.dut.bus__rvalid.value = 0
        self.dut.bus__rresp.value = RespType.OKAY.value
        while True:
            await RisingEdge(self.dut.clk)
            self.dut.bus__rvalid.value = 0
            if self.bus_rfifo.qsize():
                rpayload = await self.bus_rfifo.get()
                self.dut.bus__rvalid.value = 1
                self.dut.bus__rdata.value = rpayload["data"]
                self.dut.bus__rlast.value = rpayload["last"]
                await ReadWrite()
                wait_cycles = 0
                while self.dut.bus__rready.value == 0:
                    await RisingEdge(self.dut.clk)
                    await ReadWrite()
                    wait_cycles += 1
                    assert wait_cycles < timeout, "timeout waiting bus rready"

    async def bus_awprocess(self): 
        self.dut.bus__awready.value = 1
        while True:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            if self.dut.bus__awvalid.value:
                await self.bus_awfifo.put({
                    "addr": self.dut.bus__awaddr.value.to_unsigned(),
                    "len": self.dut.bus__awlen.value.to_unsigned(),
                    "size": self.dut.bus__awsize.value.to_unsigned(),
                })

    async def bus_wprocess(self): 
        self.dut.bus__wready.value = 1
        while True:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            if self.dut.bus__wvalid.value:
                await self.bus_wfifo.put({
                    "data": self.dut.bus__wdata.value.to_unsigned(),
                    "last": self.dut.bus__wlast.value,
                })

    async def memory_write_process(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.bus_awfifo.qsize() and self.bus_wfifo.qsize():
                req = await self.bus_awfifo.get()
                addr = req["addr"]
                size = 1 << req["size"]
                assert addr >= 0 and addr + size * (req["len"] + 1) <= len(self.memory)

                data = []
                while True:
                    wdata = await self.bus_wfifo.get()
                    data.extend(unpacked(wdata["data"], size * 8, 8))
                    if wdata["last"]:
                        break
                self.memory[addr:addr + size * (req["len"] + 1)] = data
                await self.bus_bfifo.put(RespType.OKAY.value)

    async def bus_bprocess(self, timeout=10): 
        self.dut.bus__bvalid.value = 0
        while True:
            await RisingEdge(self.dut.clk)
            self.dut.bus__bvalid.value = 0
            if self.bus_bfifo.qsize():
                bresp = await self.bus_bfifo.get()
                self.dut.bus__bresp.value  = bresp
                self.dut.bus__bvalid.value = 1
                await ReadWrite()
                wait_cycles = 0
                while self.dut.bus__bready.value == 0:
                    await RisingEdge(self.dut.clk)
                    await ReadWrite()
                    wait_cycles += 1
                    assert wait_cycles < timeout, "timeout waiting bus bready"

    async def csr_arprocess(self, timeout=10): 
        self.dut.ctrl__arvalid.value = 0
        self.dut.ctrl__arprot.value  = 0
        while True:
            await RisingEdge(self.dut.clk)
            self.dut.ctrl__arvalid.value = 0
            if self.csr_arfifo.qsize():
                self.dut.ctrl__arvalid.value = 1
                self.dut.ctrl__araddr.value = await self.csr_arfifo.get()
                await ReadWrite()
                wait_cycles = 0
                while self.dut.ctrl__arready.value == 0:
                    await RisingEdge(self.dut.clk)
                    await ReadWrite()
                    wait_cycles += 1
                    assert wait_cycles < timeout, "timeout waiting ctrl arready"

    async def csr_rprocess(self): 
        self.dut.ctrl__rready.value = 1
        while True:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            if self.dut.ctrl__rvalid.value:
                await self.csr_rfifo.put(self.dut.ctrl__rdata.value)

    async def csr_awprocess(self, timeout=10): 
        self.dut.ctrl__awvalid.value = 0
        self.dut.ctrl__awprot.value  = 0
        while True:
            await RisingEdge(self.dut.clk)
            self.dut.ctrl__awvalid.value = 0
            if self.csr_awfifo.qsize():
                self.dut.ctrl__awvalid.value = 1
                self.dut.ctrl__awaddr.value = await self.csr_awfifo.get()
                await ReadWrite()
                wait_cycles = 0
                while self.dut.ctrl__awready.value == 0:
                    await RisingEdge(self.dut.clk)
                    await ReadWrite()
                    wait_cycles += 1
                    assert wait_cycles < timeout, "timeout waiting ctrl awready"

    async def csr_wprocess(self, timeout=10): 
        self.dut.ctrl__wvalid.value = 1
        while True:
            await RisingEdge(self.dut.clk)
            self.dut.ctrl__wvalid.value = 0
            if self.csr_wfifo.qsize():
                self.dut.ctrl__wvalid.value = 1
                self.dut.ctrl__wdata.value = await self.csr_wfifo.get()
                await ReadWrite()
                wait_cycles = 0
                while self.dut.ctrl__wready.value == 0:
                    await RisingEdge(self.dut.clk)
                    await ReadWrite()
                    wait_cycles += 1
                    assert wait_cycles < timeout, "timeout waiting ctrl awready"

    async def csr_bprocess(self):
        self.dut.ctrl__bready.value = 1
        while True:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            if self.dut.ctrl__bvalid.value:
                await self.csr_bfifo.put(self.dut.ctrl__bresp.value)
