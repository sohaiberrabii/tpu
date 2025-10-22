import inspect
from pathlib import Path
from subprocess import Popen

import numpy as np

import cocotb
from cocotb.triggers import RisingEdge, ReadOnly, ReadWrite, Timer
from cocotb.queue import Queue
from cocotb_tools.runner import get_runner

from naccel.bus import RespType
from test.helpers import packed, unpacked

def cocotb_run(srcs, top, module=None, sim="verilator", always=False, waves=False, timescale=("1ns", "1ps"),
    test_args=[], build_args=[]):
    if waves and sim == "verilator":
        build_args += ["--Wno-fatal", "--trace-fst", "--quiet"]
    module = inspect.getmodule(inspect.stack()[1].frame).__name__ if module is None else module
    runner = get_runner(sim)
    simdir = Path(__file__).parent / "sim_build"
    runner.build(sources=srcs, hdl_toplevel=top, build_args=build_args, always=always, waves=waves, build_dir=simdir, timescale=timescale,
        log_file=simdir / "build.log")
    runner.test(hdl_toplevel=top, test_module=module, waves=waves, test_args=test_args, build_dir=simdir, test_dir=simdir,
        log_file=simdir / "test.log")

#TODO: refactor or maybe make it generic
class TPUAxiInterface:
    def __init__(self, dut, mem_size=4 * 1024 * 1024):
        self.dut = dut

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

    async def init(self):
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
                    "len": self.dut.bus__arlen.value.to_unsigned(),
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
