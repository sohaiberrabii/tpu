import inspect
import random
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
    instrs = tpu_matmul(m, k, n, config, zd.item(), shamt.item(), qmul.item(), actfn=actfn,
        a_haddr=len(b_bytes) + len(c_bytes), c_haddr=len(b_bytes), d_haddr=res_offset)

    ibuf = repack([config.isa_layout.const(instr).as_bits() for instr in instrs], config.isa_layout.size, config.host_data_width, aligned=True)
    instr_bytes = [byte for word in ibuf for byte in unpacked(word, config.host_data_width, 8)]
    instr_baseaddr = len(tpu_axi.memory) - len(instr_bytes)
    tpu_axi.memory[instr_baseaddr:] = instr_bytes

    result_size = m * ceildiv(n, config.cols) * aligned_size(config.act_dtype.width * config.rows, config.host_data_width) // 8
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

# stream -> queue
async def stream_consumer(clk, queue, ready, valid, payload, rand=False):
    while True:
        await RisingEdge(clk)
        ready.value = random.getrandbits(1) if rand else 1
        await ReadOnly()
        if ready.value and valid.value:
            await queue.put({k: v.value for k, v in payload.items()})

# queue -> stream, waits for ready
async def stream_producer(clk, queue, ready, valid, payload, timeout=1000):
    while True:
        await RisingEdge(clk)
        valid.value = 0
        if queue.qsize():
            valid.value = 1
            for k, v in (await queue.get()).items():
                payload[k].value = v
            await ReadWrite()
            wait_cycles = 0
            while ready.value == 0:
                await RisingEdge(clk)
                await ReadWrite()
                wait_cycles += 1
                assert wait_cycles < timeout, "timeout waiting ready"

class TPUAxiInterface:
    def __init__(self, dut, mem_size=4 * 1024 * 1024, rand=False):
        self.dut = dut
        self.clock = Clock(dut.clk, 10, unit="ns")

        self.memory = np.zeros(mem_size, dtype=np.uint8)
        self.rand = rand

        self.producer_intfs = {
            "ctrl__aw": {"queue": Queue(), "payload": ["addr", "prot"]},
            "ctrl__w": {"queue": Queue(), "payload": ["data"]},
            "ctrl__ar": {"queue": Queue(), "payload": ["addr", "prot"]},
            "bus__r": {"queue": Queue(), "payload": ["resp", "data", "last"]},
            "bus__b": {"queue": Queue(), "payload": ["resp"]}
        }

        self.consumer_intfs = {
            "ctrl__r": {"queue": Queue(), "payload": ["data"]},
            "ctrl__b": {"queue": Queue(), "payload": ["resp"]},
            "bus__ar": {"queue": Queue(), "payload": ["addr", "len", "size"]},
            "bus__aw": {"queue": Queue(), "payload": ["addr", "len", "size"]},
            "bus__w": {"queue": Queue(), "payload": ["data", "last"]}
        }

    async def reset(self):
        self.dut.rst.set(Immediate(1))
        await Timer(100, unit="ns")
        self.dut.rst.set(Immediate(0))

    async def init(self):
        cocotb.start_soon(self.clock.start(start_high=False))
        cocotb.start_soon(self.memory_read_process())
        cocotb.start_soon(self.memory_write_process())

        for k, v in self.producer_intfs.items():
            cocotb.start_soon(stream_producer(self.dut.clk, v["queue"], **self._stream_payload(k, v["payload"])))

        for k, v in self.consumer_intfs.items():
            cocotb.start_soon(stream_consumer(self.dut.clk, v["queue"], **self._stream_payload(k, v["payload"]), rand=self.rand))

    async def read_csr(self, addr):
        await self.producer_intfs["ctrl__ar"]["queue"].put({"addr": addr, "prot": 0})
        return (await self.consumer_intfs["ctrl__r"]["queue"].get())["data"]

    async def write_csr(self, addr, data):
        await self.producer_intfs["ctrl__aw"]["queue"].put({"addr": addr, "prot": 0})
        await self.producer_intfs["ctrl__w"]["queue"].put({"data": data})
        assert (await self.consumer_intfs["ctrl__b"]["queue"].get())["resp"] == RespType.OKAY.value

    async def memory_read_process(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.consumer_intfs["bus__ar"]["queue"].qsize():
                req = await self.consumer_intfs["bus__ar"]["queue"].get()
                addr = req["addr"].to_unsigned()
                size = 1 << req["size"].to_unsigned()
                length = req["len"].to_unsigned()
                assert addr >= 0 and addr + size * (length + 1) <= len(self.memory)
                for i in range(0, length + 1):
                    await self.producer_intfs["bus__r"]["queue"].put({
                        "resp": RespType.OKAY.value,
                        "data": packed(self.memory[addr + size * i:addr + size * (i + 1)].tolist(), 8),
                        "last": int(i == length),
                    })

    async def memory_write_process(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.consumer_intfs["bus__aw"]["queue"].qsize() and self.consumer_intfs["bus__w"]["queue"].qsize():
                req = await self.consumer_intfs["bus__aw"]["queue"].get()
                addr = req["addr"].to_unsigned()
                size = 1 << req["size"].to_unsigned()
                length = req["len"].to_unsigned()
                assert addr >= 0 and addr + size * (length + 1) <= len(self.memory)

                data = []
                while True:
                    wdata = await self.consumer_intfs["bus__w"]["queue"].get()
                    data.extend(unpacked(wdata["data"].to_unsigned(), size * 8, 8))
                    if wdata["last"]:
                        break
                self.memory[addr:addr + size * (length + 1)] = data
                await self.producer_intfs["bus__b"]["queue"].put({"resp": RespType.OKAY.value})

    def _stream_payload(self, name, payload):
        hs_sigs = {k: getattr(self.dut, name + k) for k in ["ready", "valid"]}
        return {"payload": {sn: getattr(self.dut, name + sn) for sn in payload}, **hs_sigs}
