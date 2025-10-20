from dataclasses import dataclass

import amaranth as am
from amaranth.lib.wiring import In, Out, Component, connect, flipped
from amaranth.lib import data, fifo
from amaranth.utils import ceil_log2, exact_log2
from amaranth_soc import csr

from naccel.pe import PE, SystolicDelay
from naccel.controller import ExecuteController, PreloadController, ActivationController, LoadController, StoreController
from naccel.memory import Scratchpad, Accumulator
from naccel.decoder import InstructionDecoder
from naccel import bus
from naccel.bus import DMAReader, DMAWriter, Arbiter, AXI4Lite, AXI4LiteCSRBridge
from naccel.eltwise import ActivationUnit


@dataclass(frozen=True)
class IntType:
    width: int
    signed: bool

    @property
    def shape(self):
        return am.signed(self.width) if self.signed else am.unsigned(self.width)

@dataclass(frozen=True)
class TPUConfig:
    rows: int = 2
    cols: int = 2
    weight_dtype: IntType = IntType(width=8, signed=True)
    act_dtype: IntType = IntType(width=8, signed=False)
    acc_dtype: IntType = IntType(width=32, signed=True)
    double_buffered: int = True

    act_mem_depth: int = 8
    acc_mem_depth: int = 8
    weight_fifo_depth: int = 8
    instr_fifo_depth: int = 8

    host_data_width: int = 32
    host_addr_width: int = 32

    max_reps: int = 15

    csr_addr_width: int = 2
    csr_data_width: int = 32
    csr_offsets = {"tpur": 0x0, "tpus": 0x4, "insadr": 0x8, "nins": 0xc}


#FIXME: different max reps? the only constrained max_reps is the load due to axi burst even though a burst sequencer solves it
class TPU(Component):
    def __init__(self, config: TPUConfig):
        assert config.rows == config.cols
        self.config = config
        maxreps = config.max_reps
        acc_addr_width = exact_log2(config.acc_mem_depth)
        act_addr_width = exact_log2(config.act_mem_depth)
        a_shape, b_shape, c_shape = config.act_dtype.shape, config.weight_dtype.shape, config.acc_dtype.shape

        self.decoder = InstructionDecoder(config.host_addr_width, act_addr_width, acc_addr_width, maxreps)

        self.acc_mem = Accumulator(depth=config.acc_mem_depth, width=data.ArrayLayout(c_shape, config.cols))
        self.act_mem = Scratchpad(depth=config.act_mem_depth, width=config.act_dtype.width * config.rows)
        self.weight_fifo = fifo.SyncFIFO(width=config.weight_dtype.width * config.cols, depth=config.weight_fifo_depth)
        self.instr_fifo = fifo.SyncFIFO(width=self.decoder.instr.payload.shape().size, depth=config.instr_fifo_depth)

        self.load_ctrl = LoadController(config.host_addr_width, act_addr_width, self.act_mem.width, maxreps)
        self.act_mem.add_writer(self.load_ctrl.dst)

        self.sa = SystolicDelay(PE(a_shape, b_shape, c_shape, latency=1).tile(config.rows, config.cols), "a")
        self.actfn = ActivationUnit(self.acc_mem.width, data.ArrayLayout(a_shape, config.rows), act_addr_width)
        self.act_mem.add_writer(self.actfn.resp)

        self.ex_ctrl = ExecuteController(
            act_addr_width, acc_addr_width, maxreps, a_shape.width * config.rows, c_shape.width * config.cols, self.sa.latency(("a", "cout")))
        self.act_mem.add_reader(self.ex_ctrl.input_req, self.ex_ctrl.input_resp)

        self.preload_ctrl = PreloadController(b_shape.width * config.cols, config.rows)
        self.activation_ctrl = ActivationController(acc_addr_width, act_addr_width, maxreps, c_shape.width * config.rows)

        self.store_ctrl = StoreController(act_addr_width, config.host_addr_width, self.act_mem.width, maxreps)
        self.act_mem.add_reader(self.store_ctrl.src_req, self.store_ctrl.src_resp)

        self.instr_dma_reader  = DMAReader(
            addr_width=config.host_addr_width, data_width=config.host_data_width, size=self.instr_fifo.width, max_repeats=maxreps, lastctrl=False)
        self.weight_dma_reader = DMAReader(
            addr_width=config.host_addr_width, data_width=config.host_data_width, size=self.weight_fifo.width, max_repeats=maxreps, lastctrl=False)
        self.act_dma_reader    = DMAReader(
            addr_width=config.host_addr_width, data_width=config.host_data_width, size=self.act_mem.width, max_repeats=maxreps)
        self.read_arbiter = Arbiter(addr_width=config.host_addr_width, data_width=config.host_data_width, is_read=True)
        for reader in [self.instr_dma_reader, self.weight_dma_reader, self.act_dma_reader]:
            self.read_arbiter.add(reader.bus)

        self.act_dma_writer = DMAWriter(
            addr_width=config.host_addr_width, data_width=config.host_data_width, size=self.act_mem.width, max_repeats=maxreps)

        csrs = csr.Builder(addr_width=config.csr_addr_width, data_width=config.csr_data_width)
        self.tpu_ready = csrs.add("tpu_ready", csr.Register({"tpur": csr.Field(csr.action.R, 1)}, access="r"), offset=TPUConfig.csr_offsets["tpur"])
        self.tpu_start = csrs.add("tpu_start", csr.Register({"tpus": csr.Field(csr.action.W, 0)}, access="w"), offset=TPUConfig.csr_offsets["tpus"])
        self.ninstrs   = csrs.add(
            "ninstrs", csr.Register({"nins": csr.Field(csr.action.RW, ceil_log2(config.instr_fifo_depth + 1))}, access="rw"),
            offset=TPUConfig.csr_offsets["nins"])
        self.instr_adr = csrs.add(
            "instr_adr", csr.Register({"insadr": csr.Field(csr.action.RW, config.host_addr_width)}, access="rw"), offset=TPUConfig.csr_offsets["insadr"])
        self.csr_bridge = csr.Bridge(csrs.as_memory_map())
        self.axi4lite_bridge = AXI4LiteCSRBridge(self.csr_bridge.bus)

        super().__init__({
            "bus": Out(bus.Signature(addr_width=config.host_addr_width, data_width=config.host_data_width)),
            "ctrl": In(AXI4Lite(addr_width=config.csr_addr_width + 2, data_width=config.csr_data_width))
        })

    def elaborate(self, _):
        m = am.Module()

        m.submodules.csr_bridge        = self.csr_bridge
        m.submodules.axi4lite_bridge   = self.axi4lite_bridge
        connect(m, flipped(self.ctrl), self.axi4lite_bridge.bus)

        m.submodules.instr_fifo        = self.instr_fifo
        m.submodules.decoder           = self.decoder
        m.submodules.acc_mem           = self.acc_mem
        m.submodules.act_mem           = self.act_mem
        m.submodules.weight_fifo       = self.weight_fifo
        m.submodules.sa                = self.sa
        m.submodules.ex_ctrl           = self.ex_ctrl
        m.submodules.preload_ctrl      = self.preload_ctrl
        m.submodules.activation_ctrl   = self.activation_ctrl
        m.submodules.load_ctrl         = self.load_ctrl
        m.submodules.store_ctrl        = self.store_ctrl

        m.submodules.actfn             = self.actfn

        m.submodules.instr_dma_reader  = self.instr_dma_reader
        m.submodules.weight_dma_reader = self.weight_dma_reader
        m.submodules.act_dma_reader    = self.act_dma_reader
        m.submodules.read_arbiter      = self.read_arbiter

        m.submodules.act_dma_writer    = self.act_dma_writer
        connect(m, self.decoder.store_req, self.store_ctrl.req)
        connect(m, self.store_ctrl.dst_req, self.act_dma_writer.req)
        connect(m, self.store_ctrl.dst, self.act_dma_writer.src)
        connect(m, flipped(self.bus), self.read_arbiter.bus)
        #FIXME: connect write and read channels separately
        # connect(m, flipped(self.bus), self.act_dma_writer.bus)
        m.d.comb += [
            self.bus.awvalid.eq(self.act_dma_writer.bus.awvalid),
            self.bus.awaddr.eq(self.act_dma_writer.bus.awaddr),
            self.bus.awlen.eq(self.act_dma_writer.bus.awlen),
            self.act_dma_writer.bus.awready.eq(self.bus.awready),

            self.bus.wvalid.eq(self.act_dma_writer.bus.wvalid),
            self.bus.wdata.eq(self.act_dma_writer.bus.wdata),
            self.bus.wlast.eq(self.act_dma_writer.bus.wlast),
            self.act_dma_writer.bus.wready.eq(self.bus.wready),

            self.act_dma_writer.bus.bvalid.eq(self.bus.bvalid),
            self.bus.bready.eq(self.act_dma_writer.bus.bready),
        ]

        connect(m, self.load_ctrl.src_req, self.act_dma_reader.req)
        connect(m, self.act_dma_reader.resp, self.load_ctrl.src_resp)

        connect(m, self.decoder.load_req, self.load_ctrl.req)
        connect(m, self.decoder.preload_req, self.preload_ctrl.req)
        connect(m, self.decoder.ex_req, self.ex_ctrl.req)
        connect(m, self.decoder.activation_req, self.activation_ctrl.req)
        m.d.comb += self.decoder.ex_done.eq(self.ex_ctrl.done)

        connect(m, self.acc_mem.write, self.ex_ctrl.write_output)
        connect(m, self.activation_ctrl.src_req, self.acc_mem.read.req)
        connect(m, self.acc_mem.read.resp, self.activation_ctrl.src_resp)
        connect(m, self.activation_ctrl.dst, self.actfn.req)

        connect(m, self.weight_fifo.r_stream, self.preload_ctrl.src)
        connect(m, self.weight_dma_reader.resp, self.weight_fifo.w_stream)
        connect(m, self.decoder.weight_load_req, self.weight_dma_reader.req)

        connect(m, self.instr_fifo.r_stream, self.decoder.instr)
        connect(m, self.instr_dma_reader.resp, self.instr_fifo.w_stream)
        m.d.comb += [
            self.tpu_ready.f.tpur.r_data.eq(self.instr_dma_reader.req.ready & (self.instr_fifo.r_level == 0)),
            self.instr_dma_reader.req.valid.eq(self.tpu_start.f.tpus.w_stb),
            self.instr_dma_reader.req.payload.addr.eq(self.instr_adr.f.insadr.data),
            self.instr_dma_reader.req.payload.reps.eq(self.ninstrs.f.nins.data),
        ]

        m.d.comb += [
            self.sa.b.eq(self.preload_ctrl.dst.payload),
            self.sa.load.eq(self.preload_ctrl.dst.valid),

            self.sa.a.eq(self.ex_ctrl.input_resp.payload),
            self.ex_ctrl.write_output.payload.data.eq(self.sa.cout),
        ]
        return m


if __name__ == '__main__':
    from amaranth.back import verilog

    config = TPUConfig(rows=8, cols=8, max_reps=15, instr_fifo_depth=32, act_mem_depth=32, acc_mem_depth=32, host_data_width=64, weight_fifo_depth=16)
    tpu = TPU(config)
    with open("tpu.v", 'w') as f:
        f.write(verilog.convert(tpu, name="TPU", emit_src=False))
