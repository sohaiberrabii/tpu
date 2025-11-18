from dataclasses import dataclass

import amaranth as am
from amaranth.lib.wiring import In, Out, Component, connect, flipped
from amaranth.lib import data, fifo
from amaranth.utils import ceil_log2, exact_log2
from amaranth_soc import csr

from tpu.pe import SystolicArray
from tpu.controller import ExecuteController, PreloadController, ActivationController, LoadController, StoreController
from tpu.memory import Scratchpad, Accumulator, FixedPriorityArbiter
from tpu.isa import ISALayout
from tpu.decoder import InstructionDecoder
from tpu import bus
from tpu.bus import DMAReader, DMAWriter, Arbiter, AXI4Lite, AXI4LiteCSRBridge
from tpu.eltwise import ActivationUnit
from tpu.sw import IntType


@dataclass(frozen=True)
class TPUConfig:
    rows: int = 8
    cols: int = 8
    weight_dtype: IntType = IntType(width=8, signed=True)
    act_dtype: IntType = IntType(width=8, signed=True)
    acc_dtype: IntType = IntType(width=32, signed=True)
    double_buffered: int = True

    act_mem_depth: int = 32
    acc_mem_depth: int = 32
    weight_fifo_depth: int = 32
    instr_fifo_depth: int = 16

    host_data_width: int = 64
    host_addr_width: int = 32

    max_reps: int = 15
    acc_max_reps: int = 4

    csr_addr_width: int = 2
    csr_data_width: int = 32
    csr_offsets = {"tpur": 0x0, "tpus": 0x4, "insadr": 0x8, "nins": 0xc}

    @staticmethod
    def fromdict(config: dict):
        return TPUConfig(**{k: IntType(**v) if k in ["weight_dtype", "act_dtype", "acc_dtype"] else v for k, v in config.items()})
    @property
    def isa_layout(self):
        return ISALayout(self.act_dtype.shape, self.acc_dtype.shape, self.host_addr_width,
            exact_log2(self.act_mem_depth), exact_log2(self.acc_mem_depth), self.max_reps)

    def __post_init__(self):
        assert self.rows == self.cols, "non square sa is not supported yet"
        assert self.host_data_width % 8 == 0

#FIXME: different max reps? the only constrained max_reps is the load due to axi burst even though a burst sequencer solves it
class TPU(Component):
    def __init__(self, config: TPUConfig):
        assert config.rows == config.cols
        self.config = config
        maxreps = config.max_reps
        acc_addr_width = exact_log2(config.acc_mem_depth)
        act_addr_width = exact_log2(config.act_mem_depth)
        a_shape, b_shape, c_shape = config.act_dtype.shape, config.weight_dtype.shape, config.acc_dtype.shape

        self.instr_fifo = fifo.SyncFIFO(width=config.isa_layout.size, depth=config.instr_fifo_depth)
        self.decoder = InstructionDecoder(a_shape, c_shape, config.host_addr_width, act_addr_width, acc_addr_width, maxreps, config.acc_max_reps)
        self.act_mem = Scratchpad(depth=config.act_mem_depth, width=config.act_dtype.width * config.rows)
        self.weight_fifo = fifo.SyncFIFO(width=config.weight_dtype.width * config.cols, depth=config.weight_fifo_depth)
        self.acc_mem = Accumulator(depth=config.acc_mem_depth, width=data.ArrayLayout(c_shape, config.cols))

        self.act_warb = FixedPriorityArbiter(self.act_mem.write.signature.flip())
        self.act_rarb = FixedPriorityArbiter(self.act_mem.read.signature.flip())

        self.load_ctrl = LoadController(config.host_addr_width, act_addr_width, self.act_mem.width, maxreps,
            acc_addr_width, self.acc_mem.width, config.acc_max_reps)
        self.act_warb.add(self.load_ctrl.dst)

        self.sa = SystolicArray(config.rows, config.cols, a_shape, b_shape, c_shape)
        self.actfn = ActivationUnit(self.acc_mem.width, data.ArrayLayout(a_shape, config.rows), act_addr_width)
        self.act_warb.add(self.actfn.write)

        self.ex_ctrl = ExecuteController(
            act_addr_width, acc_addr_width, maxreps, a_shape.width * config.rows, c_shape.width * config.cols, self.sa.latency)
        self.act_rarb.add(self.ex_ctrl.read)

        self.preload_ctrl = PreloadController(b_shape.width * config.cols, config.rows)
        self.activation_ctrl = ActivationController(acc_addr_width, act_addr_width, maxreps, c_shape.width * config.rows)

        self.store_ctrl = StoreController(act_addr_width, config.host_addr_width, self.act_mem.width, maxreps)
        self.act_rarb.add(self.store_ctrl.read)

        self.acc_warb = FixedPriorityArbiter(self.acc_mem.write.signature.flip())
        self.acc_warb.add(self.ex_ctrl.write)
        self.acc_warb.add(self.load_ctrl.acc_dst)

        #FIXME: this can be simplified to a single DMAReader with configurable de-serializer based on destination width
        self.instr_dma_reader  = DMAReader(
            addr_width=config.host_addr_width, data_width=config.host_data_width, size=self.instr_fifo.width, max_repeats=maxreps)
        self.weight_dma_reader = DMAReader(
            addr_width=config.host_addr_width, data_width=config.host_data_width, size=self.weight_fifo.width, max_repeats=maxreps)
        self.act_dma_reader    = DMAReader(
            addr_width=config.host_addr_width, data_width=config.host_data_width, size=self.act_mem.width, max_repeats=maxreps)
        self.acc_dma_reader    = DMAReader(addr_width=config.host_addr_width, data_width=config.host_data_width,
            size=config.acc_dtype.width * config.cols, max_repeats=config.acc_max_reps)
        self.read_arbiter = Arbiter(addr_width=config.host_addr_width, data_width=config.host_data_width, is_read=True)
        for reader in [self.instr_dma_reader, self.weight_dma_reader, self.act_dma_reader, self.acc_dma_reader]:
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
        m.submodules.acc_warb          = self.acc_warb
        m.submodules.act_mem           = self.act_mem
        m.submodules.act_rarb          = self.act_rarb
        m.submodules.act_warb          = self.act_warb
        m.submodules.weight_fifo       = self.weight_fifo
        m.submodules.sa                = self.sa
        m.submodules.ex_ctrl           = self.ex_ctrl
        m.submodules.preload_ctrl      = self.preload_ctrl

        m.submodules.load_ctrl         = self.load_ctrl
        m.submodules.store_ctrl        = self.store_ctrl

        m.submodules.activation_ctrl   = self.activation_ctrl
        m.submodules.actfn             = self.actfn

        m.submodules.instr_dma_reader  = self.instr_dma_reader
        m.submodules.weight_dma_reader = self.weight_dma_reader
        m.submodules.act_dma_reader    = self.act_dma_reader
        m.submodules.acc_dma_reader    = self.acc_dma_reader
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
        connect(m, self.load_ctrl.acc_src_req, self.acc_dma_reader.req)
        connect(m, self.acc_dma_reader.resp, self.load_ctrl.acc_src_resp)

        connect(m, self.decoder.load_req, self.load_ctrl.req)
        connect(m, self.decoder.load_acc_req, self.load_ctrl.acc_req)
        connect(m, self.decoder.preload_req, self.preload_ctrl.req)
        connect(m, self.decoder.ex_req, self.ex_ctrl.req)
        connect(m, self.decoder.activation_req, self.activation_ctrl.req)
        connect(m, self.decoder.scaler_cfg, self.actfn.config)
        m.d.comb += [
            self.decoder.ex_done.eq(self.ex_ctrl.done),
            self.decoder.act_done.eq(self.activation_ctrl.done),
            self.activation_ctrl.actfn_done.eq(self.actfn.done),
        ]

        connect(m, self.act_rarb, self.act_mem.read)
        connect(m, self.act_warb, self.act_mem.write)

        connect(m, self.acc_warb, self.acc_mem.write)
        connect(m, self.acc_mem.read, self.activation_ctrl.src)

        connect(m, self.activation_ctrl.dst, self.actfn.req)

        connect(m, self.weight_fifo.r_stream, self.preload_ctrl.src)
        connect(m, self.weight_dma_reader.resp, self.weight_fifo.w_stream)
        connect(m, self.decoder.weight_load_req, self.weight_dma_reader.req)

        connect(m, self.instr_fifo.r_stream, self.decoder.instr)
        connect(m, self.instr_dma_reader.resp, self.instr_fifo.w_stream)
        m.d.comb += [
            #FIXME: insn = 1, instr dma becomes ready one cycle before r_level is set to 1 leading to fasle tpu ready/done
            self.tpu_ready.f.tpur.r_data.eq(self.instr_dma_reader.req.ready & (self.instr_fifo.r_level == 0)),
            self.instr_dma_reader.req.valid.eq(self.tpu_start.f.tpus.w_stb),
            self.instr_dma_reader.req.payload.addr.eq(self.instr_adr.f.insadr.data),
            self.instr_dma_reader.req.payload.reps.eq(self.ninstrs.f.nins.data),
        ]

        connect(m, self.sa.execute, self.ex_ctrl.exec)
        connect(m, self.sa.preload, self.preload_ctrl.preload)
        return m


if __name__ == '__main__':
    from amaranth.back import verilog

    config = TPUConfig(rows=8, cols=8, max_reps=15, instr_fifo_depth=32, act_mem_depth=32, acc_mem_depth=32, host_data_width=64, weight_fifo_depth=16)
    tpu = TPU(config)
    with open("tpu.v", 'w') as f:
        f.write(verilog.convert(tpu, name="TPU", emit_src=False))
