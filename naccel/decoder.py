import amaranth as am
from amaranth.lib import stream, data
from amaranth.lib.wiring import In, Out, Component
from amaranth.utils import ceil_log2

from naccel.isa import Op, ISALayout, LoadFunct, MoveFunct
from naccel.controller import ActivationRequest, Request, ExecuteRequest
from naccel.eltwise import ScalerConfig

class InstructionDecoder(Component):
    def __init__(self, act_shape, acc_shape, haddr_width, sp_addr_width, acc_addr_width, max_repeats, acc_max_repeats):
        self.instr_layout = ISALayout(act_shape, acc_shape, haddr_width, sp_addr_width, acc_addr_width, max(max_repeats, acc_max_repeats))
        super().__init__({
            "instr": In(stream.Signature(self.instr_layout)),

            "load_req":        Out(stream.Signature(Request(haddr_width, sp_addr_width, max_repeats))),
            "load_acc_req":    Out(stream.Signature(Request(haddr_width, acc_addr_width, acc_max_repeats))),
            "weight_load_req": Out(stream.Signature(data.StructLayout({"addr": haddr_width, "reps": ceil_log2(max_repeats + 1)}))),
            "preload_req":     Out(stream.Signature(data.StructLayout({"wsel": 1}))),
            "ex_req":          Out(stream.Signature(ExecuteRequest(sp_addr_width, acc_addr_width, max_repeats))),
            "activation_req":  Out(stream.Signature(ActivationRequest(acc_addr_width, sp_addr_width, max_repeats))),
            "scaler_cfg":      Out(stream.Signature(ScalerConfig(acc_shape, act_shape, ceil_log2(acc_shape.width * 2)))),
            "store_req":       Out(stream.Signature(Request(sp_addr_width, haddr_width, max_repeats))),

            "ex_done":         In(1),
        })
    def elaborate(self, _):
        m = am.Module()

        preload_sync = self.instr.valid & (self.instr.payload.op == Op.PRELOAD_SYNC)
        matmul_sync = self.instr.valid & (self.instr.payload.op == Op.MATMUL_SYNC)
        spad_sync = self.instr.valid & (self.instr.payload.op == Op.SPAD_SYNC)
        acc_sync = self.instr.valid & (self.instr.payload.op == Op.ACC_SYNC)

        stall_fetch = (
            (self.load_req.valid & ~self.load_req.ready) |
            (self.load_acc_req.valid & ~self.load_acc_req.ready) |
            (self.preload_req.valid & ~self.preload_req.ready) |
            (self.activation_req.valid & ~self.activation_req.ready) |
            (self.scaler_cfg.valid & ~self.scaler_cfg.ready) |
            (self.weight_load_req.valid & ~self.weight_load_req.ready) |
            (self.ex_req.valid & ~self.ex_req.ready) |
            (self.store_req.valid & ~self.store_req.ready) |
            (preload_sync & ~self.preload_req.ready) |
            (matmul_sync & ~self.ex_done) |
            (spad_sync & ~(self.load_req.ready & self.activation_req.ready & self.store_req.ready)) |
            (acc_sync & ~(self.ex_done & self.load_acc_req.ready))
        )

        m.d.comb += [
            self.instr.ready.eq(~stall_fetch),

            self.load_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.LOAD) & (self.instr.payload.funct.load == LoadFunct.HOST_ACT)),
            self.load_req.payload.reps.eq(self.instr.payload.args.exec.reps),
            self.load_req.payload.src_addr.eq(self.instr.payload.args.exec.addr1.load_store),
            self.load_req.payload.dst_addr.eq(self.instr.payload.args.exec.addr2.act),

            self.load_acc_req.valid.eq(
                self.instr.valid & (self.instr.payload.op == Op.LOAD) & (self.instr.payload.funct.load == LoadFunct.HOST_BIAS)),
            self.load_acc_req.payload.reps.eq(self.instr.payload.args.exec.reps),
            self.load_acc_req.payload.src_addr.eq(self.instr.payload.args.exec.addr1.load_store),
            self.load_acc_req.payload.dst_addr.eq(self.instr.payload.args.exec.addr2.acc),

            self.weight_load_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.LOAD) & 
                (self.instr.payload.funct.load == LoadFunct.HOST_WEIGHT)),
            self.weight_load_req.payload.reps.eq(self.instr.payload.args.exec.reps),
            self.weight_load_req.payload.addr.eq(self.instr.payload.args.exec.addr1.load_store),

            self.preload_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.MOVE) & 
                (self.instr.payload.funct.move == MoveFunct.PRELOAD_WEIGHT)),
            self.preload_req.payload.wsel.eq(self.instr.payload.args.exec.opt.acc_wsel.wsel),

            self.ex_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.MATMUL)),
            self.ex_req.payload.reps.eq(self.instr.payload.args.exec.reps),
            self.ex_req.payload.src_addr.eq(self.instr.payload.args.exec.addr2.act),
            self.ex_req.payload.dst_raddr.eq(self.instr.payload.args.exec.addr1.move_exec.raddr),
            self.ex_req.payload.dst_waddr.eq(self.instr.payload.args.exec.addr1.move_exec.waddr),
            self.ex_req.payload.acc.eq(self.instr.payload.args.exec.opt.acc_wsel.acc),
            self.ex_req.payload.wsel.eq(self.instr.payload.args.exec.opt.acc_wsel.wsel),

            self.activation_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.MOVE) & (self.instr.payload.funct.move == MoveFunct.ACTIVATE)),
            self.activation_req.payload.src_addr.eq(self.instr.payload.args.exec.addr1.move_exec.raddr),
            self.activation_req.payload.dst_addr.eq(self.instr.payload.args.exec.addr2.act),
            self.activation_req.payload.reps.eq(self.instr.payload.args.exec.reps),
            self.activation_req.payload.actfn.eq(self.instr.payload.args.exec.opt.actfn),

            self.scaler_cfg.valid.eq(self.instr.valid & (self.instr.payload.op == Op.SCALER_CONFIG)),
            self.scaler_cfg.payload.qmul.eq(self.instr.payload.args.config.qmul),
            self.scaler_cfg.payload.shamt.eq(self.instr.payload.args.config.shamt),
            self.scaler_cfg.payload.zp.eq(self.instr.payload.args.config.zp),

            self.store_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.STORE)),
            self.store_req.payload.reps.eq(self.instr.payload.args.exec.reps),
            self.store_req.payload.src_addr.eq(self.instr.payload.args.exec.addr2.act),
            self.store_req.payload.dst_addr.eq(self.instr.payload.args.exec.addr1.load_store),
        ]
        return m
