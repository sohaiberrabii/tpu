import amaranth as am
from amaranth.lib import stream, data
from amaranth.lib.wiring import In, Out, Component
from amaranth.utils import ceil_log2

from naccel.isa import Op, ISALayout, LoadFunct, MoveFunct
from naccel.controller import ActivationRequest, Request, ExecuteRequest

class InstructionDecoder(Component):
    def __init__(self, haddr_width, sp_addr_width, acc_addr_width, max_repeats):
        self.instr_layout = ISALayout(haddr_width, sp_addr_width, acc_addr_width, max_repeats)
        super().__init__({
            "instr": In(stream.Signature(self.instr_layout)),

            "load_req":        Out(stream.Signature(Request(haddr_width, sp_addr_width, max_repeats))),
            "weight_load_req": Out(stream.Signature(data.StructLayout({"addr": haddr_width, "reps": ceil_log2(max_repeats + 1)}))),
            "preload_req":     Out(stream.Signature(data.StructLayout({"wsel": 1}))),
            "ex_req":          Out(stream.Signature(ExecuteRequest(sp_addr_width, acc_addr_width, max_repeats))),
            "activation_req":  Out(stream.Signature(ActivationRequest(acc_addr_width, sp_addr_width, max_repeats))),
            "store_req":       Out(stream.Signature(Request(sp_addr_width, haddr_width, max_repeats))),

            "ex_done":         In(1),
        })
    def elaborate(self, _):
        m = am.Module()

        preload_sync = self.instr.valid & (self.instr.payload.op == Op.PRELOAD_SYNC)
        matmul_sync = self.instr.valid & (self.instr.payload.op == Op.MATMUL_SYNC)
        spad_sync = self.instr.valid & (self.instr.payload.op == Op.SPAD_SYNC)

        stall_fetch = (
            (self.load_req.valid & ~self.load_req.ready) |
            (self.preload_req.valid & ~self.preload_req.ready) |
            (self.activation_req.valid & ~self.activation_req.ready) |
            (self.weight_load_req.valid & ~self.weight_load_req.ready) |
            (self.ex_req.valid & ~self.ex_req.ready) |
            (preload_sync & ~self.preload_req.ready) |
            (matmul_sync & ~self.ex_done) |
            (spad_sync & ~(self.load_req.ready & self.activation_req.ready))
        )

        m.d.comb += [
            self.instr.ready.eq(~stall_fetch),

            self.load_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.LOAD) & (self.instr.payload.funct.load == LoadFunct.HOST_ACT)),
            self.load_req.payload.reps.eq(self.instr.payload.reps),
            self.load_req.payload.src_addr.eq(self.instr.payload.addr1.load_store),
            self.load_req.payload.dst_addr.eq(self.instr.payload.addr2),

            self.weight_load_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.LOAD) & 
                (self.instr.payload.funct.load == LoadFunct.HOST_WEIGHT)),
            self.weight_load_req.payload.reps.eq(self.instr.payload.reps),
            self.weight_load_req.payload.addr.eq(self.instr.payload.addr1.load_store),

            self.preload_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.MOVE) & 
                (self.instr.payload.funct.move == MoveFunct.PRELOAD_WEIGHT)),
            self.preload_req.payload.wsel.eq(self.instr.payload.opt.acc_wsel.wsel),

            self.ex_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.MATMUL)),
            self.ex_req.payload.reps.eq(self.instr.payload.reps),
            self.ex_req.payload.src_addr.eq(self.instr.payload.addr1.move_exec),
            self.ex_req.payload.dst_addr.eq(self.instr.payload.addr2),
            self.ex_req.payload.acc.eq(self.instr.payload.opt.acc_wsel.acc),
            self.ex_req.payload.wsel.eq(self.instr.payload.opt.acc_wsel.wsel),

            self.activation_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.MOVE) & (self.instr.payload.funct.move == MoveFunct.ACTIVATE)),
            self.activation_req.payload.src_addr.eq(self.instr.payload.addr1.move_exec),
            self.activation_req.payload.dst_addr.eq(self.instr.payload.addr2),
            self.activation_req.payload.reps.eq(self.instr.payload.reps),
            self.activation_req.payload.actfn.eq(self.instr.payload.opt.actfn),

            self.store_req.valid.eq(self.instr.valid & (self.instr.payload.op == Op.STORE)),
            self.store_req.payload.reps.eq(self.instr.payload.reps),
            self.store_req.payload.src_addr.eq(self.instr.payload.addr2),
            self.store_req.payload.dst_addr.eq(self.instr.payload.addr1.load_store),
        ]
        return m
