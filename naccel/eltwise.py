import amaranth as am
from amaranth.lib import data, stream
from amaranth.lib.wiring import In, Out, Component

from naccel.isa import Activation

class ActivationUnit(Component):
    def __init__(self, src_layout, dst_layout, addr_width, shift_amt=8):
        assert isinstance(src_layout, data.ArrayLayout) and isinstance(dst_layout, data.ArrayLayout)
        assert src_layout.elem_shape == am.signed(32)
        self.dst_width = dst_layout.elem_shape.width
        self.shift_amt = shift_amt
        self.max = (1 << self.dst_width) - 1
        super().__init__({
            "req": In(stream.Signature(data.StructLayout({"addr": addr_width, "data": src_layout, "actfn": Activation}))),
            "resp": Out(stream.Signature(data.StructLayout({"addr": addr_width, "data": dst_layout}))),
        })
    def elaborate(self, _):
        m = am.Module()
        activated = am.Signal.like(self.req.payload.data)
        with m.Switch(self.req.payload.actfn):
            with m.Case(Activation.NOP):
                m.d.comb += activated.eq(self.req.payload.data)
            with m.Case(Activation.RELU):
                m.d.comb += activated.eq(am.Cat(am.Mux(p[-1], 0, p).as_unsigned() for p in self.req.payload.data))
        m.d.comb += [
            self.resp.payload.data.eq(am.Cat(am.Mux((shifted := (p >> self.shift_amt)) > self.max, self.max, shifted)[:self.dst_width]
                for p in activated)),
            self.resp.payload.addr.eq(self.req.payload.addr),
            self.resp.valid.eq(self.req.valid),
            self.req.ready.eq(self.resp.ready),
        ]
        return m
