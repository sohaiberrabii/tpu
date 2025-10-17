import amaranth as am
from amaranth.lib.wiring import In, Out, Component
from amaranth.lib import data

class Shifter(Component):
    def __init__(self, shape, length):
        assert length >= 0
        self.shape, self.length = shape, length
        self.regs = am.Signal(data.ArrayLayout(shape, length))
        super().__init__({"en": In(1, init=1), "d": In(shape), "q": Out(shape)})

    def elaborate(self, _):
        m = am.Module()
        if self.length == 0:
            m.d.comb += self.q.eq(self.d)
        else:
            m.d.comb += self.q.eq(self.regs[-1])
            with m.If(self.en):
                m.d.sync += self.regs.eq(am.Cat(self.d, self.regs[:-1]))
        return m

