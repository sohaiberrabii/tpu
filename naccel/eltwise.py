import amaranth as am
from amaranth.lib import data, stream
from amaranth.lib.wiring import In, Out, Component
from amaranth.utils import ceil_log2

from naccel.isa import Activation


def clip(x, min, max):
    return am.Mux(x < min, min, am.Mux(x > max, max, x))

class Scaler(Component):
    """((a * b) >> shamt) + zp
    """
    def __init__(self, in_shape, out_shape, max_shift=63):
        assert isinstance(in_shape, data.ArrayLayout) and isinstance(out_shape, data.ArrayLayout) and in_shape.length == out_shape.length

        out_width = out_shape.elem_shape.width
        self.min, self.max = -(1 << out_width - 1), (1 << out_width - 1) - 1
        super().__init__({
            "req": In(stream.Signature(data.StructLayout({
                "a": in_shape, "b": in_shape, "zp": out_shape.elem_shape, "shamt": ceil_log2(max_shift + 1)}))),
            "resp": Out(stream.Signature(out_shape)),
        })
    def elaborate(self, _):
        m = am.Module()
        assert all(isinstance(x.shape(), data.ArrayLayout) for x in [self.req.payload.a, self.req.payload.b, self.resp.payload])

        m.d.comb += [
            self.resp.valid.eq(self.req.valid),
            self.req.ready.eq(self.resp.ready),
        ]

        m.d.comb += [resp.eq(clip(((a * b) >> self.req.payload.shamt) + self.req.payload.zp, self.min, self.max))
            for resp, a, b in zip(self.resp.payload, self.req.payload.a, self.req.payload.b)]
        return m

class ScalerConfig(data.StructLayout):
    def __init__(self, qmul_shape, zp_shape, shamt_width):
        super().__init__({"qmul": qmul_shape, "shamt": shamt_width, "zp": zp_shape})

class ActivationUnit(Component):
    def __init__(self, src_layout, dst_layout, addr_width):
        assert isinstance(src_layout, data.ArrayLayout) and isinstance(dst_layout, data.ArrayLayout)

        self.qmul = am.Signal(src_layout.elem_shape)
        self.shamt = am.Signal(ceil_log2(src_layout.elem_shape.width * 2))
        self.zp = am.Signal(dst_layout.elem_shape)

        self.scaler = Scaler(src_layout, dst_layout, max_shift=src_layout.elem_shape.width * 2 - 1)

        super().__init__({
            "config": In(stream.Signature(ScalerConfig(self.qmul.shape(), self.zp.shape(), self.shamt.shape()), always_ready=True)),
            "req": In(stream.Signature(data.StructLayout({"addr": addr_width, "data": src_layout, "actfn": Activation}))),
            "resp": Out(stream.Signature(data.StructLayout({"addr": addr_width, "data": dst_layout}))),
        })
    def elaborate(self, _):
        m = am.Module()

        with m.If(self.config.ready & self.config.valid):
            m.d.sync += [
                self.qmul.eq(self.config.payload.qmul),
                self.shamt.eq(self.config.payload.shamt),
                self.zp.eq(self.config.payload.zp),
            ]

        activated = am.Signal.like(self.req.payload.data)
        with m.Switch(self.req.payload.actfn):
            with m.Case(Activation.NONE):
                m.d.comb += activated.eq(self.req.payload.data)
            with m.Case(Activation.RELU):
                m.d.comb += [act.eq(am.Mux(p[-1], 0, p)) for act, p in zip(activated, self.req.payload.data)]

        m.submodules.scaler = self.scaler
        m.d.comb += [
            self.scaler.req.valid.eq(self.req.valid),
            self.scaler.req.payload.a.eq(activated),
            self.scaler.req.payload.b.eq(self.qmul.replicate(activated.shape().length)),
            self.scaler.req.payload.shamt.eq(self.shamt),
            self.scaler.req.payload.zp.eq(self.zp),
            self.req.ready.eq(self.scaler.req.ready),
            self.scaler.resp.ready.eq(self.resp.ready),
            self.resp.valid.eq(self.scaler.resp.valid),
            self.resp.payload.data.eq(self.scaler.resp.payload),

            self.resp.payload.addr.eq(self.req.payload.addr),
        ]
        return m
