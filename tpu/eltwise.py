import amaranth as am
from amaranth.lib import data, stream
from amaranth.lib.wiring import In, Out, Component
from amaranth.utils import ceil_log2

from tpu.isa import Activation
from tpu.helpers import Shifter


def clip(x, min, max):
    return am.Mux(x < min, min, am.Mux(x > max, max, x))

class Scaler(Component):
    """Reg((Reg(a * b) >> shamt) + zp)
    """
    def __init__(self, in_shape, out_shape, max_shift=63):
        assert isinstance(in_shape, data.ArrayLayout) and (
                isinstance(out_shape, data.ArrayLayout) and in_shape.length == out_shape.length)

        out_width = out_shape.elem_shape.width
        self.min, self.max = -(1 << out_width - 1), (1 << out_width - 1) - 1
        self.muls = am.Signal(data.ArrayLayout(
            am.Shape(in_shape.elem_shape.width * 2, signed=in_shape.elem_shape.signed), in_shape.length))

        super().__init__({
            "req": In(stream.Signature(data.StructLayout({
                "a": in_shape, "b": in_shape, "zp": out_shape.elem_shape, "shamt": ceil_log2(max_shift + 1)}))),
            "resp": Out(stream.Signature(out_shape)),
            "done": Out(1),
        })
    def elaborate(self, _):
        m = am.Module()
        assert all(isinstance(x.shape(), data.ArrayLayout) for x in [self.req.payload.a, self.req.payload.b, self.resp.payload])

        valid_muls = am.Signal()
        with m.If(~self.resp.valid | self.resp.ready):
            m.d.sync += self.resp.valid.eq(valid_muls)
            with m.If(valid_muls):
                m.d.sync += [resp.eq(clip((p >> self.req.payload.shamt) + self.req.payload.zp, self.min, self.max))
                    for resp, p in zip(self.resp.payload, self.muls)]

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += [p.eq(a * b) for p, a, b in zip(self.muls, self.req.payload.a, self.req.payload.b)]
            m.d.sync += valid_muls.eq(1)
        with m.Elif(~self.resp.valid | self.resp.ready):
            m.d.sync += valid_muls.eq(0)

        m.d.comb += [
            self.req.ready.eq(~valid_muls | ~self.resp.valid | self.resp.ready),
            self.done.eq(~valid_muls & ~self.resp.valid),
        ]

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
        self.sync_shifter = Shifter(addr_width, 2) #TODO: make this generic along with scaler pipeline depth

        super().__init__({
            "config": In(stream.Signature(ScalerConfig(self.qmul.shape(), self.zp.shape(), self.shamt.shape()), always_ready=True)),
            "req": In(stream.Signature(data.StructLayout({"addr": addr_width, "data": src_layout, "actfn": Activation}))),
            "resp": Out(stream.Signature(data.StructLayout({"addr": addr_width, "data": dst_layout}))),
            "done": Out(1),
        })

    def elaborate(self, _):
        m = am.Module()
        with m.If(self.config.ready & self.config.valid):
            m.d.sync += [
                self.qmul.eq(self.config.payload.qmul),
                self.shamt.eq(self.config.payload.shamt),
                self.zp.eq(self.config.payload.zp),
            ]

        m.submodules.scaler = self.scaler
        m.submodules.sync_shifter = self.sync_shifter

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += [
                self.sync_shifter.d.eq(self.req.payload.addr),
                self.scaler.req.valid.eq(1),
            ]
            with m.Switch(self.req.payload.actfn):
                with m.Case(Activation.NONE):
                    m.d.sync += self.scaler.req.payload.a.eq(self.req.payload.data)
                with m.Case(Activation.RELU):
                    m.d.sync += [act.eq(am.Mux(p[-1], 0, p)) for act, p in zip(self.scaler.req.payload.a, self.req.payload.data)]
        with m.Elif(self.scaler.req.ready):
            m.d.sync += self.scaler.req.valid.eq(0)

        m.d.comb += [
            self.sync_shifter.en.eq(self.scaler.req.ready),
            self.resp.payload.addr.eq(self.sync_shifter.q),

            self.scaler.req.payload.b.eq(self.qmul.replicate(self.req.payload.data.shape().length)),
            self.scaler.req.payload.shamt.eq(self.shamt),
            self.scaler.req.payload.zp.eq(self.zp),
            self.scaler.resp.ready.eq(self.resp.ready),

            self.req.ready.eq(self.scaler.req.ready),
            self.resp.valid.eq(self.scaler.resp.valid),
            self.resp.payload.data.eq(self.scaler.resp.payload),
            self.done.eq(self.scaler.done & ~self.scaler.req.valid),
        ]
        return m
