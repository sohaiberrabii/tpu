import amaranth as am
from amaranth.lib.wiring import In, Out, Component, Signature
from amaranth.lib.memory import Memory
from amaranth.lib import data, stream
from amaranth.utils import exact_log2, ceil_log2

from tpu.isa import AccMode

class MemoryReadIO(Signature):
    def __init__(self, addr_shape, data_shape, req_always_ready=False, resp_always_ready=False):
        super().__init__({
            "req": Out(stream.Signature(data.StructLayout({"addr": addr_shape}), always_ready=req_always_ready)),
            "resp": In(stream.Signature(data.StructLayout({"data": data_shape}), always_ready=resp_always_ready)),
        })

class MemoryWriteIO(Signature):
    def __init__(self, addr_shape, data_shape, req_always_ready=False):
        super().__init__({
            "req": Out(stream.Signature(data.StructLayout({"addr": addr_shape, "data": data_shape}), always_ready=req_always_ready)),
        })

class AccWriteIO(Signature):
    def __init__(self, addr_shape, data_shape, always_ready=False):
        super().__init__({
            "req": Out(stream.Signature(data.StructLayout({
                "waddr": addr_shape, "raddr": addr_shape, "data": data_shape, "acc": AccMode}), always_ready=always_ready)),
        })

#NOTE: This is currently not compatible with Accumulator read intf (resp is not necessarily valid 1cycle after valid handshake)
class FixedPriorityArbiter(Component):
    """A generic arbiter for streams.
    Assumes:
    - If it exists, `resp` stream is always ready and it is valid one cycle after req handshake.

    NOTE:
    grant is evaluated combinationally.
    This adds the delay of evaluating grant signal + multiplexer for selecting the payload.
    """
    def __init__(self, sig):
        assert "req" in sig.members and isinstance(sig.members["req"].signature, stream.Signature)
        assert "resp" not in sig.members or (
            isinstance(sig.members["resp"].signature, stream.Signature) and sig.members["resp"].signature.always_ready)
        self.intfs = []
        super().__init__(sig)

    def add(self, intf):
        self.intfs.append(intf)

    def elaborate(self, _):
        m = am.Module()
        grant = am.Signal(ceil_log2(len(self.intfs)))
        if hasattr(self, "resp"):
            grantq = am.Signal.like(grant)
            m.d.sync += grantq.eq(grant)
        for i in reversed(range(len(self.intfs))):
            with m.If(self.intfs[i].req.valid):
                m.d.comb += grant.eq(i)
            with m.If(grant == i):
                m.d.comb += [
                    self.intfs[i].req.ready.eq(self.req.ready),
                    self.req.valid.eq(self.intfs[i].req.valid),
                    self.req.payload.eq(self.intfs[i].req.payload)
                ]
            if hasattr(self, "resp"):
                m.d.comb += [
                    self.intfs[i].resp.valid.eq(self.resp.valid & (grantq == i)),
                    self.intfs[i].resp.payload.eq(self.resp.payload)
                ]
        return m

class Accumulator(Component):
    def __init__(self, depth, width):
        self.width, self.depth, addr_width = width, depth, exact_log2(depth)
        self.write_q = am.Signal(data.StructLayout({"valid": 1, "waddr": addr_width, "data": width, "acc": AccMode}))
        self.resp_q = am.Signal(data.StructLayout({"valid": 1, "payload": width}))

        super().__init__({
            "read":  In(MemoryReadIO(addr_width, width)),
            "write": In(AccWriteIO(addr_width, width, always_ready=True)),
        })

    def elaborate(self, _):
        m = am.Module()
        m.d.sync += [
            self.write_q.valid.eq(self.write.req.valid),
            self.write_q.waddr.eq(self.write.req.payload.waddr),
            self.write_q.data.eq(self.write.req.payload.data),
            self.write_q.acc.eq(self.write.req.payload.acc),
        ]

        m.submodules.mem = self.mem = Memory(shape=self.width, depth=self.depth, init=[])
        wr_port = self.mem.write_port()
        rd_port = self.mem.read_port()

        acc_write = self.write.req.valid & (self.write.req.payload.acc != AccMode.NO)
        ren = acc_write | (self.read.req.valid & self.read.req.ready)
        raddr = am.Mux(acc_write, am.Mux(
            self.write.req.payload.acc == AccMode.SAME, self.write.req.payload.waddr, self.write.req.payload.raddr), self.read.req.payload.addr)

        wen = self.write_q.valid
        waddr = self.write_q.waddr

        same_addr_rdw = am.Signal(1)
        wdata_q = am.Signal.like(wr_port.data)
        m.d.sync += [
            same_addr_rdw.eq(ren & wen & (waddr == raddr)),
            wdata_q.eq(wr_port.data),
        ]

        rdata = am.Signal(self.width)
        m.d.comb += rdata.eq(am.Mux(same_addr_rdw, wdata_q, rd_port.data))

        resp_valid = am.Signal()
        m.d.sync += resp_valid.eq(self.read.req.valid & self.read.req.ready)
        m.d.comb += [
            rd_port.en.eq(ren),
            rd_port.addr.eq(raddr),
            self.read.req.ready.eq((self.read.resp.ready | ~self.resp_q.valid) & ~acc_write),
            self.read.resp.payload.eq(am.Mux(self.resp_q.valid, self.resp_q.payload, rdata)),
            self.read.resp.valid.eq(resp_valid | self.resp_q.valid)
        ]

        with m.If(resp_valid & (~self.read.resp.ready | self.resp_q.valid)):
            m.d.sync += [self.resp_q.valid.eq(1), self.resp_q.payload.eq(rdata)]
        with m.Elif(self.resp_q.valid & self.read.resp.ready):
            m.d.sync += self.resp_q.valid.eq(0)

        if isinstance(self.width, data.ArrayLayout):
            acc_data = am.Cat((x + y)[:self.width.elem_shape.width] for x, y in zip(rdata, self.write_q.data))
        else:
            acc_data = rdata + self.write_q.data
        wdata = am.Mux(self.write_q.acc != AccMode.NO, acc_data, self.write_q.data)
        m.d.comb += [wr_port.en.eq(wen), wr_port.data.eq(wdata), wr_port.addr.eq(waddr)]
        return m

class Scratchpad(Component):
    def __init__(self, depth, width):
        self.width, self.depth, addr_width = width, depth, exact_log2(depth)
        self.mem = Memory(shape=width, depth=depth, init=[])
        super().__init__({
            "read":  In(MemoryReadIO(addr_width, width, req_always_ready=True, resp_always_ready=True)),
            "write": In(MemoryWriteIO(addr_width, width, req_always_ready=True)),
        })

    def elaborate(self, _):
        m = am.Module()
        m.submodules.mem = self.mem

        wr_port = self.mem.write_port()
        m.d.comb += [
            wr_port.en.eq(self.write.req.valid & self.write.req.ready),
            wr_port.addr.eq(self.write.req.payload.addr),
            wr_port.data.eq(self.write.req.payload.data),
        ]

        rd_port = self.mem.read_port()
        m.d.comb += [
            rd_port.en.eq(self.read.req.valid & self.read.req.ready),
            rd_port.addr.eq(self.read.req.payload.addr),
            self.read.resp.payload.data.eq(rd_port.data),
        ]
        m.d.sync += self.read.resp.valid.eq(rd_port.en)
        return m
