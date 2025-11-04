import amaranth as am
from amaranth.lib.wiring import In, Out, Component, Signature
from amaranth.lib.memory import Memory
from amaranth.lib import data, stream
from amaranth.utils import exact_log2, ceil_log2

from naccel.isa import AccMode

class Accumulator(Component):
    def __init__(self, depth, width):
        self.width, self.depth, addr_width = width, depth, exact_log2(depth)
        self.write_q = am.Signal(data.StructLayout({"valid": 1, "waddr": addr_width, "data": width, "acc": AccMode}))
        self.resp_q = am.Signal(data.StructLayout({"valid": 1, "payload": width}))

        super().__init__({
            "write": In(stream.Signature(data.StructLayout({
                "waddr": addr_width, "raddr": addr_width, "data": width, "acc": AccMode}), always_ready=True)),
            "read": Out(Signature({
                "req": In(stream.Signature(data.StructLayout({"addr": addr_width}))),
                "resp": Out(stream.Signature(data.StructLayout({"data": width}))),
            }))
        })

    def elaborate(self, _):
        m = am.Module()
        m.d.sync += [
            self.write_q.valid.eq(self.write.valid),
            self.write_q.waddr.eq(self.write.payload.waddr),
            self.write_q.data.eq(self.write.payload.data),
            self.write_q.acc.eq(self.write.payload.acc),
        ]

        m.submodules.mem = self.mem = Memory(shape=self.width, depth=self.depth, init=[])
        wr_port = self.mem.write_port()
        rd_port = self.mem.read_port()

        acc_write = self.write.valid & (self.write.payload.acc != AccMode.NO)
        ren = acc_write | (self.read.req.valid & self.read.req.ready)
        raddr = am.Mux(acc_write, am.Mux(
            self.write.payload.acc == AccMode.SAME, self.write.payload.waddr, self.write.payload.raddr), self.read.req.payload.addr)

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

#FIXME: resp intf of readers is not stallable
class Scratchpad(Component):
    def __init__(self, depth, width):
        self.width, self.depth = width, depth
        self.mem = Memory(shape=width, depth=depth, init=[])

        super().__init__({})
        self.writers, self.readers = [], []

    def elaborate(self, _):
        m = am.Module()
        m.submodules.mem = self.mem

        wr_port = self.mem.write_port()
        m.submodules.wr_arbiter = wr_arbiter = FixedPriorityArbiter(len(self.writers))
        w_grant_q = am.Signal.like(wr_arbiter.grant) #NOTE: upstream deserializer causes a comb cycle
        m.d.sync += w_grant_q.eq(wr_arbiter.grant)
        for i, wintf in enumerate(self.writers):
            m.d.comb += wr_arbiter.req[i].eq(wintf.valid)
            with m.If(w_grant_q == i):
                m.d.comb += [wintf.ready.eq(1), wr_port.en.eq(wintf.valid), wr_port.data.eq(wintf.payload.data), wr_port.addr.eq(wintf.payload.addr)]

        rd_port = self.mem.read_port()
        m.submodules.rd_arbiter = rd_arbiter = FixedPriorityArbiter(len(self.readers))
        for i, (req, resp) in enumerate(self.readers):
            granted = rd_arbiter.grant == i
            m.d.comb += [rd_arbiter.req[i].eq(req.valid), req.ready.eq(granted), resp.payload.eq(rd_port.data)]
            m.d.sync += resp.valid.eq(granted & req.valid)
            with m.If(granted):
                m.d.comb += [rd_port.addr.eq(req.payload), rd_port.en.eq(req.valid)]

        return m

    def add_writer(self, intf):
        self.writers.append(intf)

    def add_reader(self, req_intf, resp_intf):
        self.readers.append((req_intf, resp_intf))

class FixedPriorityArbiter(Component):
    def __init__(self, n):
        super().__init__({"req": In(n), "grant": Out(ceil_log2(n))})

    def elaborate(self, _):
        m = am.Module()
        for i in reversed(range(len(self.req))):
            with m.If(self.req[i]):
                m.d.comb += self.grant.eq(i)
        return m
