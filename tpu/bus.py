from enum import Enum

import amaranth as am
from amaranth.lib.wiring import In, Out, Component, connect, flipped
from amaranth.lib import wiring, stream, data
from amaranth.utils import exact_log2, ceil_log2


class BurstType(Enum):
    FIXED = 0b00
    INCR  = 0b01
    WRAP  = 0b10

class RespType(Enum):
    OKAY   = 0b00
    EXOKAY = 0b01
    SLVERR = 0b10
    DECERR = 0b11

class AXI4Lite(wiring.Signature):
    def __init__(self, *, addr_width, data_width):
        assert data_width % 8 == 0
        super().__init__({
            # Write address channel
            "awaddr":  Out(addr_width),
            "awprot":  Out(3),
            "awvalid": Out(1),
            "awready": In(1),

            # Write data channel
            "wdata":   Out(data_width),
            "wstrb":   Out(data_width // 8),
            "wvalid":  Out(1),
            "wready":  In(1),

            # Write response channel
            "bresp":   In(RespType),
            "bvalid":  In(1),
            "bready":  Out(1),

            # Read address channel
            "araddr":  Out(addr_width),
            "arprot":  Out(3),
            "arvalid": Out(1),
            "arready": In(1),

            # Read data channel
            "rdata":   In(data_width),
            "rresp":   In(RespType),
            "rvalid":  In(1),
            "rready":  Out(1),
        })

class AXI4LiteCSRBridge(wiring.Component):
    def __init__(self, csr_bus, *, data_width=None):
        assert csr_bus.data_width == 32
        self.csr_bus = csr_bus
        data_width = csr_bus.data_width if data_width is None else data_width

        # ratio  = data_width // csr_bus.data_width
        axi4lite_sig = AXI4Lite(addr_width=csr_bus.addr_width + 2, data_width=data_width)
        super().__init__({"bus": In(axi4lite_sig)})

    def elaborate(self, _):
        m = am.Module()

        aw_hs = self.bus.awvalid & self.bus.awready
        w_hs = self.bus.wvalid & self.bus.wready
        b_hs = self.bus.bvalid & self.bus.bready
        ar_hs = self.bus.arvalid & self.bus.arready
        r_hs = self.bus.rvalid & self.bus.rready

        write_ready = self.bus.awvalid & self.bus.wvalid & (~self.bus.bvalid | self.bus.bready)

        m.d.comb += [
            self.bus.awready.eq(write_ready),
            self.bus.wready.eq(write_ready),
            self.csr_bus.w_stb.eq(aw_hs & w_hs),
            self.csr_bus.w_data.eq(self.bus.wdata),
            self.csr_bus.addr.eq(am.Mux(self.csr_bus.w_stb, self.bus.awaddr >> 2, self.bus.araddr >> 2)),

            self.csr_bus.r_stb.eq(ar_hs),
            self.bus.rdata.eq(self.csr_bus.r_data),
            self.bus.arready.eq((~self.bus.rvalid | self.bus.rready) & ~self.csr_bus.w_stb),
        ]

        with m.If(self.csr_bus.w_stb):
            m.d.sync += self.bus.bvalid.eq(1)
        with m.Elif(b_hs):
            m.d.sync += self.bus.bvalid.eq(0)

        with m.If(self.csr_bus.r_stb):
            m.d.sync += self.bus.rvalid.eq(1)
        with m.Elif(r_hs):
            m.d.sync += self.bus.rvalid.eq(0)

        return m 

class Signature(wiring.Signature):
    """Single channel AXI4 interface signature.
    """
    def __init__(self, *, addr_width, data_width):
        if data_width % 8 != 0 or data_width < 8 or data_width > 1024:
            raise ValueError(f'invalid data_width: {data_width}')
        self.addr_width  = addr_width
        self.data_width  = data_width

        members = {
            # AW Channel
            "awid":     Out(1),
            "awaddr":   Out(addr_width),
            "awlen":    Out(4), # 8 for AXI4
            "awsize":   Out(3, init=exact_log2(data_width // 8)),
            "awburst":  Out(BurstType, init=BurstType.INCR),
            "awlock":   Out(2), # 1 for AXI4 or axi4lite
            "awcache":  Out(4),
            "awprot":   Out(3),
            "awvalid":  Out(1),
            "awready":  In(1),

            "awqos":    Out(4),
            # "awregion": Out(4),

            # W Channel
            "wid":     Out(1),
            "wdata":    Out(data_width),
            "wstrb":    Out(data_width // 8, init=(1 << data_width // 8) - 1),
            "wlast":    Out(1),
            "wvalid":   Out(1),
            "wready":   In(1),

            # B Channel
            "bid":      In(1),
            "bresp":    In(RespType),
            "bvalid":   In(1),
            "bready":   Out(1),

            # AR Channel
            "arid":     Out(1),
            "araddr":   Out(addr_width),
            "arlen":    Out(4), # 8 for AXI4
            "arsize":   Out(3, init=exact_log2(data_width // 8)),
            "arburst":  Out(BurstType, init=BurstType.INCR),
            "arlock":   Out(2), # 1 for axi4 or axi4lite
            "arcache":  Out(4),
            "arprot":   Out(3),
            "arvalid":  Out(1),
            "arready":  In(1),

            "arqos":    Out(4),
            # "arregion": Out(4),

            # R Channel
            "rid":      In(1),
            "rdata":    In(data_width),
            "rresp":    In(RespType),
            "rlast":    In(1),
            "rvalid":   In(1),
            "rready":   Out(1),
        }
        super().__init__(members)

class Serializer(Component):
    def __init__(self, *, src_width, dst_width):
        self.nbeats = -(-src_width // dst_width)

        self.pisoreg = am.Signal(data.ArrayLayout(dst_width, self.nbeats)) #TODO:maybe send first input directly?
        self.beat_counter = am.Signal(range(self.nbeats + 1))

        super().__init__({
            "src": In(stream.Signature(data.StructLayout({"data": src_width, "last": 1}))),
            "dst": Out(stream.Signature(data.StructLayout({"data": dst_width, "last": 1})))
        })

    def elaborate(self, _):
        m = am.Module()

        last_q = am.Signal()
        done = self.beat_counter == 0
        last_beat = self.beat_counter == 1

        with m.If(self.src.valid & self.src.ready):
            m.d.sync += [
                self.beat_counter.eq(self.nbeats),
                self.pisoreg.eq(self.src.payload.data),
                last_q.eq(self.src.payload.last),
            ]
        with m.Elif(~done & self.dst.ready):
            m.d.sync += [
                self.beat_counter.eq(self.beat_counter - 1),
                self.pisoreg.eq(self.pisoreg[1:]),
            ]
            with m.If(last_q & last_beat):
                m.d.sync += last_q.eq(0)

        m.d.comb += [
            self.src.ready.eq(done),
            self.dst.valid.eq(~done),
            self.dst.payload.data.eq(self.pisoreg[0]),
            self.dst.payload.last.eq(last_q & last_beat),
        ]
        return m


class DMAWriter(Component):

    def __init__(self, *, addr_width, data_width, size, max_repeats):
        nbeats = -(-size // data_width)

        self.shamt = exact_log2(nbeats)
        self.serializer = Serializer(src_width=size, dst_width=data_width)

        super().__init__({
            "req": In(stream.Signature(data.StructLayout({"addr": addr_width, "reps": ceil_log2(max_repeats + 1)}))),
            "src": In(stream.Signature(data.StructLayout({"data": size, "last": 1}))),
            "bus": Out(Signature(addr_width=addr_width, data_width=data_width)),
        })
        assert ceil_log2(max_repeats << self.shamt) <= len(self.bus.awlen), f"{max_repeats}, {self.shamt}, {len(self.bus.awlen)}"

    def elaborate(self, _):
        m = am.Module()

        m.submodules.serializer = self.serializer
        connect(m, flipped(self.src), self.serializer.src)
        m.d.comb += [
            self.bus.wdata.eq(self.serializer.dst.payload.data),
            self.bus.wlast.eq(self.serializer.dst.payload.last),

            self.bus.awaddr.eq(self.req.payload.addr),
            self.bus.awlen.eq((self.req.payload.reps << self.shamt) - 1),
            self.bus.awburst.eq(BurstType.INCR),

            self.bus.bready.eq(1),
        ]

        with m.FSM():
            with m.State("SEND_AW"):
                m.d.comb += [
                    self.req.ready.eq(self.bus.awready),
                    self.bus.awvalid.eq(self.req.valid),
                ]
                with m.If(self.bus.awvalid & self.bus.awready):
                    m.next = "SEND_W"

            with m.State("SEND_W"):
                m.d.comb += [
                    self.serializer.dst.ready.eq(self.bus.wready),
                    self.bus.wvalid.eq(self.serializer.dst.valid),
                ]
                with m.If(self.bus.wlast & self.bus.wvalid & self.bus.wready):
                    m.next = "WAIT_B"
            with m.State("WAIT_B"):
                with m.If(self.bus.bvalid & self.bus.bready):
                    m.next = "SEND_AW"
        return m

class Deserializer(Component):
    def __init__(self, *, src_width, dst_width):
        self.nbeats = -(-dst_width // src_width)
        self.siporeg = am.Signal(data.ArrayLayout(src_width, self.nbeats))
        self.beat_counter = am.Signal(range(self.nbeats + 1))
        super().__init__({"src": In(stream.Signature(src_width)), "dst": Out(stream.Signature(dst_width))})

    def elaborate(self, _):
        m = am.Module()
        
        done = self.beat_counter == self.nbeats

        with m.If(self.src.valid & self.src.ready):
            m.d.sync += [
                self.beat_counter.eq(am.Mux(done, 1, self.beat_counter + 1)),
                self.siporeg.eq(am.Cat(self.siporeg[1:], self.src.payload)),
            ]
        with m.Elif(self.dst.valid & self.dst.ready):
            m.d.sync += self.beat_counter.eq(0)

        m.d.comb += [
            self.src.ready.eq(~done | self.dst.ready),
            self.dst.valid.eq(done),
            self.dst.payload.eq(self.siporeg),
        ]
        return m

class DMAReader(Component):
    def __init__(self, *, addr_width, data_width, size, max_repeats):
        self.nbeats = -(-size // data_width)

        self.shamt = exact_log2(self.nbeats)
        self.deserializer = Deserializer(src_width=data_width, dst_width=size)

        super().__init__({
            "req": In(stream.Signature(data.StructLayout({"addr": addr_width, "reps": ceil_log2(max_repeats + 1)}))),
            "resp": Out(stream.Signature(size)),
            "bus": Out(Signature(addr_width=addr_width, data_width=data_width)),
        })
        assert ceil_log2(max_repeats << self.shamt) <= len(self.bus.arlen), f"{max_repeats}, {self.shamt}, {len(self.bus.arlen)}"

    def elaborate(self, _):
        m = am.Module()

        m.submodules.deserializer = self.deserializer
        m.d.comb += [
            self.resp.valid.eq(self.deserializer.dst.valid),
            self.resp.payload.eq(self.deserializer.dst.payload),
            self.deserializer.dst.ready.eq(self.resp.ready),
        ]

        m.d.comb += [
            self.deserializer.src.payload.eq(self.bus.rdata),
            self.bus.arburst.eq(BurstType.INCR),
            self.bus.rready.eq(self.deserializer.src.ready),
        ]

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.req.ready.eq(1)
                with m.If(self.req.valid & self.req.ready):
                    m.d.sync += [
                        self.bus.araddr.eq(self.req.payload.addr),
                        self.bus.arlen.eq((self.req.payload.reps << self.shamt) - 1),
                    ]
                    m.next = "SEND_AR"

            with m.State("SEND_AR"):
                m.d.comb += self.bus.arvalid.eq(1)
                with m.If(self.bus.arvalid & self.bus.arready):
                    m.next = "WAIT_R"

            with m.State("WAIT_R"):
                m.d.comb += self.deserializer.src.valid.eq(self.bus.rvalid)
                with m.If(self.bus.rlast & self.bus.rvalid & self.bus.rready):
                    m.next = "IDLE"
        return m

class RoundRobin(Component):
    def __init__(self, num_masters):
        super().__init__({"requests": In(num_masters), "busy": In(1), "grant": Out(num_masters)})

    def elaborate(self, _):
        m = am.Module()
        with m.If(~self.busy):
            with m.Switch(self.grant):
                for i in range(len(self.requests)):
                    with m.Case(i):
                        for pred in reversed(range(i)):
                            with m.If(self.requests[pred]):
                                m.d.sync += self.grant.eq(pred)
                        for succ in reversed(range(i + 1, len(self.requests))):
                            with m.If(self.requests[succ]):
                                m.d.sync += self.grant.eq(succ)
        return m

class Arbiter(Component):
    def __init__(self, *, addr_width, data_width, is_read=False):
        self.is_read = is_read 
        self.masters = []

        super().__init__({"bus": Out(Signature(addr_width=addr_width, data_width=data_width))})

    def add(self, master):
        self.masters.append(master)

    def elaborate(self, _):
        m = am.Module()

        m.submodules.arb = arb = RoundRobin(len(self.masters))

        if self.is_read:
            assert_busy = self.bus.arvalid & self.bus.arready
            deassert_busy = self.bus.rvalid & self.bus.rready & self.bus.rlast
            requests = am.Cat(mstr.arvalid for mstr in self.masters)
        else:
            assert_busy = self.bus.awvalid & self.bus.awready
            deassert_busy = self.bus.bvalid & self.bus.bready
            requests = am.Cat(mstr.awvalid for mstr in self.masters)

        bus_busy = am.Signal()
        with m.If(assert_busy):
            m.d.sync += bus_busy.eq(1)
        with m.Elif(deassert_busy):
            m.d.sync += bus_busy.eq(0)

        m.d.comb += [arb.requests.eq(requests), arb.busy.eq(bus_busy | assert_busy)]

        with m.Switch(arb.grant):
            for i, mstr in enumerate(self.masters):
                with m.Case(i):
                    if self.is_read:
                        # READ
                        m.d.comb += [
                            self.bus.araddr.eq(mstr.araddr),
                            self.bus.arlen.eq(mstr.arlen),
                            self.bus.arsize.eq(mstr.arsize),
                            self.bus.arburst.eq(mstr.arburst),
                            self.bus.arvalid.eq(mstr.arvalid),
                            mstr.arready.eq(self.bus.arready),

                            mstr.rdata.eq(self.bus.rdata),
                            mstr.rresp.eq(self.bus.rresp),
                            mstr.rlast.eq(self.bus.rlast),
                            mstr.rvalid.eq(self.bus.rvalid),
                            self.bus.rready.eq(mstr.rready),
                        ]
                    else:
                        # WRITE
                        m.d.comb += [
                            self.bus.awaddr.eq(mstr.awaddr),
                            self.bus.awlen.eq(mstr.awlen),
                            self.bus.awsize.eq(mstr.awsize),
                            self.bus.awburst.eq(mstr.awburst),
                            self.bus.awvalid.eq(mstr.awvalid),
                            mstr.awready.eq(self.bus.awready & ~arb.busy),

                            self.bus.wdata.eq(mstr.wdata),
                            self.bus.wstrb.eq(mstr.wstrb),
                            self.bus.wlast.eq(mstr.wlast),
                            self.bus.wvalid.eq(mstr.wvalid),
                            mstr.wready.eq(self.bus.wready),

                            mstr.bresp.eq(self.bus.bresp),
                            mstr.bvalid.eq(self.bus.bvalid),
                            self.bus.bready.eq(mstr.bready),
                        ]
        return m
