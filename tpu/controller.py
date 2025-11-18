import amaranth as am
from amaranth.lib.wiring import In, Out, Component
from amaranth.lib import stream, data
from amaranth.utils import ceil_log2

from tpu.isa import Activation, AccMode
from tpu.helpers import Shifter
from tpu.memory import AccWriteIO, MemoryReadIO, MemoryWriteIO
from tpu.pe import ExecuteIO, PreloadIO


class Request(data.StructLayout):
    def __init__(self, src_addr_width, dst_addr_width, max_repeats, extra=None):
        members = {"src_addr": src_addr_width, "dst_addr": dst_addr_width, "reps": ceil_log2(max_repeats + 1)}
        if extra: 
            members.update(extra)
        super().__init__(members)

class LoadController(Component):
    def __init__(self, src_addr_width, dst_addr_width, width, max_repeats, acc_dst_addr_width, acc_width, acc_max_repeats):
        super().__init__({
            "req": In(stream.Signature(Request(src_addr_width, dst_addr_width, max_repeats))),
            "src_req":  Out(stream.Signature(data.StructLayout({"addr": src_addr_width, "reps": ceil_log2(max_repeats + 1)}))),
            "src_resp": In(stream.Signature(width)),
            "dst": Out(MemoryWriteIO(dst_addr_width, width)),

            "acc_req": In(stream.Signature(Request(src_addr_width, acc_dst_addr_width, acc_max_repeats))),
            "acc_src_req":  Out(stream.Signature(data.StructLayout({"addr": src_addr_width, "reps": ceil_log2(acc_max_repeats + 1)}))),
            "acc_src_resp": In(stream.Signature(acc_width)),
            "acc_dst": Out(AccWriteIO(acc_dst_addr_width, acc_width)),
        })

    def elaborate(self, _):
        m = am.Module()

        m.d.comb += [
            self.req.ready.eq(self.src_req.ready),
            self.src_req.valid.eq(self.req.valid),
            self.src_req.payload.addr.eq(self.req.payload.src_addr),
            self.src_req.payload.reps.eq(self.req.payload.reps),

            self.src_resp.ready.eq(self.dst.req.ready),
            self.dst.req.valid.eq(self.src_resp.valid),
            self.dst.req.payload.data.eq(self.src_resp.payload),
        ]

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += self.dst.req.payload.addr.eq(self.req.payload.dst_addr)
        with m.Elif(self.dst.req.valid & self.dst.req.ready):
            m.d.sync += self.dst.req.payload.addr.eq(self.dst.req.payload.addr + 1)

        m.d.comb += [
            self.acc_req.ready.eq(self.acc_src_req.ready),
            self.acc_src_req.valid.eq(self.acc_req.valid),
            self.acc_src_req.payload.addr.eq(self.acc_req.payload.src_addr),
            self.acc_src_req.payload.reps.eq(self.acc_req.payload.reps),

            self.acc_src_resp.ready.eq(self.acc_dst.req.ready),
            self.acc_dst.req.valid.eq(self.acc_src_resp.valid),
            self.acc_dst.req.payload.data.eq(self.acc_src_resp.payload),
            self.acc_dst.req.payload.acc.eq(AccMode.NO),
            self.acc_dst.req.payload.raddr.eq(0),
        ]

        with m.If(self.acc_req.valid & self.acc_req.ready):
            m.d.sync += self.acc_dst.req.payload.waddr.eq(self.acc_req.payload.dst_addr)
        with m.Elif(self.acc_dst.req.valid & self.acc_dst.req.ready):
            m.d.sync += self.acc_dst.req.payload.waddr.eq(self.acc_dst.req.payload.waddr + 1)
        return m

class StoreController(Component):
    def __init__(self, src_addr_width, dst_addr_width, width, max_repeats):
        self.repeat_counter = am.Signal(range(max_repeats + 1))
        self.ack_counter = am.Signal(range(max_repeats + 1))
        super().__init__({
            "req": In(stream.Signature(Request(src_addr_width, dst_addr_width, max_repeats))),
            "read": In(MemoryReadIO(src_addr_width, width, resp_always_ready=True)),
            "dst_req": Out(stream.Signature(data.StructLayout({"addr": dst_addr_width, "reps": ceil_log2(max_repeats + 1)}))),
            "dst": Out(stream.Signature(data.StructLayout({"data": width, "last": 1}))),
        })

    def elaborate(self, _):
        m = am.Module()

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.req.payload.reps),
                self.read.req.payload.addr.eq(self.req.payload.src_addr)
            ]
        with m.Elif(self.read.req.valid & self.read.req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.repeat_counter - 1),
                self.read.req.payload.addr.eq(self.read.req.payload.addr + 1)
            ]

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += [
                self.ack_counter.eq(self.req.payload.reps),
            ]
        with m.Elif(self.dst.valid & self.dst.ready):
            m.d.sync += [
                self.ack_counter.eq(self.ack_counter - 1),
            ]

        #NOTE: this is needed because the scratchpad read.resp stream is not stallable.
        read_resp_hs = self.read.resp.valid & self.read.resp.ready
        read_resp_q = am.Signal(data.StructLayout({"valid": 1, "payload": len(self.read.resp.payload.data)}))
        with m.If(read_resp_hs &  ~self.dst.ready & ~read_resp_q.valid):
            m.d.sync += [read_resp_q.valid.eq(1), read_resp_q.payload.eq(self.read.resp.payload)]
        with m.Elif(read_resp_q.valid & self.dst.ready):
            m.d.sync += read_resp_q.valid.eq(0)

        src_req_done = self.repeat_counter == 0
        m.d.comb += [
            self.req.ready.eq(src_req_done & (self.ack_counter == 0) & self.dst_req.ready),
            self.read.req.valid.eq(~src_req_done & self.dst.ready),

            self.dst_req.valid.eq(self.req.valid),
            self.dst_req.payload.addr.eq(self.req.payload.dst_addr),
            self.dst_req.payload.reps.eq(self.req.payload.reps),

            self.dst.valid.eq(self.read.resp.valid | read_resp_q.valid),
            self.dst.payload.data.eq(am.Mux(read_resp_q.valid, read_resp_q.payload, self.read.resp.payload)),
            self.dst.payload.last.eq(self.ack_counter == 1),
        ]

        return m

class ExecuteRequest(data.StructLayout):
    def __init__(self, src_addr_width, dst_addr_width, max_repeats):
        super().__init__({"src_addr": src_addr_width, "dst_raddr": dst_addr_width, "dst_waddr": dst_addr_width,
            "reps": ceil_log2(max_repeats + 1), "acc": AccMode, "wsel": 1})

class ExecuteController(Component):
    def __init__(self, src_addr_width, dst_addr_width, max_repeats, input_width, output_width, exec_latency):
        assert exec_latency > 0
        self.repeat_counter = am.Signal(range(max_repeats + 1))

        self.latency = exec_latency + 1
        self.latency_counter = am.Signal(range(self.latency))

        self.valid_shifter = Shifter(1, exec_latency)

        self.req_q = am.Signal(data.StructLayout({"valid": 1, "dst_waddr": dst_addr_width, "dst_raddr": dst_addr_width, "acc": AccMode}))
        self.lat_q = am.Signal(data.StructLayout({"count": range(self.latency), "valid": 1}))

        super().__init__({
            "req":   In(stream.Signature(ExecuteRequest(src_addr_width, dst_addr_width, max_repeats))),
            "read":  Out(MemoryReadIO(src_addr_width, input_width, resp_always_ready=True)),
            "exec":  Out(ExecuteIO(input_width, output_width)),
            "write": Out(AccWriteIO(dst_addr_width, output_width)),
            "done":  Out(1),
        })

    def elaborate(self, _):
        m = am.Module()
        m.submodules.valid_shifter = self.valid_shifter

        last_output = self.latency_counter == 0
        req_read_done = self.repeat_counter == 0

        m.d.comb += [
            self.done.eq(self.req.ready & last_output),
            self.req.ready.eq(req_read_done & ~self.req_q.valid),
            self.read.req.valid.eq(~req_read_done),
            self.valid_shifter.d.eq(self.read.resp.valid),
            self.write.req.valid.eq(self.valid_shifter.q),

            self.exec.a.eq(self.read.resp.payload.data),
            self.write.req.payload.data.eq(self.exec.cout),
        ]

        # source address control

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.req.payload.reps),
                self.read.req.payload.addr.eq(self.req.payload.src_addr),
            ]
        with m.Elif(self.read.req.valid & self.read.req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.repeat_counter - 1),
                self.read.req.payload.addr.eq(self.read.req.payload.addr + 1),
            ]

        # destination address control

        with m.If((self.repeat_counter == 1) & ~last_output):
            m.d.sync += [self.lat_q.count.eq(self.latency_counter), self.lat_q.valid.eq(1)]
        with m.If(self.req.valid & self.req.ready & ~last_output):
            m.d.sync += [
                self.req_q.valid.eq(1),
                self.req_q.dst_waddr.eq(self.req.payload.dst_waddr),
                self.req_q.dst_raddr.eq(self.req.payload.dst_raddr),
                self.req_q.acc.eq(self.req.payload.acc), 
            ]

        with m.If(self.lat_q.valid & last_output):
            m.d.sync += [self.latency_counter.eq(self.latency - 1 - self.lat_q.count), self.lat_q.valid.eq(0)]
        with m.Elif((self.repeat_counter == 1) & last_output):
            m.d.sync += self.latency_counter.eq(self.latency - 1)
        with m.Elif(~last_output): #NOTE: only correct because the pipeline spad -> sa -> acc is not stallable.
            m.d.sync += self.latency_counter.eq(self.latency_counter - 1)

        with m.If(last_output & self.req_q.valid):
            m.d.sync += [
                self.req_q.valid.eq(0),
                self.write.req.payload.waddr.eq(self.req_q.dst_waddr),
                self.write.req.payload.raddr.eq(self.req_q.dst_raddr),
                self.write.req.payload.acc.eq(self.req_q.acc),
            ]
        with m.Elif(last_output & self.req.valid & self.req.ready):
            m.d.sync += [
                self.write.req.payload.raddr.eq(self.req.payload.dst_raddr),
                self.write.req.payload.waddr.eq(self.req.payload.dst_waddr),
                self.write.req.payload.acc.eq(self.req.payload.acc),
            ]
        with m.Elif(self.write.req.valid & self.write.req.ready):
            m.d.sync += self.write.req.payload.waddr.eq(self.write.req.payload.waddr + 1)

        return m

class PreloadController(Component):
    def __init__(self, transfer_unit, reps):
        self.reps = reps
        self.repeat_counter = am.Signal(range(reps + 1))
        super().__init__({
            "req": In(stream.Signature(data.StructLayout({"wsel": 1}))),
            "src": In(stream.Signature(transfer_unit)),
            "preload": Out(PreloadIO(transfer_unit)),
        })
    def elaborate(self, _):
        m = am.Module()
        done = self.repeat_counter == 0

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += self.repeat_counter.eq(self.reps)
        with m.Elif(self.src.valid & self.src.ready):
            m.d.sync += self.repeat_counter.eq(self.repeat_counter - 1)

        m.d.comb += [
            self.req.ready.eq(done),
            self.src.ready.eq(~done),
            self.preload.b.eq(self.src.payload),
            self.preload.load.eq(self.src.valid & ~done),
        ]
        return m

class ActivationRequest(Request):
    def __init__(self, src_addr_width, dst_addr_width, max_repeats):
        super().__init__(src_addr_width, dst_addr_width, max_repeats, extra={"actfn": Activation})

class ActivationController(Component):
    def __init__(self, src_addr_width, dst_addr_width, max_repeats, width):
        self.repeat_counter = am.Signal(range(max_repeats + 1))
        self.ack_counter = am.Signal(range(max_repeats + 1))

        super().__init__({
            "req": In(stream.Signature(ActivationRequest(src_addr_width, dst_addr_width, max_repeats))),
            "src": Out(MemoryReadIO(src_addr_width, width)),
            "dst": Out(stream.Signature(data.StructLayout({"addr": dst_addr_width, "data": width, "actfn": Activation}))),

            "actfn_done": In(1),
            "done":       Out(1),
        })

    def elaborate(self, _):
        m = am.Module()

        req_read_done = self.repeat_counter == 0
        req_output_done = self.ack_counter == 0

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.req.payload.reps),
                self.src.req.payload.addr.eq(self.req.payload.src_addr),
            ]
        with m.Elif(self.src.req.valid & self.src.req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.repeat_counter - 1),
                self.src.req.payload.addr.eq(self.src.req.payload.addr + 1),
            ]

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += [
                self.ack_counter.eq(self.req.payload.reps),
                self.dst.payload.addr.eq(self.req.payload.dst_addr),
                self.dst.payload.actfn.eq(self.req.payload.actfn),
            ]
        with m.Elif(self.dst.valid & self.dst.ready):
            m.d.sync += [
                self.ack_counter.eq(self.ack_counter - 1),
                self.dst.payload.addr.eq(self.dst.payload.addr + 1),
            ]

        m.d.comb += [
            self.done.eq(self.req.ready & self.actfn_done),
            self.req.ready.eq(req_read_done & req_output_done),
            self.src.req.valid.eq(~req_read_done & self.dst.ready), #acc mem is not stallable, only issue request if dst is also ready

            self.src.resp.ready.eq(self.dst.ready),
            self.dst.valid.eq(self.src.resp.valid),
            self.dst.payload.data.eq(self.src.resp.payload.data),
        ]
        return m
