import amaranth as am
from amaranth.lib.wiring import In, Out, Component
from amaranth.lib import stream, data
from amaranth.utils import ceil_log2

from naccel.isa import Activation
from naccel.helpers import Shifter

class Request(data.StructLayout):
    def __init__(self, src_addr_width, dst_addr_width, max_repeats, extra=None):
        members = {"src_addr": src_addr_width, "dst_addr": dst_addr_width, "reps": ceil_log2(max_repeats + 1)}
        if extra: 
            members.update(extra)
        super().__init__(members)

class LoadController(Component):
    def __init__(self, src_addr_width, dst_addr_width, width, max_repeats):
        super().__init__({
            "req": In(stream.Signature(Request(src_addr_width, dst_addr_width, max_repeats))),
            "src_req":  Out(stream.Signature(data.StructLayout({"addr": src_addr_width, "reps": ceil_log2(max_repeats + 1)}))),
            "src_resp": In(stream.Signature(data.StructLayout({"data": width, "last": 1}))),
            "dst": Out(stream.Signature(data.StructLayout({"addr": dst_addr_width, "data": width}))),
        })

    def elaborate(self, _):
        m = am.Module()

        m.d.comb += [
            self.req.ready.eq(self.src_req.ready),
            self.src_req.valid.eq(self.req.valid),
            self.src_req.payload.addr.eq(self.req.payload.src_addr),
            self.src_req.payload.reps.eq(self.req.payload.reps),

            self.src_resp.ready.eq(self.dst.ready),
            self.dst.valid.eq(self.src_resp.valid),
            self.dst.payload.data.eq(self.src_resp.payload.data),
        ]

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += self.dst.payload.addr.eq(self.req.payload.dst_addr)
        with m.Elif(self.dst.valid & self.dst.ready & ~self.src_resp.payload.last):
            m.d.sync += self.dst.payload.addr.eq(self.dst.payload.addr + 1)

        return m

class StoreController(Component):
    def __init__(self, src_addr_width, dst_addr_width, width, max_repeats):
        self.repeat_counter = am.Signal(range(max_repeats + 1))
        self.ack_counter = am.Signal(range(max_repeats + 1))
        super().__init__({
            "req": In(stream.Signature(Request(src_addr_width, dst_addr_width, max_repeats))),
            "src_req":  Out(stream.Signature(src_addr_width)),
            "src_resp": In(stream.Signature(width, always_ready=True)), #FIXME:
            "dst_req": Out(stream.Signature(data.StructLayout({"addr": dst_addr_width, "reps": ceil_log2(max_repeats + 1)}))),
            "dst": Out(stream.Signature(data.StructLayout({"data": width, "last": 1}))),
        })

    def elaborate(self, _):
        m = am.Module()

        src_req_done = self.repeat_counter == 0

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.req.payload.reps),
                self.src_req.payload.eq(self.req.payload.src_addr)
            ]
        with m.Elif(self.src_req.valid & self.src_req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.repeat_counter - 1),
                self.src_req.payload.eq(self.src_req.payload + 1)
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
        src_resp_hs = self.src_resp.valid & self.src_resp.ready
        src_resp_q = am.Signal(data.StructLayout({"valid": 1, "payload": len(self.src_resp.payload)}))
        with m.If(src_resp_hs &  ~self.dst.ready & ~src_resp_q.valid):
            m.d.sync += [src_resp_q.valid.eq(1), src_resp_q.payload.eq(self.src_resp.payload)]
        with m.Elif(src_resp_q.valid & self.dst.ready):
            m.d.sync += src_resp_q.valid.eq(0)

        m.d.comb += [
            self.req.ready.eq(src_req_done & (self.ack_counter == 0)),
            self.src_req.valid.eq(~src_req_done & self.dst.ready),

            self.dst_req.valid.eq(self.req.valid),
            self.dst_req.payload.addr.eq(self.req.payload.dst_addr),
            self.dst_req.payload.reps.eq(self.req.payload.reps),

            self.dst.valid.eq(self.src_resp.valid | src_resp_q.valid),
            self.dst.payload.data.eq(am.Mux(src_resp_q.valid, src_resp_q.payload, self.src_resp.payload)),
            self.dst.payload.last.eq(self.ack_counter == 1),
        ]

        return m

class ExecuteRequest(Request):
    def __init__(self, src_addr_width, dst_addr_width, max_repeats):
        super().__init__(src_addr_width, dst_addr_width, max_repeats, extra={"acc": 1, "wsel": 1})

class ExecuteController(Component):
    def __init__(self, src_addr_width, dst_addr_width, max_repeats, input_width, output_width, exec_latency):
        assert exec_latency > 0
        self.repeat_counter = am.Signal(range(max_repeats + 1))

        self.latency = exec_latency + 1
        self.latency_counter = am.Signal(range(self.latency))

        self.valid_shifter = Shifter(1, exec_latency)

        self.req_q = am.Signal(data.StructLayout({"valid": 1, "dst_addr": dst_addr_width, "acc": 1}))
        self.lat_q = am.Signal(data.StructLayout({"count": range(self.latency), "valid": 1}))

        super().__init__({
            "req":          In(stream.Signature(ExecuteRequest(src_addr_width, dst_addr_width, max_repeats))),
            "input_resp":   In(stream.Signature(input_width, always_ready=True)),
            "input_req":    Out(stream.Signature(src_addr_width)),
            "write_output": Out(stream.Signature(data.StructLayout({"addr": dst_addr_width, "data": output_width, "acc": 1}), always_ready=True)),
            "done": Out(1),
        })


    def elaborate(self, _):
        m = am.Module()
        m.submodules.valid_shifter = self.valid_shifter

        last_output = self.latency_counter == 0
        req_read_done = self.repeat_counter == 0

        m.d.comb += [
            self.done.eq(self.req.ready & last_output),
            self.req.ready.eq(req_read_done & ~self.req_q.valid),
            self.input_req.valid.eq(~req_read_done),
            self.valid_shifter.d.eq(self.input_resp.valid),
            self.write_output.valid.eq(self.valid_shifter.q),
        ]

        # source address control

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.req.payload.reps),
                self.input_req.payload.eq(self.req.payload.src_addr),
            ]
        with m.Elif(self.input_req.valid & self.input_req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.repeat_counter - 1),
                self.input_req.payload.eq(self.input_req.payload + 1),
            ]

        # destination address control

        with m.If((self.repeat_counter == 1) & ~last_output):
            m.d.sync += [self.lat_q.count.eq(self.latency_counter), self.lat_q.valid.eq(1)]
        with m.If(self.req.valid & self.req.ready & ~last_output):
            m.d.sync += [
                self.req_q.valid.eq(1),
                self.req_q.dst_addr.eq(self.req.payload.dst_addr),
                self.req_q.acc.eq(self.req.payload.acc), 
            ]

        with m.If(self.lat_q.valid & last_output):
            m.d.sync += [self.latency_counter.eq(self.lat_q.count), self.lat_q.valid.eq(0)]
        with m.Elif((self.repeat_counter ==  1) & last_output):
            m.d.sync += self.latency_counter.eq(self.latency - 1)
        with m.Elif(~last_output): #NOTE: only correct because the pipeline spad -> sa -> acc is not stallable.
            m.d.sync += self.latency_counter.eq(self.latency_counter - 1)

        with m.If(last_output & self.req_q.valid):
            m.d.sync += [
                self.req_q.valid.eq(0),
                self.write_output.payload.addr.eq(self.req_q.dst_addr),
                self.write_output.payload.acc.eq(self.req_q.acc),
            ]
        with m.Elif(last_output & self.req.valid & self.req.ready):
            m.d.sync += [
                self.write_output.payload.addr.eq(self.req.payload.dst_addr),
                self.write_output.payload.acc.eq(self.req.payload.acc),
            ]
        with m.Elif(self.write_output.valid & self.write_output.ready):
            m.d.sync += self.write_output.payload.addr.eq(self.write_output.payload.addr + 1)

        return m

class PreloadController(Component):
    def __init__(self, transfer_unit, reps):
        self.reps = reps
        self.repeat_counter = am.Signal(range(reps + 1))
        super().__init__({
            "req": In(stream.Signature(data.StructLayout({"wsel": 1}))),
            "src": In(stream.Signature(transfer_unit)),
            "dst": Out(stream.Signature(transfer_unit, always_ready=True)),
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
            self.src.ready.eq(self.dst.ready & ~done),
            self.dst.payload.eq(self.src.payload),
            self.dst.valid.eq(self.src.valid),
        ]
        return m

class ActivationRequest(Request):
    def __init__(self, src_addr_width, dst_addr_width, max_repeats):
        super().__init__(src_addr_width, dst_addr_width, max_repeats, extra={"actfn": Activation})

#NOTE: this is not properly pipelined. plus this is basically the same control as executecontroller. 
# Except the latter was designed with the assumption of a fixed latency. 
class ActivationController(Component):
    def __init__(self, src_addr_width, dst_addr_width, max_repeats, width):
        self.repeat_counter = am.Signal(range(max_repeats + 1))
        self.ack_counter = am.Signal(range(max_repeats + 1))

        super().__init__({
            "req":      In(stream.Signature(ActivationRequest(src_addr_width, dst_addr_width, max_repeats))),
            "src_req":  Out(stream.Signature(data.StructLayout({"addr": src_addr_width}))),
            "src_resp": In(stream.Signature(data.StructLayout({"data": width}))),
            "dst":      Out(stream.Signature(data.StructLayout({"addr": dst_addr_width, "data": width, "actfn": Activation}))),
        })
    def elaborate(self, _):
        m = am.Module()

        req_read_done = self.repeat_counter == 0
        req_output_done = self.ack_counter == 0

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.req.payload.reps),
                self.src_req.payload.addr.eq(self.req.payload.src_addr),
                self.dst.payload.addr.eq(self.req.payload.dst_addr),
                self.dst.payload.actfn.eq(self.req.payload.actfn),
            ]
        with m.Elif(self.src_req.valid & self.src_req.ready):
            m.d.sync += [
                self.repeat_counter.eq(self.repeat_counter - 1),
                self.src_req.payload.addr.eq(self.src_req.payload.addr + 1),
            ]

        with m.If(self.req.valid & self.req.ready):
            m.d.sync += [
                self.ack_counter.eq(self.req.payload.reps),
                self.dst.payload.addr.eq(self.req.payload.dst_addr),
            ]
        with m.Elif(self.dst.valid & self.dst.ready):
            m.d.sync += [
                self.ack_counter.eq(self.ack_counter - 1),
                self.dst.payload.addr.eq(self.dst.payload.addr + 1),
            ]

        m.d.comb += [
            self.req.ready.eq(req_read_done & req_output_done),
            self.src_req.valid.eq(~req_read_done),

            self.src_resp.ready.eq(self.dst.ready),
            self.dst.valid.eq(self.src_resp.valid),
            self.dst.payload.data.eq(self.src_resp.payload.data),
        ]
        return m
