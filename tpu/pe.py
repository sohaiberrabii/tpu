from __future__ import annotations
import amaranth as am
from amaranth.lib.wiring import In, Out, Component, Signature
from amaranth.lib import data

from tpu.helpers import Shifter


class ExecuteIO(Signature):
    def __init__(self, input_width, output_width):
        super().__init__({"a": Out(input_width), "cout": In(output_width)})

class PreloadIO(Signature):
    def __init__(self, width):
        super().__init__({"load": Out(1), "b": Out(width)})

class SystolicArray(Component):
    def __init__(self, rows, cols, a_shape, b_shape, c_shape):
        self.sa = SystolicDelay(PE(a_shape, b_shape, c_shape, latency=1).tile(rows, cols), "a")
        super().__init__({
            "preload": In(PreloadIO(b_shape.width * cols)),
            "execute": In(ExecuteIO(a_shape.width * rows, c_shape.width * cols)),
        })
    def elaborate(self, _):
        m = am.Module()
        m.submodules.sa = self.sa
        m.d.comb += [
            self.sa.b.eq(self.preload.b),
            self.sa.load.eq(self.preload.load),

            self.sa.a.eq(self.execute.a),
            self.execute.cout.eq(self.sa.cout),
        ]
        return m

    @property
    def latency(self): return self.sa.latency(("a", "cout"))

class PE(Component):
    def __init__(self, a_shape, b_shape, c_shape, latency=0):
        assert latency <= 1, f"PE impl only supports a latency <= 1, got {latency}"
        self._latency = latency
        self.a_shape, self.b_shape, self.c_shape = a_shape, b_shape, c_shape
        super().__init__({
            "a": In(a_shape),
            "b": In(b_shape),
            "c": In(c_shape),
            "load": In(1),
            "aout": Out(a_shape),
            "bout": Out(b_shape),
            "cout": Out(c_shape),
        })

    def elaborate(self, _):
        m = am.Module()
        with m.If(self.load):
            m.d.sync += self.bout.eq(self.b)
        m.d.comb += self.aout.eq(self.a)

        mac = self.cout.eq(self.a * self.bout + self.c)
        if self._latency == 1:
            m.d.sync += mac
        else:
            m.d.comb += mac
        return m

    def copy(self):
        return self.__class__(self.a_shape, self.b_shape, self.c_shape, latency=self._latency)

    def tile(self, rows, cols):
        return Tile(Tile(self, cols, [("a", "aout")], ["b", "bout", "c", "cout"], ["load"]), 
            rows, [("b", "bout"), ("c", "cout")], ["a", "aout"], ["load"])
    
    def latency(self, ports):
        match ports:
            case ("b", "bout"): return 1
            case ("a", "aout"): return 0
            case ("a" | "c", "cout"): return self._latency
            case _:
                raise ValueError(f"Requested latency for ports {ports}")

class Tile(Component):
    def __init__(self, element, n, conn_ports, gather_ports, broadcast_ports):
        assert n > 0
        assert all(element.signature.members[p].flow == In for p in broadcast_ports)

        self.conn_ports, self.gather_ports, self.broadcast_ports = conn_ports, gather_ports, broadcast_ports
        self.elements = [element] + [element.copy() for _ in range(n - 1)]
        super().__init__({
            k: v.flow(data.ArrayLayout(v.shape, n) if k in gather_ports else v.shape) for k, v in element.signature.members.items()
        })

    def elaborate(self, _):
        m = am.Module()
        m.submodules += self.elements

        # interconnect
        for pin, pout in self.conn_ports:
            m.d.comb += [getattr(eout, pin).eq(getattr(ein, pout)) for ein, eout in zip(self.elements[:-1], self.elements[1:])]
            m.d.comb += [getattr(self.elements[0], pin).eq(getattr(self, pin)), getattr(self, pout).eq(getattr(self.elements[-1], pout))]

        # gather
        for p in self.gather_ports:
            packed = am.Cat(getattr(e, p) for e in self.elements)
            m.d.comb += packed.eq(getattr(self, p)) if self.signature.members[p].flow == In else getattr(self, p).eq(packed)

        # broadcast
        m.d.comb += [getattr(e, p).eq(getattr(self, p)) for p in self.broadcast_ports for e in self.elements]
        return m

    def copy(self):
        return Tile(self.elements[0].copy(), len(self.elements), self.conn_ports, self.gather_ports, self.broadcast_ports)
    
    def latency(self, ports):
        elat = self.elements[0].latency(ports)
        match ports:
            case p if p in self.conn_ports:
                return elat * len(self.elements)
            case _:
                return elat

class SystolicDelay(Component):
    def __init__(self, component, port, start=0):
        assert start >= 0
        assert component.signature.members[port].flow == In
        port_shape = component.signature.members[port].shape
        assert isinstance(port_shape, data.ArrayLayout)
        self.port = port
        self.component = component
        self.shifters = [Shifter(port_shape.elem_shape, start + i) for i in range(port_shape.length)]
        super().__init__(component.signature)

    def elaborate(self, _):
        m = am.Module()
        m.submodules += [self.component] + self.shifters

        # add shift registers
        for i, shifter in enumerate(self.shifters):
            m.d.comb += [shifter.d.eq(getattr(self, self.port)[i]), getattr(self.component, self.port)[i].eq(shifter.q)]

        # connect other ports
        m.d.comb += [getattr(self.component, k).eq(getattr(self, k)) if v.flow == In else getattr(self, k).eq(getattr(self.component, k)) 
            for k, v in self.signature.members.items() if k != self.port]

        return m

    def latency(self, ports):
        elat = self.component.latency(ports)
        match ports:
            case (p, _) if p == self.port:
                return self.shifters[-1].length + elat
            case _:
                return elat

if __name__ == '__main__':
    from amaranth.back import verilog
    with open("sa.v", 'w') as f:
        f.write(verilog.convert(SystolicDelay(PE(8, 8, 32).tile(2, 2), "a")))
