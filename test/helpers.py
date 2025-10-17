import operator
import functools

from amaranth.sim import Simulator, Period

def run_sim(dut, tb, processes=[], comb=False, vcdfn=None):
    sim = Simulator(dut)
    if not comb:
        sim.add_clock(Period(MHz=1))
    sim.add_testbench(tb)
    for p in processes:
        sim.add_process(p)
    if vcdfn is None:
        sim.run()
    else:
        with sim.write_vcd(vcdfn):
           sim.run()

def dtype_to_bounds(dt): return (-1 << dt.width - 1, (1 << dt.width - 1) - 1) if dt.signed else (0, (1 << dt.width) - 1)

def unpacked(v, width, gran):
    return [(v >> i * gran) & ((1 << gran) - 1) for i in range(-(-width // gran))]

def packed(vals, width):
    return functools.reduce(operator.or_, map(lambda x: (x[1] & ((1 << width) - 1)) << x[0] * width, enumerate(vals)))

def repack(vals, src_width, target_width, aligned=False):
    aligned_width = -(-src_width // target_width) * target_width if aligned else src_width
    return unpacked(packed(vals, aligned_width), len(vals) * aligned_width, target_width)

async def axi4lite_write(ctx, bus, addr, wdata):
    ctx.set(bus.bready, 1)
    ctx.set(bus.awvalid, 1)
    ctx.set(bus.wvalid, 1)
    ctx.set(bus.awaddr, addr)
    ctx.set(bus.wdata, wdata)
    await ctx.tick()
    ctx.set(bus.awvalid, 0)
    ctx.set(bus.wvalid, 0)

async def axi4lite_read(ctx, bus, addr):
    ctx.set(bus.rready, 1)
    ctx.set(bus.arvalid, 1)
    ctx.set(bus.araddr, addr)
    await ctx.tick()
    return ctx.get(bus.rdata)
