import pytest


from naccel.memory import Accumulator
from test.helpers import run_sim 


def write(dut, ctx, addr, data, acc):
    ctx.set(dut.write.valid, 1)
    ctx.set(dut.write.payload, {"raddr": addr, "waddr": addr, "data": data, "acc": acc})

def read_req(dut, ctx, addr):
    ctx.set(dut.read.resp.ready, 1)
    ctx.set(dut.read.req.valid, 1)
    ctx.set(dut.read.req.payload.addr, addr)

@pytest.fixture(params=[(2, 8)])
def dut(request):
    depth, width = request.param
    return Accumulator(depth=depth, width=width)

def test_basic_accumulation(dut):
    async def tb(ctx):
        write(dut, ctx, addr=0, data=5, acc=1)
        await ctx.tick()
        assert ctx.get(dut.mem.data[0]) == 0
        write(dut, ctx, addr=0, data=3, acc=1)
        await ctx.tick()
        assert ctx.get(dut.mem.data[0]) == 5
        write(dut, ctx, addr=1, data=2, acc=1)
        await ctx.tick()
        assert ctx.get(dut.mem.data[0]) == 8
        await ctx.tick()
        assert ctx.get(dut.mem.data[1]) == 2
    run_sim(dut, tb)

def test_init_acc_rw(dut):
    async def tb(ctx):
        write(dut, ctx, addr=0, data=1, acc=0)
        await ctx.tick()
        write(dut, ctx, addr=0, data=6, acc=1)
        await ctx.tick()
        write(dut, ctx, addr=0, data=3, acc=0)
        read_req(dut, ctx, addr=0)
        await ctx.tick()
        assert ctx.get(dut.read.resp.valid) and ctx.get(dut.read.resp.payload.data) == 7
        await ctx.tick()
        assert ctx.get(dut.read.resp.valid) and ctx.get(dut.read.resp.payload.data) == 3
    run_sim(dut, tb)
