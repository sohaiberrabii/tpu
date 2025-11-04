from functools import partial
import pytest

from tpu.tpu import TPU, TPUConfig
from test.helpers import *

beats_remaining = 0
current_addr = None

async def axis_aw_process(ctx, bus):
    ctx.set(bus.awready, 1)
    global beats_remaining
    global current_addr

    # AW
    async for _, _, awready, awvalid, awaddr, awlen in ctx.tick().sample(bus.awready, bus.awvalid, bus.awaddr, bus.awlen):
        if awready and awvalid:
            current_addr = awaddr
            beats_remaining = awlen + 1

async def axis_w_process(ctx, bus, mem, data_width):
    ctx.set(bus.wready, 1)
    beat_bytes = data_width // 8
    global beats_remaining
    global current_addr

    # W
    async for _, _, wready, wvalid, wdata, wlast in ctx.tick().sample(bus.wready, bus.wvalid, bus.wdata, bus.wlast):
        if wready and wvalid:
            assert beats_remaining > 0
            assert not wlast or beats_remaining == 1
            mem[current_addr // beat_bytes] = wdata
            beats_remaining -= 1
            current_addr += beat_bytes

            if beats_remaining == 0:
                ctx.set(bus.bvalid, 1)
                ctx.set(bus.bresp, 0)

#FIXME: can't be stalled with rready
async def axi4_r_process(ctx, bus, mem, data_width):
    ctx.set(bus.arready, 1)
    beat_bytes = data_width // 8
    beats_remaining = 0
    current_addr = 0 

    async for _, _, arready, arvalid, araddr, arlen in ctx.tick().sample(bus.arready, bus.arvalid, bus.araddr, bus.arlen):
        ctx.set(bus.rvalid, beats_remaining > 0)
        ctx.set(bus.rdata, mem[(current_addr // beat_bytes) % len(mem)])
        ctx.set(bus.rlast, beats_remaining == 1)
        if beats_remaining > 0:
            beats_remaining -= 1
            current_addr += beat_bytes

        if arready and arvalid:
            current_addr = araddr
            beats_remaining = arlen + 1

def run_tpu_sim(tpu, mem, total_instrs, instr_offset, vcdfn="tpu.vcd"):
    async def tb(ctx):
        xfer_size = min(tpu.config.instr_fifo_depth, tpu.config.max_reps)
        ins_xfers = -(-total_instrs // xfer_size)
        for i in range(ins_xfers):
            await ctx.tick().until(tpu.tpu_ready.f.tpur.r_data)
            nins = xfer_size if i < ins_xfers - 1 else total_instrs - i * xfer_size

            await axi4lite_write(ctx, tpu.ctrl, TPUConfig.csr_offsets["nins"], nins)
            ins_word_adr = instr_offset + i * xfer_size * -(-tpu.instr_fifo.width // tpu.config.host_data_width)
            await axi4lite_write(ctx, tpu.ctrl, TPUConfig.csr_offsets["insadr"], ins_word_adr * tpu.config.host_data_width // 8)
            await axi4lite_write(ctx, tpu.ctrl, TPUConfig.csr_offsets["tpus"], 1)
            await ctx.tick()

        await ctx.tick().repeat(100)
        # await ctx.tick().until(tpu.instr_counter == len(instrs))
        # print(f"CYCLES: {ctx.get(tpu.cycle_counter)}, RETINS: {ctx.get(tpu.instr_counter)}")

    awp = partial(axis_aw_process, bus=tpu.bus)
    wp = partial(axis_w_process, bus=tpu.bus, mem=mem, data_width=tpu.config.host_data_width)
    rp = partial(axi4_r_process, bus=tpu.bus, mem=mem, data_width=tpu.config.host_data_width)
    periph_processes = [awp, wp, rp]
    run_sim(tpu, tb, processes=periph_processes, vcdfn=vcdfn)

#FIXME: weights are reloaded even if tiling only occurs on m dim

@pytest.mark.parametrize("data_width", [64])
@pytest.mark.parametrize("max_reps", [15])
@pytest.mark.parametrize("instr_fifo_depth", [32])
@pytest.mark.parametrize("act_mem_depth", [32])
@pytest.mark.parametrize("weight_fifo_depth", [16])
@pytest.mark.parametrize(
    "dim, acc_mem_depth, m, k, n", [
    (8, 32, 23, 65, 67),
])
def test_tpu_standalone(m, k, n, dim, act_mem_depth, acc_mem_depth, weight_fifo_depth, instr_fifo_depth, data_width, max_reps, actfn=Activation.RELU):
    config = TPUConfig(rows=dim, cols=dim, instr_fifo_depth=instr_fifo_depth, weight_fifo_depth=weight_fifo_depth,
        acc_mem_depth=acc_mem_depth, act_mem_depth=act_mem_depth, host_data_width=data_width, max_reps=max_reps)
    tpu = TPU(config)

    actbuf, wbuf, expected = matmul_case(m, k, n, config, actfn=actfn)
    act_offset = len(wbuf)
    d_offset = act_offset + len(actbuf)
    instrs = tpu_matmul(m, k, n, config, actfn=actfn, act_haddr=act_offset * data_width // 8, d_haddr=d_offset * data_width // 8)

    resbuf = unpacked(0, m * -(-n // config.cols) * -(-config.act_dtype.width * config.rows // data_width) * data_width, data_width)
    instr_offset = d_offset + len(resbuf)
    ibuf = repack([tpu.decoder.instr.payload.shape().const(instr).as_bits() for instr in instrs], tpu.instr_fifo.width, data_width, aligned=True)
    mem = wbuf + actbuf + resbuf + ibuf

    run_tpu_sim(tpu, mem, len(instrs), instr_offset, vcdfn="tpu.vcd")
    assert mem[d_offset:d_offset + len(resbuf)] == expected
