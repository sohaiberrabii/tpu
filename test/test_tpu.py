from functools import partial
import pytest
import numpy as np

from naccel.tpu import TPU, TPUConfig
from naccel.isa import Activation, Op, LoadFunct, MoveFunct
from test.helpers import run_sim, unpacked, repack, axi4lite_write, dtype_to_bounds

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

def matmul_case(m, k, n, config, actfn=Activation.RELU):
    a = np.random.randint(*dtype_to_bounds(config.act_dtype), size=(m, k))
    b = np.random.randint(*dtype_to_bounds(config.weight_dtype), size=(k, n))
    print("a:", ["".join([hex(x & ((1 << config.act_dtype.width) - 1))[2:] for x in row][::-1]) for row in a])
    d = a @ b
    print("d:", ["".join([hex(x & ((1 << config.acc_dtype.width) - 1))[2:] for x in row][::-1]) for row in d])
    match actfn:
        case Activation.RELU: expected = (d >> 8).clip(min=0, max=255)
        case Activation.NOP: expected = (d >> 8).clip(max=255) & 0xFF
    print("exp:", ["".join([hex(x & ((1 << config.act_dtype.width) - 1))[2:] for x in row][::-1]) for row in expected])

    mrows = config.acc_mem_depth
    kblocks = -(-k // config.rows)
    pacts = np.pad(a, ((0, 0), (0, kblocks * config.rows - k))).reshape(-1, kblocks, config.rows)
    acts = np.concatenate([
        pacts[:(m // mrows) * mrows].reshape(-1, mrows, kblocks, config.rows).transpose(0, 2, 1, 3).reshape(-1, config.rows),
        pacts[-(m % mrows):].reshape(-1, kblocks, config.rows).transpose(1, 0, 2).reshape(-1, config.rows)])
    act_words = [word for row in acts.tolist() for word in repack(row, config.act_dtype.width, config.host_data_width)]

    nblocks = -(-n // config.cols)
    pweights = np.pad(b, ((0, kblocks * config.rows - k), (0, nblocks * config.cols - n))).reshape(kblocks, config.rows, nblocks, config.cols)
    flipped_weights = np.flip(pweights, axis=1).transpose(2, 0, 1, 3).reshape(-1, config.cols)
    weight_words = [word for row in flipped_weights.tolist() for word in repack(row, config.weight_dtype.width, config.host_data_width)]

    pexp = np.pad(expected, ((0, 0), (0, nblocks * config.cols - n))).reshape(m, nblocks, config.cols).transpose(1, 0, 2).reshape(-1, config.cols)
    expected_words = [word for row in pexp.tolist() for word in repack(row, config.act_dtype.width, config.host_data_width)]
    return act_words, weight_words, expected_words

def load_hw(weight_haddr, nrows):
    return {"op": Op.LOAD, "funct": {"load": LoadFunct.HOST_WEIGHT}, "reps": nrows, "addr1": {"load_store": weight_haddr}}
def load_ha(act_haddr, act_laddr, nrows):
    return {"op": Op.LOAD, "funct": {"load": LoadFunct.HOST_ACT}, "reps": nrows, "addr1": {"load_store": act_haddr}, "addr2": act_laddr}
def load_w(wsel=0): return {"op": Op.MOVE, "funct": {"move": MoveFunct.PRELOAD_WEIGHT}, "opt": {"acc_wsel": {"wsel": wsel}}}
def spad_sync(): return {"op": Op.SPAD_SYNC}
def preload_sync(): return {"op": Op.PRELOAD_SYNC}
def matmul_sync(): return {"op": Op.MATMUL_SYNC}
def nop(): return {"op": Op.NOP}
def matmul(act_addr, acc_addr, nrows, wsel=0, acc=0):
    return {"op": Op.MATMUL, "addr1": {"move_exec": acc_addr}, "addr2": act_addr, "reps": nrows, "opt": {"acc_wsel": {"wsel": wsel, "acc": acc}}}
def activate(act_addr, acc_addr, nrows, actfn=Activation.RELU):
    return {"op": Op.MOVE, "funct": {"move": MoveFunct.ACTIVATE}, "addr1": {"move_exec": acc_addr}, "addr2": act_addr, "reps": nrows,
        "opt": {"actfn": actfn}}
def store(act_haddr, act_laddr, nrows):
    return {"op": Op.STORE, "reps": nrows, "addr1": {"load_store": act_haddr}, "addr2": act_laddr}

def run_tpu_sim(tpu, mem, total_instrs, instr_offset, vcdfn="tpu.vcd"):
    async def tb(ctx):
        xfer_size = min(tpu.config.instr_fifo_depth, (1 << len(tpu.bus.arlen)) // tpu.instr_dma_reader.nbeats)
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

def tpu_matmul(m, k, n, tpu_conf, actfn=Activation.RELU, act_haddr=0, weight_haddr=0, d_haddr=0):
    mblocks = -(-m // tpu_conf.acc_mem_depth)
    nblocks = -(-n // tpu_conf.cols)
    kblocks = -(-k // tpu_conf.rows)

    weight_tile = -(-tpu_conf.weight_dtype.width * tpu_conf.cols // tpu_conf.host_data_width) * tpu_conf.cols
    act_tile = -(-tpu_conf.rows * tpu_conf.act_dtype.width // tpu_conf.host_data_width)

    program = []
    st_act_offset = 0
    for j in range(nblocks):
        ld_act_offset = 0
        for k in range(mblocks):
            nrows = tpu_conf.acc_mem_depth if k < mblocks - 1 else m - k * tpu_conf.acc_mem_depth
            for i in range(kblocks):
                program += [
                    load_ha(act_haddr + ld_act_offset, 0, nrows),
                    load_hw(weight_haddr + (i * weight_tile + j * weight_tile * kblocks), tpu_conf.rows),
                    load_w(wsel=i % 2),
                    spad_sync(),
                    preload_sync(),
                ]
                program.append(matmul(0, 0, nrows, wsel=i % 2, acc=i > 0))
                ld_act_offset += act_tile * nrows
            program += [
                matmul_sync(),
                activate(0, 0, nrows, actfn=actfn),
                spad_sync(),
                store(d_haddr + st_act_offset, 0, nrows),
            ]
            st_act_offset += act_tile * nrows
    return program + [nop()]

@pytest.mark.parametrize("data_width", [64])
@pytest.mark.parametrize("instr_fifo_depth", [32])
@pytest.mark.parametrize(
    "dim, acc_mem_depth, m, k, n", [
    (8, 32, 8, 8, 8),
])
def test_tpu_standalone(m, k, n, dim, acc_mem_depth, instr_fifo_depth, data_width, actfn=Activation.RELU):
    config = TPUConfig(rows=dim, cols=dim, instr_fifo_depth=instr_fifo_depth, acc_mem_depth=acc_mem_depth, host_data_width=data_width)
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
    print(mem)
    assert mem[d_offset:d_offset + len(resbuf)] == expected
