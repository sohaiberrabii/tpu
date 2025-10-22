import operator
import functools

import numpy as np
from amaranth.sim import Simulator, Period

from naccel.isa import Activation, Op, LoadFunct, MoveFunct

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

def matmul_case(m, k, n, config, actfn=Activation.RELU):
    a = np.random.randint(*dtype_to_bounds(config.act_dtype), size=(m, k))
    b = np.random.randint(*dtype_to_bounds(config.weight_dtype), size=(k, n))
    d = a @ b
    match actfn:
        case Activation.RELU: expected = (d >> 8).clip(min=0, max=255)
        case Activation.NOP: expected = (d >> 8).clip(max=255) & 0xFF

    mrows = config.acc_mem_depth
    kblocks = -(-k // config.rows)
    pacts = np.pad(a, ((0, 0), (0, kblocks * config.rows - k))).reshape(-1, kblocks, config.rows)
    acts = np.concatenate([
        pacts[:(m // mrows) * mrows].reshape(-1, mrows, kblocks, config.rows).transpose(0, 2, 1, 3).reshape(-1, config.rows),
        pacts[(m // mrows) * mrows:].reshape(-1, kblocks, config.rows).transpose(1, 0, 2).reshape(-1, config.rows)])
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

def batched(n, m):
    q, r = divmod(n, m)
    return [m] * q + ([r] if r else [])

def tpu_matmul(m, k, n, tpu_conf, actfn=Activation.RELU, act_haddr=0, weight_haddr=0, d_haddr=0):
    mblocks = -(-m // tpu_conf.acc_mem_depth)
    nblocks = -(-n // tpu_conf.cols)
    kblocks = -(-k // tpu_conf.rows)

    w_tile_bytes = -(-tpu_conf.weight_dtype.width * tpu_conf.cols // tpu_conf.host_data_width) * tpu_conf.rows * tpu_conf.host_data_width // 8
    act_row_bytes = -(-tpu_conf.rows * tpu_conf.act_dtype.width // tpu_conf.host_data_width) * tpu_conf.host_data_width // 8

    program = []
    st_act_offset = 0
    for j in range(nblocks):
        ld_act_offset = 0
        for k in range(mblocks):
            nrows = tpu_conf.acc_mem_depth if k < mblocks - 1 else m - k * tpu_conf.acc_mem_depth
            for i in range(kblocks):
                program += [
                    *[load_ha(act_haddr + ld_act_offset + i * tpu_conf.max_reps * act_row_bytes, i * tpu_conf.max_reps, rep)
                        for i, rep in enumerate(batched(nrows, tpu_conf.max_reps))],
                    load_hw(weight_haddr + (i * w_tile_bytes + j * w_tile_bytes * kblocks), tpu_conf.rows),
                    load_w(),
                    spad_sync(),
                    preload_sync(),
                ]
                program.extend([matmul(ii * tpu_conf.max_reps, ii * tpu_conf.max_reps, rep, acc=i > 0)
                    for ii, rep in enumerate(batched(nrows, tpu_conf.max_reps))])
                ld_act_offset += act_row_bytes * nrows
            program += [
                matmul_sync(),
                *[activate(i * tpu_conf.max_reps, i * tpu_conf.max_reps, rep, actfn=actfn) for i, rep in enumerate(batched(nrows, tpu_conf.max_reps))],
                spad_sync(),
                *[store(d_haddr + st_act_offset + i * tpu_conf.max_reps * act_row_bytes, i * tpu_conf.max_reps, rep)
                    for i, rep in enumerate(batched(nrows, tpu_conf.max_reps))],
                spad_sync(),
            ]
            st_act_offset += act_row_bytes * nrows
    return program + [nop()]
