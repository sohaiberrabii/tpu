import random

import numpy as np
from amaranth.sim import Simulator, Period

from tpu.isa import *
from tpu.sw import *
from tpu.tpu import IntType

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

def matmul_case(m, k, n, config):
    np.set_printoptions(linewidth=150)
    a_dtype, b_dtype, c_dtype = config.act_dtype, config.weight_dtype, config.acc_dtype
    a = np.random.randint(*(a_bounds := dtype_to_bounds(a_dtype)), size=(m, k))
    b = np.random.randint(*dtype_to_bounds(b_dtype), size=(k, n))
    # print("a:")
    # print(a)
    # for row in a:
    #     print(hex(packed(row.tolist(), config.act_dtype.width))[2:])
    # print("b:")
    # print(b)
    # for row in b:
    #     print(hex(packed(row.tolist(), config.weight_dtype.width))[2:])
    # print("psums:")
    # for i in range(2):
    #     print(f"psum {i}:")
    #     print(a[:, i * 8:(i + 1) * 8] @ b[i*8:(i + 1)*8, :])

    #NOTE: if c values are too large, then a and b psums become irrelevant after saturating scaling
    testc_dtype = IntType(width=a_dtype.width + b_dtype.width, signed=a_dtype.signed or b_dtype.signed)
    c = np.random.randint(*dtype_to_bounds(testc_dtype), size=(1, n))

    # print(c)
    # print("c:", hex(packed(c[0].tolist(), config.acc_dtype.width))[2:])

    actfn = random.choice(list(Activation))
    zd    = random.randint(*a_bounds)
    shamt = np.random.randint(30, 40, size=(1,))
    qmul  = np.random.randint(0, dtype_to_bounds(c_dtype)[1], size=(1,))
    expected = qmatmul(a, b, c, zd, qmul, shamt, a_dtype, c_dtype, actfn=actfn)

    # print("exp:")
    # print(expected)
    # for row in expected:
    #     print(hex(packed(row.tolist(), config.act_dtype.width))[2:])

    # a_words, b_words = reshape_for_matmul(a, b, config)
    # a_bytes = pack_activations(a, config)
    # b_bytes = pack_weights(b, config)
    # c_bytes = pack_bias(c, config)
    # nblocks = -(-n // config.cols)
    # pexp = np.pad(expected, ((0, 0), (0, nblocks * config.cols - n))).reshape(m, nblocks, config.cols).transpose(1, 0, 2).reshape(-1, config.cols)
    # expected_words = [word for row in pexp.tolist() for word in repack(row, config.act_dtype.width, config.host_data_width)]
    return pack_activations(a, config), pack_weights(b, config), pack_bias(c, config), actfn, zd, shamt, qmul, expected

def batched(n, m):
    q, r = divmod(n, m)
    return [m] * q + ([r] if r else [])

def tpu_matmul(m, k, n, tpu_conf, output_zp, shamt, qmul, actfn=Activation.RELU, a_haddr=0, b_haddr=0, c_haddr=0, d_haddr=0):
    bias_laddr = tpu_conf.acc_mem_depth - 1
    mblocks = -(-m // (tpu_conf.acc_mem_depth - 1))
    nblocks = -(-n // tpu_conf.cols)
    kblocks = -(-k // tpu_conf.rows)

    w_tile_bytes = -(-tpu_conf.weight_dtype.width * tpu_conf.cols // tpu_conf.host_data_width) * tpu_conf.rows * tpu_conf.host_data_width // 8
    bias_row_bytes = -(-tpu_conf.acc_dtype.width * tpu_conf.cols // tpu_conf.host_data_width) * tpu_conf.host_data_width // 8
    act_row_bytes = -(-tpu_conf.act_dtype.width * tpu_conf.rows // tpu_conf.host_data_width) * tpu_conf.host_data_width // 8

    program = [scaler_config(qmul, shamt, output_zp)]
    st_act_offset = 0
    for j in range(nblocks):
        ld_act_offset = 0
        program += [load_hb(c_haddr + j * bias_row_bytes, bias_laddr, 1)]
        for k in range(mblocks):
            nrows = (tpu_conf.acc_mem_depth - 1) if k < mblocks - 1 else m - k * (tpu_conf.acc_mem_depth - 1)
            for i in range(kblocks):
                program += [
                    *[load_ha(a_haddr + ld_act_offset + i * tpu_conf.max_reps * act_row_bytes, i * tpu_conf.max_reps, rep)
                        for i, rep in enumerate(batched(nrows, tpu_conf.max_reps))],
                    load_hw(b_haddr + (i * w_tile_bytes + j * w_tile_bytes * kblocks), tpu_conf.rows),
                    matmul_sync(), #NOTE: wsel is not yet supported
                    load_w(),
                    spad_sync(),
                    preload_sync(),
                    acc_sync(),
                ]
                acc_mode = AccMode.CONST if i == 0 else AccMode.SAME
                acc_raddr = bias_laddr if i == 0 else 0
                program.extend([matmul(waddr := ii * tpu_conf.max_reps, waddr, acc_raddr, rep, acc=acc_mode)
                    for ii, rep in enumerate(batched(nrows, tpu_conf.max_reps))])
                ld_act_offset += act_row_bytes * nrows
            program += [
                acc_sync(),
                *[activate(i * tpu_conf.max_reps, i * tpu_conf.max_reps, rep, actfn=actfn) for i, rep in enumerate(batched(nrows, tpu_conf.max_reps))],
                spad_sync(),
                *[store(d_haddr + st_act_offset + i * tpu_conf.max_reps * act_row_bytes, i * tpu_conf.max_reps, rep)
                    for i, rep in enumerate(batched(nrows, tpu_conf.max_reps))],
                spad_sync(),
            ]
            st_act_offset += act_row_bytes * nrows
    return program + [nop()]
