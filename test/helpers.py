import random

import numpy as np
from amaranth.sim import Simulator, Period

from tpu.isa import Activation
from tpu.sw import *


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

def _matmul_case(m, k, n, config):
    a_dtype, b_dtype, c_dtype = config.act_dtype, config.weight_dtype, config.acc_dtype
    a = np.random.randint(*(a_bounds := dtype_to_bounds(a_dtype)), size=(m, k))
    b = np.random.randint(*dtype_to_bounds(b_dtype), size=(k, n))

    #NOTE: if c values are too large, then a and b psums become irrelevant after saturating scaling
    testc_dtype = IntType(width=a_dtype.width + b_dtype.width, signed=a_dtype.signed or b_dtype.signed)
    c = np.random.randint(*dtype_to_bounds(testc_dtype), size=(1, n))

    actfn = random.choice(list(Activation))
    zd    = random.randint(*a_bounds)
    shamt = np.random.randint(0, c_dtype.width * 2 - 1, size=(1,))
    qmul  = np.random.randint(0, dtype_to_bounds(c_dtype)[1], size=(1,))
    expected = qmatmul(a, b, c, zd, qmul, shamt, a_dtype, c_dtype, actfn=actfn)
    return pack_activations(a, config), pack_weights(b, config), pack_bias(c, config), actfn, zd, shamt, qmul, expected

def matmul_case(m, k, n, config, mem, encode=True):
    actbuf, wbuf, biasbuf, actfn, zd, shamt, qmul, expected = _matmul_case(m, k, n, config)
    data_bytes = wbuf + biasbuf + actbuf
    mem[:len(data_bytes)] = data_bytes

    act_offset = len(wbuf) + len(biasbuf)
    res_offset = len(data_bytes)
    instrs = tpu_matmul(m, k, n, config, zd, shamt.item(), qmul.item(), actfn=actfn,
        a_haddr=act_offset, c_haddr=len(wbuf), d_haddr=res_offset)

    encoded_instrs = [config.isa_layout.const(instr).as_bits() for instr in instrs]
    ibuf = repack(encoded_instrs, config.isa_layout.size, config.host_data_width, aligned=True)
    instr_bytes = [byte for word in ibuf for byte in unpacked(word, config.host_data_width, 8)]
    instr_baseaddr = len(mem) - len(instr_bytes)
    mem[instr_baseaddr:] = instr_bytes
    result_size = m * ceildiv(n, config.cols) * aligned_size(config.act_dtype.width * config.rows, config.host_data_width) // 8

    return (instr_baseaddr, len(instrs)) if encode else instrs, (res_offset, result_size), expected
