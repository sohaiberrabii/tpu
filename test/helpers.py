import random

import numpy as np
from amaranth.sim import Simulator, Period

from tpu.isa import Activation
from tpu.tpu import IntType
from tpu.sw import dtype_to_bounds, pack_activations, pack_bias, pack_weights, qmatmul

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

def matmul_case(m, k, n, config):
    np.set_printoptions(linewidth=150)
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
