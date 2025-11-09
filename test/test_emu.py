import os
import pytest
import numpy as np

from tpu.tpu import TPUConfig
from tpu.isa import *
from tpu.sw import unpack_activations
from tpu.emu import run
from test.helpers import matmul_case

TESTREPS = int(os.getenv("TESTENV", 1000))

@pytest.mark.parametrize(
    "dim, acc_mem_depth, act_mem_depth, weight_fifo_depth", [
    (8, 32, 32, 32),
])
def test_emulator(dim, acc_mem_depth, act_mem_depth, weight_fifo_depth):
    config = TPUConfig(rows=dim, cols=dim, acc_mem_depth=acc_mem_depth, act_mem_depth=act_mem_depth,
        weight_fifo_depth=weight_fifo_depth)
    mem = np.zeros((4 * 1024 * 1024), dtype=np.uint8)

    m, k, n = 8, 8, 8
    for _ in range(TESTREPS):
        a_bytes, b_bytes, c_bytes, actfn, zp, shamt, qmul, expected = matmul_case(m, k, n, config)
        mem[:len(b_bytes)] = b_bytes
        mem[len(b_bytes):len(b_bytes) + len(a_bytes)] = a_bytes

        c_offset = len(a_bytes) + len(b_bytes)
        mem[c_offset:c_offset + len(c_bytes)] = c_bytes
        d_offset = c_offset + len(c_bytes)
        instrs = [
            scaler_config(qmul, shamt, zp),
            load_hb(c_offset, 31, 1),
            load_ha(len(b_bytes), 0, 8),
            load_hw(0, 8),
            load_w(wsel=0),
            acc_sync(),
            preload_sync(),
            spad_sync(),
            matmul(0, 0, 31, 8, wsel=0, acc=AccMode.CONST),
            acc_sync(),
            activate(0, 0, 8, actfn=actfn),
            spad_sync(),
            store(d_offset, 0, 8),
            spad_sync(),
        ]
        run(instrs, config, mem)
        result = unpack_activations(mem[d_offset:d_offset + m * config.act_dtype.width * config.rows // 8], m, n, config)
        np.testing.assert_array_equal(result, expected)
