import os
import pytest
import numpy as np

from tpu.tpu import TPUConfig
from tpu.isa import *
from tpu.sw import unpack_activations
from tpu.emu import run
from test.helpers import matmul_case

TESTREPS = int(os.getenv("TESTENV", 100))

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
        instrs, (d_offset, d_size), expected = matmul_case(m, k, n, config, mem, encode=False)
        run(instrs, config, mem)
        result = unpack_activations(mem[d_offset:d_offset + d_size], m, n, config)
        np.testing.assert_array_equal(result, expected)
