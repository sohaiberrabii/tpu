from enum import Enum

from amaranth.lib import data
from amaranth.utils import ceil_log2

class Activation(Enum):
    NOP  = 0
    RELU = 1

class Op(Enum):
    NOP          = 0
    LOAD         = 1  # LOAD from host to local FIFO / Scratchpad / Accumulator
    MATMUL       = 2  # MATMUL by reading from scratchpad and writing to accumulator
    MOVE         = 3  # MOVE instructions such as scaling from accumulator to scratchpad, W FIFO to PE regs
    STORE        = 4  # STORE from local mems to host
    SPAD_SYNC    = 6  # wait for scratchpad load
    PRELOAD_SYNC = 7  # wait for weight preload from weight fifo to array
    MATMUL_SYNC  = 8  # wait for MATMUL

class LoadFunct(Enum):
    HOST_WEIGHT = 0
    HOST_ACT    = 1

class MoveFunct(Enum):
    PRELOAD_WEIGHT = 0
    ACTIVATE       = 1

class ISALayout(data.StructLayout):
    def __init__(self, haddr_width, sp_addr_width, acc_addr_width, max_repeats):
        super().__init__({
        "op":    Op,
        "funct": data.UnionLayout({"load": LoadFunct, "move": MoveFunct}),
        "reps":  ceil_log2(max_repeats + 1),
        "addr1": data.UnionLayout({"load_store": haddr_width, "move_exec": acc_addr_width}),
        "addr2": sp_addr_width,
        "opt":   data.UnionLayout({"acc_wsel": data.StructLayout({"acc": 1, "wsel": 1}), "actfn": Activation}),
    })
