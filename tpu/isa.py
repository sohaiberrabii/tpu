from enum import Enum

from amaranth.lib import data
from amaranth.utils import ceil_log2

class Activation(Enum):
    NONE = 0
    RELU = 1

class Op(Enum):
    NOP           = 0
    LOAD          = 1     # LOAD from host to local FIFO / Scratchpad / Accumulator
    MATMUL        = 2     # MATMUL by reading from scratchpad and writing to accumulator
    MOVE          = 3     # MOVE instructions such as scaling from accumulator to scratchpad, W FIFO to PE regs
    STORE         = 4     # STORE from local mems to host
    SPAD_SYNC     = 6     # wait for scratchpad load
    PRELOAD_SYNC  = 7     # wait for weight preload from weight fifo to array
    MATMUL_SYNC   = 8     # wait for MATMUL
    ACC_SYNC      = 9     # wait for accumulator writes (load from host / matmul output)
    SCALER_CONFIG = 10    # CONFIGURE the scaler unit

class LoadFunct(Enum):
    HOST_WEIGHT = 0
    HOST_ACT    = 1
    HOST_BIAS   = 2

class MoveFunct(Enum):
    PRELOAD_WEIGHT = 0
    ACTIVATE       = 1

class AccMode(Enum):
    NO    = 0  # no accumulation, overwrite
    SAME  = 1  # accumulate over the same addresses of the write
    CONST = 2  # use some other constant address while the write address is incremented

class ISALayout(data.StructLayout):
    def __init__(self, act_shape, acc_shape, haddr_width, sp_addr_width, acc_addr_width, max_repeats):
        super().__init__({
        "op":    Op,
        "funct": data.UnionLayout({"load": LoadFunct, "move": MoveFunct}),
        "args": data.UnionLayout({
            "exec": data.StructLayout({
                "reps":  ceil_log2(max_repeats + 1),
                "addr1": data.UnionLayout({
                        "load_store": haddr_width,
                        "move_exec": data.StructLayout({"raddr": acc_addr_width, "waddr": acc_addr_width})}),
                "addr2": data.UnionLayout({"act": sp_addr_width, "acc": acc_addr_width}),
                "opt":  data.UnionLayout({"acc_wsel": data.StructLayout({"acc": AccMode, "wsel": 1}), "actfn": Activation}),
            }),
            "config": data.StructLayout({
                "qmul": acc_shape,
                "shamt": ceil_log2(acc_shape.width * 2),
                "zp": act_shape,
            }),
        }),
    })

def exec_args(kwargs): return {"args": {"exec": kwargs}}
def config_args(kwargs): return {"args": {"config": kwargs}}

def load_hw(haddr, nrows):
    return {"op": Op.LOAD, "funct": {"load": LoadFunct.HOST_WEIGHT}, **exec_args({"reps": nrows, "addr1": {"load_store": haddr}})}

def load_hb(haddr, laddr, nrows):
    return {"op": Op.LOAD, "funct": {"load": LoadFunct.HOST_BIAS}, **exec_args({
        "reps": nrows, "addr1": {"load_store": haddr}, "addr2": {"acc": laddr}})}

def load_ha(haddr, laddr, nrows):
    return {"op": Op.LOAD, "funct": {"load": LoadFunct.HOST_ACT}, **exec_args({
        "reps": nrows, "addr1": {"load_store": haddr}, "addr2": {"act": laddr}})}

def load_w(wsel=0):
    return {"op": Op.MOVE, "funct": {"move": MoveFunct.PRELOAD_WEIGHT}, **exec_args({"opt": {"acc_wsel": {"wsel": wsel}}})}

# in AccMode.SAME raddr is not used
def matmul(act_addr, acc_addr, psum_addr, nrows, wsel=0, acc=AccMode.NO):
    return {"op": Op.MATMUL, **exec_args({"addr1": {"move_exec": {"waddr": acc_addr, "raddr": psum_addr}},
        "addr2": {"act": act_addr}, "reps": nrows, "opt": {"acc_wsel": {"wsel": wsel, "acc": acc}}})}

def activate(act_addr, acc_addr, nrows, actfn=Activation.RELU):
    return {"op": Op.MOVE, "funct": {"move": MoveFunct.ACTIVATE}, **exec_args({
        "addr1": {"move_exec": {"raddr": acc_addr}}, "addr2": {"act": act_addr}, "reps": nrows, "opt": {"actfn": actfn}})}

def store(act_haddr, act_laddr, nrows):
    return {"op": Op.STORE, **exec_args({"reps": nrows, "addr1": {"load_store": act_haddr}, "addr2": {"act": act_laddr}})}

def scaler_config(qmul, shamt, zp):
    return {"op": Op.SCALER_CONFIG, **config_args({"qmul": qmul, "shamt": shamt, "zp": zp})}

def acc_sync(): return {"op": Op.ACC_SYNC}
def spad_sync(): return {"op": Op.SPAD_SYNC}
def preload_sync(): return {"op": Op.PRELOAD_SYNC}
def matmul_sync(): return {"op": Op.MATMUL_SYNC}
def nop(): return {"op": Op.NOP}
