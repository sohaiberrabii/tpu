import numpy as np

from tpu.tpu import TPUConfig
from tpu.isa import *
from tpu.sw import *


def pack(dat: np.ndarray, dtype: IntType):
    return np.asarray([signed_unpack(packed(row.tolist(), 8), dat.shape[1] * 8, dtype) for row in dat], dtype=dtype.numpy)

def unpack(dat: np.ndarray, dtype: IntType):
    return np.asarray([unpacked(packed(row.tolist(), dtype.width), dat.shape[1] * dtype.width, 8) for row in dat], dtype=np.uint8)

def run(instrs, config: TPUConfig, mem):
    act_mem_width = config.act_dtype.width * config.rows
    assert act_mem_width % 8 == 0
    weight_mem_width = config.weight_dtype.width * config.cols
    assert weight_mem_width % 8 == 0
    acc_mem_width = config.acc_dtype.width * config.cols
    assert acc_mem_width % 8 == 0

    act_mem = np.zeros((config.act_mem_depth, act_mem_width // 8), dtype=np.uint8)
    acc_mem = np.zeros((config.acc_mem_depth, acc_mem_width // 8), dtype=np.uint8)

    wtail = 0
    weight_mem = np.zeros((config.weight_fifo_depth, weight_mem_width // 8), dtype=np.uint8)
    weight_regs = np.zeros((config.rows, config.cols), dtype=config.weight_dtype.numpy)

    qmul, shamt, zp = 0, 0, 0

    for instr in instrs:
        match instr["op"]:
            case Op.SCALER_CONFIG:
                qmul, shamt, zp = [instr["args"]["config"][k] for k in ["qmul", "shamt", "zp"]]

            case Op.LOAD: #TODO: some way to estimate the latency in cycles given bus bandwidth
                reps = instr["args"]["exec"]["reps"]
                haddr = instr["args"]["exec"]["addr1"]["load_store"]
                match instr["funct"]["load"]:
                    case LoadFunct.HOST_ACT:
                        laddr = instr["args"]["exec"]["addr2"]["act"]
                        act_mem[laddr:laddr + reps] = mem[haddr:haddr + reps * act_mem.shape[1]].reshape(reps, -1)
                    case LoadFunct.HOST_BIAS:
                        laddr = instr["args"]["exec"]["addr2"]["acc"]
                        acc_mem[laddr: laddr + reps] = mem[haddr: haddr + reps * acc_mem.shape[1]].reshape(reps, -1)
                    case LoadFunct.HOST_WEIGHT:
                        weight_mem[wtail:wtail + reps] = np.flip(
                                mem[haddr:haddr + reps * weight_mem.shape[1]].reshape(reps, -1), axis=0)
                        wtail += reps

            case Op.MOVE:
                match instr["funct"]["move"]:
                    case MoveFunct.PRELOAD_WEIGHT:
                        weight_regs[:] = pack(weight_mem[wtail - reps:wtail], config.weight_dtype)
                        wtail -= config.rows
                    case MoveFunct.ACTIVATE:
                        reps = instr["args"]["exec"]["reps"]
                        acc_addr = instr["args"]["exec"]["addr1"]["move_exec"]["raddr"]
                        act_addr = instr["args"]["exec"]["addr2"]["act"]
                        actfn = instr["args"]["exec"]["opt"]["actfn"]

                        out = pack(acc_mem[acc_addr:acc_addr + reps], config.acc_dtype)
                        out = np.maximum(out, 0) if actfn == Activation.RELU else out
                        out = (qmul * out.astype(np.int64) >> shamt) + zp #FIXME: int64->2 * acc_dtype.width
                        out = np.clip(out, *dtype_to_bounds(config.act_dtype)).astype(config.act_dtype.numpy)
                        act_mem[act_addr:act_addr + reps] = unpack(out, config.act_dtype)
                        
            case Op.MATMUL:
                reps = instr["args"]["exec"]["reps"]
                act_addr = instr["args"]["exec"]["addr2"]["act"]
                bias_addr = instr["args"]["exec"]["addr1"]["move_exec"]["raddr"]
                acc_addr = instr["args"]["exec"]["addr1"]["move_exec"]["waddr"]
                acc_mode = instr["args"]["exec"]["opt"]["acc_wsel"]["acc"]

                acts = pack(act_mem[act_addr:act_addr + reps], config.act_dtype)
                psums = acts.astype(config.acc_dtype.numpy) @ weight_regs

                if  acc_mode is AccMode.CONST:
                    psums += pack(acc_mem[bias_addr:bias_addr + 1], config.acc_dtype)
                elif acc_mode is AccMode.SAME:
                    psums += pack(acc_mem[acc_addr:acc_addr + reps], config.acc_dtype)
                else:
                    raise RuntimeError(f"Unsupported accumulation mode {acc_mode}")

                acc_mem[acc_addr:acc_addr + reps] = unpack(psums, config.acc_dtype)

            case Op.STORE:
                reps = instr["args"]["exec"]["reps"]
                haddr = instr["args"]["exec"]["addr1"]["load_store"]
                laddr = instr["args"]["exec"]["addr2"]["act"]
                mem[haddr:haddr + reps * act_mem.shape[1]] = act_mem[laddr:laddr + reps].reshape(-1)

            case _:
                pass
