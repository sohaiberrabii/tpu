import json
import numpy as np
import pynq as pq
from pynq.overlay import Overlay

from tpu.tpu import TPUConfig
from tpu.isa import Activation
# from test.helpers import repack, packed, unpacked, reshape_for_matmul, tpu_matmul
from tpu.isa import Activation, Op, LoadFunct, MoveFunct
import operator
import functools

def unpacked(v, width, gran):
    return [(v >> i * gran) & ((1 << gran) - 1) for i in range(-(-width // gran))]

def packed(vals, width):
    return functools.reduce(operator.or_, map(lambda x: (x[1] & ((1 << width) - 1)) << x[0] * width, enumerate(vals)))

def repack(vals, src_width, target_width, aligned=False):
    aligned_width = -(-src_width // target_width) * target_width if aligned else src_width
    return unpacked(packed(vals, aligned_width), len(vals) * aligned_width, target_width)

def reshape_for_matmul(a, b, config):
    m, k, n = *a.shape, b.shape[1]
    mrows = config.acc_mem_depth
    kblocks = -(-k // config.rows)
    padded_a = np.pad(a, ((0, 0), (0, kblocks * config.rows - k))).reshape(-1, kblocks, config.rows)
    reshaped_a = np.concatenate([
        padded_a[:(m // mrows) * mrows].reshape(-1, mrows, kblocks, config.rows).transpose(0, 2, 1, 3).reshape(-1, config.rows),
        padded_a[(m // mrows) * mrows:].reshape(-1, kblocks, config.rows).transpose(1, 0, 2).reshape(-1, config.rows)])
    a_words = [word for row in reshaped_a.tolist() for word in repack(row, config.act_dtype.width, config.host_data_width)]

    nblocks = -(-n // config.cols)
    padded_b = np.pad(b, ((0, kblocks * config.rows - k), (0, nblocks * config.cols - n))).reshape(kblocks, config.rows, nblocks, config.cols)
    flipped_b = np.flip(padded_b, axis=1).transpose(2, 0, 1, 3).reshape(-1, config.cols)
    b_words = [word for row in flipped_b.tolist() for word in repack(row, config.weight_dtype.width, config.host_data_width)]
    return a_words, b_words

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

class PynqDevice:
    def __init__(self, ip_config, ip_name="TPU_0", bitstream="tpu.bit"):
        self.config = ip_config
        self.ol = Overlay(bitstream, download=False)
        self.csr_mmio = pq.MMIO(self.ol.ip_dict[ip_name]["phys_addr"], self.ol.ip_dict[ip_name]["addr_range"])

    def download(self):
        pq.pl.PL.reset()
        self.ol.download()
        self.ol.reset()

    def _start_tpu(self, instrs):
        instr_words = repack([self.config.isa_layout.const(instr).as_bits() for instr in instrs],
            self.config.isa_layout.size, self.config.host_data_width, aligned=True)
        instr_bytes = [byte for word in instr_words for byte in unpacked(word, self.config.host_data_width, 8)]
        instr_buf = pq.allocate(len(instr_bytes), dtype=np.uint8)
        instr_buf[:] = instr_bytes

        xfer_size = min(self.config.instr_fifo_depth, self.config.max_reps)
        ins_xfers = -(-len(instrs) // xfer_size)
        for i in range(ins_xfers):
            while not self.csr_mmio.read(self.config.csr_offsets["tpur"]):
                pass
            nins = xfer_size if i < ins_xfers - 1 else len(instrs) - i * xfer_size
            addr_offset = i * xfer_size * -(-self.config.isa_layout.size // self.config.host_data_width) * self.config.host_data_width // 8
            self.csr_mmio.write(self.config.csr_offsets["insadr"], instr_buf.physical_address + addr_offset)
            self.csr_mmio.write(self.config.csr_offsets["nins"], nins)
            self.csr_mmio.write(self.config.csr_offsets["tpus"], 1)

        while not self.csr_mmio.read(self.config.csr_offsets["tpur"]): pass

    def matmul(self, a, b, actfn=Activation.RELU):
        m, k, n = *a.shape, b.shape[1]
        a_words, b_words = reshape_for_matmul(a, b, self.config)
        a_pqbuf = pq.allocate(len(a_words) * self.config.host_data_width // 8, dtype=np.uint8)
        a_pqbuf[:] = [byte for word in a_words for byte in unpacked(word, self.config.host_data_width, 8)]
        b_pqbuf = pq.allocate(len(b_words) * self.config.host_data_width // 8, dtype=np.uint8)
        b_pqbuf[:] = [byte for word in b_words for byte in unpacked(word, self.config.host_data_width, 8)]

        a_tile_row_size = -(-self.config.act_dtype.width * self.config.rows // self.config.host_data_width) * self.config.host_data_width // 8
        with pq.allocate(m * -(-n // self.config.cols) * a_tile_row_size, dtype=np.uint8) as pqbuf:
            instrs = tpu_matmul(m, k, n, self.config, actfn=actfn,
                act_haddr=a_pqbuf.physical_address, weight_haddr=b_pqbuf.physical_address, d_haddr=pqbuf.physical_address)
            self._start_tpu(instrs)
            res = np.array([v for row in pqbuf.reshape(-1, a_tile_row_size) 
                for v in unpacked(packed(row.tolist(), 8), self.config.act_dtype.width * self.config.rows, self.config.act_dtype.width)])
        return res.reshape(-1, m, self.config.cols).transpose(1, 0, 2).reshape(m, -1)[:, :n]

if __name__ == '__main__':
    with open("config.json") as f:
        config = TPUConfig.fromdict(json.load(f))

    actfn = Activation.RELU
    a = np.random.randint(0, 255, size=(32, 32))
    b = np.random.randint(-128, 127, size=(32, 32))
    d = a @ b
    match actfn:
        case Activation.RELU: expected = (d >> 8).clip(min=0, max=255)
        case Activation.NOP: expected = (d >> 8).clip(max=255) & 0xFF

    dev = PynqDevice(config)
    result = dev.matmul(a, b, actfn=actfn)
    assert np.array_equal(result, expected)


