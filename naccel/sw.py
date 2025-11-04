import sys
import gzip
import urllib.request
from pathlib import Path
from collections.abc import Iterable
import operator
import functools
import json
import numpy as np

from naccel.isa import Activation

def dtype_to_bounds(dt):
    return (-1 << dt.width - 1, (1 << dt.width - 1) - 1) if dt.signed else (0, (1 << dt.width) - 1)

def unpacked(v, width, gran):
    return [(v >> i * gran) & ((1 << gran) - 1) for i in range(-(-width // gran))]

def as_signed(vals, width): return [(v + (1 << width - 1)) % (1 << width) - (1 << width - 1) for v in vals]

def signed_unpack(v, width, grandt):
    unpacked = [(v >> i * grandt.width) & ((1 << grandt.width) - 1) for i in range(-(-width // grandt.width))]
    return as_signed(unpacked, grandt.width) if grandt.signed else unpacked

def packed(vals, width):
    return functools.reduce(operator.or_, map(lambda x: (x[1] & ((1 << width) - 1)) << x[0] * width, enumerate(vals)))

def repack(vals, src_width, target_width, aligned=False):
    aligned_width = -(-src_width // target_width) * target_width if aligned else src_width
    return unpacked(packed(vals, aligned_width), len(vals) * aligned_width, target_width)

def qparams(float_range, int_dtype, symmetric=False):
    float_range = float_range if isinstance(float_range[0], Iterable) else [float_range]
    qmin, qmax = dtype_to_bounds(int_dtype)
    if symmetric and int_dtype.signed:
        scales = [max(map(abs, bounds)) / qmax for bounds in float_range]
        zeros = [0] * len(float_range)
    else:
        scales = [float(bounds[1] - bounds[0]) / (qmax - qmin) for bounds in float_range]
        zeros = [max(min(qmin - round(bounds[0] / scale), qmax), qmin) for bounds,  scale in zip(float_range, scales)]
    return np.array(scales, dtype=np.float32), np.array(zeros, dtype=np.dtype(int_dtype.numpy))

def quantize_multiplier(x):
    assert (x < 1).all() and (x > 0).all()
    n = np.zeros(x.shape, dtype=np.int8)
    while (mask := x < 0.5).any():
        n[mask] += 1
        x[mask] *= 2
    assert (n <= 32).all() and ((q := np.round(x * (1 << 31)).astype(np.int64)) < 1 << 32).all()
    return q.astype(np.int32), n + 31

def pack_activations(a, config):
    m, k = a.shape
    mrows = config.acc_mem_depth - 1
    kblocks = -(-k // config.rows)
    padded = np.pad(a, ((0, 0), (0, kblocks * config.rows - k))).reshape(-1, kblocks, config.rows)
    reshaped = np.concatenate([
        padded[:(m // mrows) * mrows].reshape(-1, mrows, kblocks, config.rows).transpose(0, 2, 1, 3).reshape(-1, config.rows),
        padded[(m // mrows) * mrows:].reshape(-1, kblocks, config.rows).transpose(1, 0, 2).reshape(-1, config.rows)])
    words = [word for row in reshaped.tolist() for word in repack(row, config.act_dtype.width, config.host_data_width)]
    return [byte for word in words for byte in unpacked(word, config.host_data_width, 8)]

def unpack_activations(dat, m, n, config):
    nblocks = -(-n // config.cols)
    act_tile_row_words = -(-config.act_dtype.width * config.rows // config.host_data_width) * config.host_data_width // 8
    acts = np.array([v for row in dat.reshape(-1, act_tile_row_words)
        for v in signed_unpack(packed(row.tolist(), 8), config.act_dtype.width * config.cols, config.act_dtype)],
        dtype=config.act_dtype.numpy)
    return acts.reshape(nblocks, m, config.rows).transpose(1, 0, 2).reshape(m, -1)[:m, :n]

def pack_weights(w, config):
    k, n = w.shape
    kblocks, nblocks = -(-k // config.rows), -(-n // config.cols)
    padded = np.pad(w, ((0, kblocks * config.rows - k), (0, nblocks * config.cols - n))).reshape(kblocks, config.rows, nblocks, config.cols)
    flipped = np.flip(padded, axis=1).transpose(2, 0, 1, 3).reshape(-1, config.cols)
    words = [word for row in flipped.tolist() for word in repack(row, config.weight_dtype.width, config.host_data_width)]
    return [byte for word in words for byte in unpacked(word, config.host_data_width, 8)]

def unpack_weights(dat, k, n, config):
    kblocks, nblocks = -(-k // config.rows), -(-n // config.cols)
    weight_tile_row_size = -(-config.weight_dtype.width * config.cols // config.host_data_width) * config.host_data_width // 8
    w = np.array([v for row in dat.reshape(-1, weight_tile_row_size) 
        for v in signed_unpack(packed(row.tolist(), 8), config.weight_dtype.width * config.cols, config.weight_dtype)],
        dtype=config.weight_dtype.numpy)
    flipped = np.flip(w.reshape(nblocks, kblocks, config.rows, config.cols), axis=2).transpose(1, 2, 0, 3)
    return flipped.reshape(kblocks * config.rows, nblocks * config.cols)[:k, :n]

def pack_bias(c, config):
    _, n = c.shape
    nblocks = -(-n // config.cols)
    padded = np.pad(c, ((0, 0), (0, nblocks * config.cols - n))).reshape(nblocks, config.cols)
    words = [word for row in padded.tolist() for word in repack(row, config.acc_dtype.width, config.host_data_width)]
    return [byte for word in words for byte in unpacked(word, config.host_data_width, 8)]

def unpack_bias(dat, n, config):
    nblocks = -(-n // config.cols)
    bias_tile_row_size = -(-config.acc_dtype.width * config.cols // config.host_data_width) * config.host_data_width // 8
    bias = np.array([v for row in dat.reshape(-1, bias_tile_row_size) 
        for v in signed_unpack(packed(row.tolist(), 8), config.acc_dtype.width * config.cols, config.acc_dtype)],
        dtype=config.acc_dtype.numpy)
    return bias.reshape(-1, nblocks * config.cols)[:, :n]

def quantize_matmul(b, c, sa, za, sb, sd, config):
    bq = np.round(b / sb[None]).astype(config.weight_dtype.numpy)
    cq = np.round(c / (sa * sb[None])).astype(config.acc_dtype.numpy)
    cq -= np.broadcast_to(za.astype(config.acc_dtype.numpy), (1, bq.shape[0])) @ bq
    qmul, shamt = quantize_multiplier(sa * sb / sd)
    return bq, cq, qmul, shamt

class tqdm:
    def __init__(self, iterable=None, pbar_length=20, desc="", total=None, display=True):
        self.desc, self.pbar_length, self.i, self.total, self.display = desc, pbar_length, 0, total or len(iterable), display
        self.iterable = iterable
    def __iter__(self):
        for item in self.iterable:
          yield item
          self.update(1)
        self.update(close=True)
    def update(self, c=0, close=False):
        self.i += c
        percent = min(100, self.i * 100 // self.total)
        filled = int(self.pbar_length * percent // 100)
        if self.display:
            print(f"\r{self.desc} [{'▰' * filled + '▱' * (self.pbar_length - filled)}] {percent}%", end='\n'*close, flush=True, file=sys.stderr)

def fetch(url, fn=None, dstdir=None, pbar_width=20):
    fp = Path(fn if fn is not None else Path(url).name)
    if dstdir is not None: fp = Path(dstdir) / fp
    if fp.is_file(): return fp
    with urllib.request.urlopen(url) as r, open(fp, 'wb') as f:
        assert r.status == 200, r.status
        pbar = tqdm(total=int(r.headers.get('content-length', 0)), desc=f"Downloading {fp}", pbar_length=pbar_width)
        while chunk := r.read(16384): pbar.update(f.write(chunk))
        pbar.update(close=True)
    return fp

def fetch_mnist(datadir="data"):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    def parse(file): return np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
    (basedir := Path(datadir)).mkdir(parents=True, exist_ok=True)
    x_train = parse(fetch(f"{base_url}train-images-idx3-ubyte.gz", dstdir=basedir))[16:].reshape(-1, 784)
    y_train = parse(fetch(f"{base_url}train-labels-idx1-ubyte.gz", dstdir=basedir))[8:]
    x_test = parse(fetch(f"{base_url}t10k-images-idx3-ubyte.gz", dstdir=basedir))[16:].reshape(-1, 784)
    y_test = parse(fetch(f"{base_url}t10k-labels-idx1-ubyte.gz", dstdir=basedir))[8:]
    return x_train, y_train, x_test, y_test

def resize_interp(in_size, out_size, half_pixel_centers=False):
    offset = 0.5 if half_pixel_centers else 0.
    pos = ((np.arange(out_size) + offset) * in_size / out_size - offset).astype(np.float32)
    lower = np.floor(pos)
    upper = np.minimum(lower.astype(int) + 1, in_size - 1)
    lerp = pos - lower
    return np.maximum(lower, 0).astype(int), upper, lerp

def resize_bilinear(x, out_h, out_w, half_pixel_centers=False):
    assert x.ndim == 3, "expects array of shape (B, H, W)"
    in_h, in_w = x.shape[1:]
    left, right, x_lerp = resize_interp(in_w, out_w, half_pixel_centers=half_pixel_centers)
    top, bot, y_lerp = resize_interp(in_h, out_h, half_pixel_centers=half_pixel_centers)

    top_left = x[:, top[..., None], left[None]]
    top_right = x[:, top[..., None], right[None]]
    top = top_left + (top_right - top_left) * x_lerp

    bot_left = x[:, bot[..., None], left[None]]
    bot_right = x[:, bot[..., None], right[None]]
    bot = bot_left + (bot_right - bot_left) * x_lerp

    output = top + (bot - top) * y_lerp[..., None]
    return output

#NOTE: acc_dtype need to be a numpy supported dtype for this to match the hardware
def qmatmul(a, b, c, zd, qmul, shamt, act_dtype, acc_dtype, actfn=Activation.RELU):
    out = a.astype(acc_dtype.numpy) @ b + c
    out = np.maximum(out, 0) if actfn == Activation.RELU else out
    out = (qmul.astype(np.int64) * out >> shamt) + zd
    return np.clip(out, *dtype_to_bounds(act_dtype)).astype(act_dtype.numpy)

class NumpyModel:
    def __init__(self, modelfn, tensorfn, config):
        self.config = config
        with open(modelfn) as fm, open(tensorfn, 'rb') as ft:
            self.ops = json.load(fm)
            self.tensors = ft.read()
        self.layers = [self._parse_op(op) for op in self.ops]

    def __call__(self, x: np.ndarray):
        return functools.reduce(lambda x, f: f(x), self.layers, x)

    def _parse_op(self, op):
        match op["op"]:
            case "fully_connected":
                qparams = {k: np.array(v) for k, v in op["qparams"].items()}
                b = unpack_weights(self._extract_tensor(op["args"]["weight"]), *op["args"]["weight"]["shape"], self.config)
                c = unpack_bias(self._extract_tensor(op["args"]["bias"]), op["args"]["bias"]["shape"][1], self.config)
                return functools.partial(qmatmul, b=b, c=c, zd=qparams["output_zp"], qmul=qparams["qmul"], shamt=qparams["shamt"],
                    act_dtype=self.config.act_dtype, acc_dtype=self.config.acc_dtype, actfn=Activation[op["act"]])
            case "flatten":
                def flatten(x):
                    start_dim, end_dim = op["args"]
                    end_dim = len(x.shape) + end_dim if end_dim < 0 else end_dim
                    return x.reshape(*x.shape[:start_dim], -1, *x.shape[end_dim + 1:])
                return flatten

    def _extract_tensor(self, info):
        return np.frombuffer(self.tensors[info["offset"]:info["offset"] + info["size"]], dtype=np.uint8)
