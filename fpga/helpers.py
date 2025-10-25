import sys
import urllib.request
import tempfile
import zipfile
import json
from pathlib import Path
import subprocess
from dataclasses import asdict
from functools import reduce

from amaranth import Module
from amaranth.lib import wiring
from amaranth.back.verilog import convert
from amaranth.lib import wiring
from naccel.tpu import TPU


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

class InterfaceRenamer(wiring.Component):
    def __init__(self, dut):
        self.dut = dut
        super().__init__({'_'.join(path): member for path, member, _ in dut.signature.flatten(dut)})

    def elaborate(self, _):
        m = Module()
        m.submodules.dut = self.dut
        for path, member, value in self.dut.signature.flatten(self.dut):
            renamed_value = getattr(self, '_'.join(path))
            dut_value = reduce(getattr, path, self.dut)
            if member.flow == wiring.In:
                m.d.sync += dut_value.eq(renamed_value)
            else:
                m.d.sync += renamed_value.eq(dut_value)
        return m

def generate_bitstream(config, board="pynq-z2", build_dir=None):
    assert board == "pynq-z2", "only pynq-z2 is currently supported"
    board_files = {
        "pynq-z2":  "https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip",
    }
    scriptdir = Path(__file__).parent
    build_dir = scriptdir / "build" if build_dir is None else Path(build_dir)
    build_dir.mkdir(exist_ok=True)
    if not any((build_dir / board).rglob("board.xml")):
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(fetch(board_files[board], dstdir=temp_dir), 'r') as zf:
                zf.extractall(build_dir)

    with open(build_dir / "tpu.v", 'w') as f:
        f.write(convert(InterfaceRenamer(TPU(config)), name="TPU", emit_src=False, strip_internal_attrs=True))

    vivado_cmd = ["vivado", "-mode", "batch", "-nolog", "-nojournal", "-source"]
    res = subprocess.run(vivado_cmd + [scriptdir / "build.tcl"], stderr=subprocess.PIPE, encoding="utf-8")
    if res.returncode: raise RuntimeError(res.stderr)


if __name__ == '__main__':
    from naccel.tpu import TPUConfig
    config = TPUConfig(rows=8, cols=8, max_reps=15, instr_fifo_depth=32, act_mem_depth=32, acc_mem_depth=32, host_data_width=64, weight_fifo_depth=16)
    generate_bitstream(config)
