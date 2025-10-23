import sys
import urllib.request
import tempfile
import zipfile
import json
from pathlib import Path
import subprocess
from dataclasses import asdict

from amaranth.back.verilog import convert
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

    #FIXME: the verilog need post gen modifications to work with vivado scripts
    print(f"Generating verilog with configuration: {json.dumps(asdict(config), indent=2)}")
    with open(build_dir / "tpu.v", 'w') as f:
        f.write(convert(TPU(config), name="TPU", emit_src=False, strip_internal_attrs=True))

    vivado_cmd = ["vivado", "-mode", "batch", "-source"]
    print("Generating Block Design")
    bd_result = subprocess.run(vivado_cmd + [scriptdir / "create_bd.tcl"], stderr=subprocess.PIPE, encoding="utf-8")
    if bd_result.returncode: raise RuntimeError(bd_result.stderr)
    print("Running Synthesis & Implementation")
    impl_result = subprocess.run(vivado_cmd + [scriptdir / "build.tcl"], stderr=subprocess.PIPE, encoding="utf-8")
    if impl_result.returncode: raise RuntimeError(impl_result.stderr)
    print("DONE")
    print("BIT:"+ next(build_dir.rglob("tpu_wrapper.bit")))
    print("HWH:"+ next(build_dir.rglob("tpu.hwh")))


if __name__ == '__main__':
    from naccel.tpu import TPUConfig
    config = TPUConfig(rows=8, cols=8, max_reps=15, instr_fifo_depth=32, act_mem_depth=32, acc_mem_depth=32, host_data_width=64, weight_fifo_depth=16)
    generate_bitstream(config)
