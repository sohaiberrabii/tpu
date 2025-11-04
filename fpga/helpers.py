import os
import tempfile
import zipfile
from pathlib import Path
import subprocess
from dataclasses import asdict
import json

from amaranth import Module
from amaranth.lib import wiring
from amaranth.back.verilog import convert
from amaranth.lib import wiring
from tpu.tpu import TPU
from tpu.sw import fetch


class InterfaceRenamer(wiring.Component):
    def __init__(self, dut):
        self.dut = dut
        super().__init__({'_'.join(path): member for path, member, _ in dut.signature.flatten(dut)})

    def elaborate(self, _):
        m = Module()
        m.submodules.dut = self.dut
        for path, member, value in self.dut.signature.flatten(self.dut):
            renamed_value = getattr(self, '_'.join(path))
            if member.flow == wiring.In:
                m.d.sync += value.eq(renamed_value)
            else:
                m.d.sync += renamed_value.eq(value)
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
    with open(build_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)

    vivado_cmd = ["vivado", "-mode", "batch", "-nolog", "-nojournal", "-source"]
    res = subprocess.run(vivado_cmd + [scriptdir / "build.tcl"], stderr=subprocess.PIPE, encoding="utf-8")
    if res.returncode:
        raise RuntimeError(res.stderr)

    return next(build_dir.rglob("*.bit")), next(build_dir.rglob("*.hwh"))

def deploy(config_fn, bit_fn, hwh_fn, remote_dir=""):
    assert (dev := os.getenv("PYNQ")),  f"PYNQ env var should be set to properly configured ssh alias of pynq-z2 device"
    try:
        subprocess.run(["scp", config_fn, bit_fn, hwh_fn, f"{dev}:{remote_dir}"], check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(f"scp failed to deploy to ssh alias PYNQ={dev}")


if __name__ == '__main__':
    from tpu.tpu import TPUConfig
    config = TPUConfig(rows=8, cols=8, max_reps=15, instr_fifo_depth=32, act_mem_depth=32, acc_mem_depth=32, host_data_width=64, weight_fifo_depth=16)
    generate_bitstream(config)
    with open("fpga/build/config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)
