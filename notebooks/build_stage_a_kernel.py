"""Stage A on Kaggle: one CPU kernel per shard of the 64-slice cache. `pipeline/build_slots64.py`.

    python notebooks/build_stage_a_kernel.py --out /tmp/sa --shard 0 --n-shards 4 --limit 20   # smoke
    python notebooks/build_stage_a_kernel.py --out /tmp/sa0 --shard 0 --n-shards 4             # a shard
    python -m kaggle kernels push -p /tmp/sa

CPU only, so no GPU quota and no two-session GPU cap. Four shards keep each output (~8 GB of
`vols.npy`) inside a kernel's ~20 GB `/kaggle/working`. Training attaches the four shard kernels as
`kernel_sources`; `train_windows.Corpus` reads them as one row-ordered view.

⛔ Smoke first (`--limit 20`) and read, before pushing any full shard:
  * `decode failures 0` -- the corpus mixes uncompressed, JPEG Lossless and JPEG 2000; a missing
    decoder shows up here, not as a crash;
  * `ordered_by {'normal': ...}` with no `instance_number` -- geometry, not a file-order guess;
  * `slot fill` near Stage 0's prediction (`DATA.md` §9): sag_1 ~1.0, sag_2 ~0.97, cor_1 ~1.0,
    cor_2 ~0.75, ax ~1.0 -- on 20 studies, roughly;
  * `render_edges.png` -- GATE A: the first and last slice of every slot is knee, not wrap or air.
The local fixture (`notebooks/stage_a_fixture.py`, `--check`) already proves order, 140 mm scale on
both axes, padding and masking, and was mutation-tested: filename ordering and the frontier's
clamped crop both fail it.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
OWNER = "raahimnawaz"
EMBED = ["pipeline/slots64.py", "pipeline/build_slots64.py"]

CELL = r'''
# ============================================================================ #
# Stage A shard -- built by notebooks/build_stage_a_kernel.py. The builder and
# the spec are embedded byte-for-byte and hash-checked.
# ============================================================================ #
import base64, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
SHARD, N_SHARDS, LIMIT, BUILT_FROM = __SHARD__, __N_SHARDS__, __LIMIT__, __BUILT_FROM__
T0 = time.time()
REPO = Path("/tmp/sa_repo")
for rel, (b64, sha) in json.loads(base64.b64decode(__FILES_B64__).decode()).items():
    p = REPO / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64))
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
        sys.exit(f"{rel}: embedded file does not match its hash")
import pydicom, cv2, numpy
print(f"shard {SHARD}/{N_SHARDS} limit {LIMIT} built from {BUILT_FROM} | pydicom {pydicom.__version__} "
      f"cv2 {cv2.__version__} | cpus {os.cpu_count()} | /kaggle/working "
      f"{shutil.disk_usage('/kaggle/working').free / 1e9:.1f} GB free", flush=True)
try:
    import pylibjpeg, openjpeg  # noqa: F401
    print("decoders: pylibjpeg + openjpeg present", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"decoders: pylibjpeg/openjpeg not importable ({e}) -- JPEG 2000 may fall back or fail", flush=True)
ROOT = None
for d, dirs, fs in os.walk("/kaggle/input"):
    dirs[:] = [x for x in dirs if x not in ("train_series", "test_series", "train_images")]
    if "train_series.csv" in fs and os.path.isdir(os.path.join(d, "train_series")):
        ROOT = d
        break
if ROOT is None:
    sys.exit("no competition root with train_series/ under /kaggle/input")
cmd = [sys.executable, "-u", str(REPO / "pipeline" / "build_slots64.py"), "--root", ROOT,
       "--out", "/kaggle/working/stage_a", "--shard", str(SHARD), "--n-shards", str(N_SHARDS),
       "--workers", str(os.cpu_count() or 4)]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]
print(" ".join(cmd), flush=True)
rc = subprocess.call(cmd)
print(f"exited {rc} after {(time.time() - T0) / 60:.1f} min", flush=True)
if rc:
    sys.exit(rc)
'''


def _git_head() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(PROJ), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(PROJ), "status", "--porcelain", "--"] + EMBED,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke: this many studies of the shard")
    a = ap.parse_args()
    files = {rel: [base64.b64encode((PROJ / rel).read_bytes()).decode(),
                   hashlib.sha256((PROJ / rel).read_bytes()).hexdigest()] for rel in EMBED}
    built_from = _git_head()
    cell = (CELL.replace("__SHARD__", str(a.shard)).replace("__N_SHARDS__", str(a.n_shards))
                .replace("__LIMIT__", str(a.limit)).replace("__BUILT_FROM__", repr(built_from))
                .replace("__FILES_B64__", repr(base64.b64encode(json.dumps(files).encode()).decode())))
    compile(cell, "cell", "exec")
    slug = f"rsna-knee-stage-a-s{a.shard}of{a.n_shards}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": cell.strip("\n").splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "dataset_sources": [], "kernel_sources": [],
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": []}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (slug + '.ipynb')}  shard {a.shard}/{a.n_shards} limit {a.limit or 'all'}  from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")


if __name__ == "__main__":
    main()
