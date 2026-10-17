"""Build the Kaggle CPU kernels that produce K, the knee-centred region-crop cache (pipeline/kcache.py), one per shard.

    python notebooks/build_kcache_kernel.py --out /tmp/k0 --shard 0              # 5 shards of the 4,407 training studies
    python notebooks/build_kcache_kernel.py --out /tmp/ks --smoke 10             # 10 studies with montages
    .venv/bin/kaggle kernels push -p /tmp/k0

The segmenter and its vendored MONAI come from heliosli's code dataset; kcache.py checks both by sha256 before use.
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
EMBED = ["pipeline/slots64.py", "pipeline/slots96.py", "pipeline/slots118.py", "pipeline/build_slots64.py",
         "pipeline/kcache.py"]
CODE_DATASET = "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"
N_SHARDS = 5

CELL = r'''
# ============================================================================ #
# K cache shard -- built by notebooks/build_kcache_kernel.py. Embedded files are
# written byte-for-byte and hash-checked; kcache.py checks the segmenter.
# ============================================================================ #
import base64, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
SHARD, N_SHARDS, LIMIT, MONTAGES, BUILT_FROM = __SHARD__, __N_SHARDS__, __LIMIT__, __MONTAGES__, __BUILT_FROM__
T0 = time.time()
REPO = Path("/tmp/k_repo")
for rel, (b64, sha) in json.loads(base64.b64decode(__FILES_B64__).decode()).items():
    p = REPO / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64))
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
        sys.exit(f"{rel}: embedded file does not match its hash")
ROOT = WEIGHTS = VENDOR = None
for d, dirs, fs in os.walk("/kaggle/input"):
    dirs[:] = [x for x in dirs if x not in ("train_series", "test_series", "train_images")]
    if ROOT is None and "train_series.csv" in fs and os.path.isdir(os.path.join(d, "train_series")):
        ROOT = d
    if WEIGHTS is None and "anatomy.pt" in fs:
        WEIGHTS = os.path.join(d, "anatomy.pt")
    if VENDOR is None and d.endswith(os.path.join("vendor_runtime", "monai060")):
        VENDOR = d
print(f"shard {SHARD}/{N_SHARDS} limit {LIMIT} built from {BUILT_FROM} | root {ROOT} | weights {WEIGHTS} | "
      f"vendor {VENDOR} | cpus {os.cpu_count()} | /kaggle/working {shutil.disk_usage('/kaggle/working').free / 1e9:.1f} GB free",
      flush=True)
if not (ROOT and WEIGHTS and VENDOR):
    sys.exit("missing the competition root, anatomy.pt or the vendored MONAI")
cmd = [sys.executable, "-u", str(REPO / "pipeline" / "kcache.py"), "--root", ROOT, "--out", "/kaggle/working/k_cache",
       "--weights", WEIGHTS, "--vendor", VENDOR, "--shard", str(SHARD), "--n-shards", str(N_SHARDS),
       "--montages", str(MONTAGES), "--deadline", str(T0 + 11.3 * 3600)]
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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="this many studies of shard 0, with montages")
    a = ap.parse_args()
    files = {rel: [base64.b64encode((PROJ / rel).read_bytes()).decode(),
                   hashlib.sha256((PROJ / rel).read_bytes()).hexdigest()] for rel in EMBED}
    built_from = _git_head()
    slug = "rsna-knee-kcache-smoke" if a.smoke else f"rsna-knee-kcache-s{a.shard}of{N_SHARDS}"
    cell = (CELL.replace("__SHARD__", str(0 if a.smoke else a.shard)).replace("__N_SHARDS__", str(N_SHARDS))
                .replace("__LIMIT__", str(a.smoke)).replace("__MONTAGES__", str(a.smoke or (6 if a.shard == 0 else 0)))
                .replace("__BUILT_FROM__", repr(built_from))
                .replace("__FILES_B64__", repr(base64.b64encode(json.dumps(files).encode()).decode())))
    compile(cell, "cell", "exec")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": cell.strip("\n").splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "keywords": [], "dataset_sources": [CODE_DATASET], "kernel_sources": [],
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": []}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (slug + '.ipynb')}  from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")


if __name__ == "__main__":
    main()
