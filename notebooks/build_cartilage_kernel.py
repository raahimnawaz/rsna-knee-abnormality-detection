"""Build the Kaggle CPU kernels of the cartilage-measurement test (LOG 10-16 ~10:30), one kernel per shard.

    python notebooks/build_cartilage_kernel.py --out /tmp/cm0 --shard 0 --n-shards 4
    python notebooks/build_cartilage_kernel.py --out /tmp/cms --smoke 4          # 4 studies, montages, no shards
    .venv/bin/kaggle kernels push -p /tmp/cm0

The studies: the 58 gold plus a random 800 of the other training studies (seed fixed below), sorted, then dealt
round-robin to the shards. The segmenter is heliosli's `anatomy.pt`, checked against R6's manifest hash.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OWNER = "raahimnawaz"
EMBED = ["pipeline/slots64.py", "pipeline/slots96.py", "pipeline/slots118.py", "pipeline/build_slots64.py",
         "pipeline/cartilage_measure.py"]
CODE_DATASET = "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"
ANATOMY_SHA = "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"   # R6 v1 manifest, anchor_weights_sha256
SEED, N_SAMPLE = 20261016, 800
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]

CELL = r'''
# ============================================================================ #
# Cartilage-measurement test shard -- built by notebooks/build_cartilage_kernel.py.
# Embedded files are written byte-for-byte and hash-checked; the segmenter is
# checked against R6's manifest hash.
# ============================================================================ #
import base64, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
SHARD, N_SHARDS, BUILT_FROM, MONTAGES = __SHARD__, __N_SHARDS__, __BUILT_FROM__, __MONTAGES__
UIDS, ANATOMY_SHA = __UIDS__, __ANATOMY_SHA__
T0 = time.time()
REPO = Path("/tmp/cm_repo")
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
print(f"shard {SHARD}/{N_SHARDS} | {len(UIDS)} studies | built from {BUILT_FROM} | root {ROOT} | weights {WEIGHTS} | "
      f"vendor {VENDOR} | cpus {os.cpu_count()}", flush=True)
if not (ROOT and WEIGHTS and VENDOR):
    sys.exit("missing the competition root, anatomy.pt or the vendored MONAI")
h = hashlib.sha256()
with open(WEIGHTS, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
if h.hexdigest() != ANATOMY_SHA:
    sys.exit(f"anatomy.pt sha256 {h.hexdigest()} is not R6's {ANATOMY_SHA}")
print("anatomy.pt sha256 ok", flush=True)
out = Path("/kaggle/working/cartilage")
out.mkdir(parents=True, exist_ok=True)
(out / "uids.json").write_text(json.dumps(UIDS))
cmd = [sys.executable, "-u", str(REPO / "pipeline" / "cartilage_measure.py"), "--root", ROOT, "--uids",
       str(out / "uids.json"), "--weights", WEIGHTS, "--vendor", VENDOR, "--out", str(out),
       "--montages", str(MONTAGES), "--deadline", str(T0 + 11.3 * 3600)]
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


def sample_uids() -> list[str]:
    """The 58 gold + a random 800 of the rest, sorted. Fixed by SEED; the same list on every machine."""
    t = pd.read_csv(PROJ / "data" / "train.csv")
    gold = t[t[LABELS].notna().any(axis=1)].StudyInstanceUID.astype(str)
    rest = sorted(t[t[LABELS].isna().all(axis=1)].StudyInstanceUID.astype(str))
    pick = np.random.default_rng(SEED).choice(len(rest), N_SAMPLE, replace=False)
    out = sorted(set(gold) | {rest[i] for i in pick})
    assert len(gold) == 58 and len(out) == 58 + N_SAMPLE, (len(gold), len(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=0, help="this many studies of shard 0 (gold first), with montages")
    a = ap.parse_args()
    uids = sample_uids()
    if a.smoke:
        t = pd.read_csv(PROJ / "data" / "train.csv")
        gold = sorted(t[t[LABELS].notna().any(axis=1)].StudyInstanceUID.astype(str))
        mine, slug, montages = gold[: a.smoke], "rsna-knee-cartilage-smoke", a.smoke
    else:
        mine = [u for i, u in enumerate(uids) if i % a.n_shards == a.shard]
        slug, montages = f"rsna-knee-cartilage-s{a.shard}of{a.n_shards}", (6 if a.shard == 0 else 0)
    files = {rel: [base64.b64encode((PROJ / rel).read_bytes()).decode(),
                   hashlib.sha256((PROJ / rel).read_bytes()).hexdigest()] for rel in EMBED}
    built_from = _git_head()
    cell = (CELL.replace("__SHARD__", str(a.shard)).replace("__N_SHARDS__", str(a.n_shards))
                .replace("__BUILT_FROM__", repr(built_from)).replace("__MONTAGES__", str(montages))
                .replace("__UIDS__", repr(mine)).replace("__ANATOMY_SHA__", repr(ANATOMY_SHA))
                .replace("__FILES_B64__", repr(base64.b64encode(json.dumps(files).encode()).decode())))
    compile(cell, "cell", "exec")
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
            "dataset_sources": [CODE_DATASET], "kernel_sources": [],
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": []}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (slug + '.ipynb')}  {len(mine)} studies  from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")


if __name__ == "__main__":
    main()
