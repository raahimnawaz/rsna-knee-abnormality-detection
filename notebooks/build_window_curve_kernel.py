"""The efficiency curve on a T4: `fusion/window_curve.py` over the 691, both GPUs, no training.

    python notebooks/build_window_curve_kernel.py --out /tmp/curve
    python -m kaggle kernels push -p /tmp/curve
    python -m kaggle kernels output raahimnawaz/rsna-knee-window-curve -p /tmp/curve_out
    for n in 36 27 18 12; do python fusion/score_oof.py /tmp/curve_out/curve/n54 /tmp/curve_out/curve/n$n; done

RUNBOOK tuning plan item 1. Same checkpoint as Submission B (`runs_ortho/last.pt`, sha-checked),
same 691 fold-0 studies it scored 0.8298 on, read from the same Stage A shards it trained on.
GPU 0 runs n = 54, 27, 12 and GPU 1 runs n = 36, 18; the n = 54 pass must reproduce the scored
OOF before anything else is written (see the module docstring). Timings are per T4.
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
SLUG = "rsna-knee-window-curve"
WEIGHTS_KERNEL = f"{OWNER}/rsna-knee-c6-train-ortho"
WEIGHTS_SHA256 = "1fe51be2ff4deaae4dfd5fafd2354f3b9bbeb0c26cda3fd918a00699fd481b98"
STAGE_A_SHARDS = [f"{OWNER}/rsna-knee-stage-a-s{i}of4" for i in range(4)]
EMBED = ["fusion/window_curve.py", "fusion/infer_windows.py", "fusion/train_windows.py",
         "pipeline/build_slots64.py", "pipeline/slots64.py", "pipeline/slots96.py",
         "fusion/orthofoundation_model.py", "extractor/metrics.py"]
SPLIT = {0: [54, 27, 12], 1: [36, 18]}

CELL = r'''
import base64, hashlib, json, os, subprocess, sys, time
from pathlib import Path
T0 = time.time()
def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

REPO, OUT = Path("/tmp/curve_repo"), Path("/kaggle/working/curve")
for rel, (b, sha) in json.loads(base64.b64decode(__FILES_B64__).decode()).items():
    p = REPO / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b))
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
        sys.exit(f"{rel}: embedded file does not match its hash")
log("built from __BUILT_FROM__")

RUN = None
for d, dirs, fs in os.walk("/kaggle/input"):
    if Path(d).name == "runs_ortho" and {"last.pt", "oof_all.csv", "val_studies.csv"} <= set(fs):
        if hashlib.sha256((Path(d) / "last.pt").read_bytes()).hexdigest() == "__WEIGHTS_SHA256__":
            RUN = Path(d)
if RUN is None:
    sys.exit("runs_ortho/last.pt with the recorded sha256 not found -- attach __WEIGHTS_KERNEL__")
log(f"weights {RUN / 'last.pt'} (sha256 ok)")

OUT.mkdir(parents=True, exist_ok=True)
procs = {}
for gpu, ns in __SPLIT__.items():
    cmd = [sys.executable, "-u", str(REPO / "fusion" / "window_curve.py"), "--corpus", "/kaggle/input",
           "--weights", str(RUN / "last.pt"), "--val", str(RUN / "val_studies.csv"),
           "--reference", str(RUN / "oof_all.csv"), "--out", str(OUT), "--n"] + [str(n) for n in ns]
    logf = open(OUT / f"gpu{gpu}.log", "w")
    procs[gpu] = (subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                   stdout=logf, stderr=subprocess.STDOUT), logf)
bad = []
for gpu, (p, logf) in procs.items():
    rc = p.wait()
    logf.close()
    print(f"----- GPU {gpu} (exit {rc})\n" + (OUT / f"gpu{gpu}.log").read_text(), flush=True)
    if rc:
        bad.append(gpu)
if bad:
    sys.exit(f"window_curve failed on GPU(s) {bad}")
log("window curve complete -- score locally with fusion/score_oof.py, paired against n54")
'''



def _check_import_closure(embed: list[str]) -> None:
    """Refuse to build if an embedded file imports a repo module that is not embedded. 2026-09-25:
    build_slots64 grew an import of slots96 and this kernel would have died at its gate on Kaggle."""
    import re
    have = {Path(r).stem for r in embed if r.endswith(".py")}
    local = {p.stem for d in ("fusion", "pipeline", "extractor") for p in (PROJ / d).glob("*.py")}
    for rel in embed:
        if not rel.endswith(".py"):
            continue
        for m in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", (PROJ / rel).read_text(), re.M):
            if m.group(1) in local and m.group(1) not in have:
                raise SystemExit(f"{rel} imports {m.group(1)}, which is not embedded -- add it to EMBED")

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
    a = ap.parse_args()
    _check_import_closure(EMBED)
    files = {}
    for rel in EMBED:
        b = (PROJ / rel).read_bytes()
        files[rel] = [base64.b64encode(b).decode(), hashlib.sha256(b).hexdigest()]
    built_from = _git_head()
    cell = (CELL.replace("__FILES_B64__", repr(base64.b64encode(json.dumps(files).encode()).decode()))
                .replace("__BUILT_FROM__", built_from).replace("__WEIGHTS_SHA256__", WEIGHTS_SHA256)
                .replace("__WEIGHTS_KERNEL__", WEIGHTS_KERNEL).replace("__SPLIT__", repr(SPLIT)))
    compile(cell, "cell", "exec")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": cell.strip("\n").splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "dataset_sources": [], "kernel_sources": STAGE_A_SHARDS + [WEIGHTS_KERNEL],
            "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (SLUG + '.ipynb')} from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")
    print(f"push:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
