"""Speed bench: what torch.compile and partial checkpointing buy on a T4, for the Stage B recipe.

v2 (2026-09-27) -- READ THIS FIRST. v1 (below, read 09-26) adopted nothing: compile cut the training step 17-25% with an
identical loss but failed the eval clause on a 12-study eval that likely timed compile warm-up. v2 changes two things:
`--compile` now compiles the TRAINING step only (eval runs under `torch.compiler.set_stance("force_eager")`), and
every variant runs as the real runs do -- DDP on BOTH T4s -- because compile has never run under DDP here and DDP adds
~2.5 GB (Stage B peaked 8.5 GB against this bench's 6.05 on one GPU). Variants: base · compT · compT12. The rule is
v1's, unchanged, in the RUNBOOK's v2 block.

    python notebooks/build_speed_bench_kernel.py --out /tmp/sb
    python -m kaggle kernels push -p /tmp/sb

WHY (AUDIT.md 09-26, pass 4). Ortho trains at ~10 TFLOPS per T4 -- 40-50% of what the card sustains on fp16
GEMMs -- with every encoder block gradient-checkpointed while peak memory sits at 8.5 GB of 15. Data loading
is already ruled out from numbers in hand (step 5: 1,751 micro-steps x 2.685 s = 78.4 min against a measured
78.7 min epoch). So two compute-side changes are timed here, each ALONE and together, against the unchanged
recipe, on ONE T4 (DDP's two-T4 scaling is already measured: 55 min/epoch on Stage B):

    base     every block checkpointed, eager               (today)
    ck12     the first 12 of 24 blocks checkpointed        (--ckpt-blocks 12)
    ck0      no checkpointing                              (--ckpt-blocks 0; may not fit: recorded if not)
    comp     torch.compile, every block checkpointed       (--compile)
    comp12   torch.compile + the first 12 checkpointed

Same 300 studies, same seed, same batches, drop-path 0 and 40 micro-steps each, so every variant sees the same
windows and its per-step LOSS can be compared with base's, and eval timed on the same 12 studies.

THE RULE, WRITTEN BEFORE THE RUN (RUNBOOK): a variant is ADOPTED for later runs if its median s per micro-step
is >= 10% below base's, its peak memory < 14 GB, and its first-40 loss trace stays within 0.02 of base's at
every step (checkpointing does not change the maths; compile's fused kernels may move the fourth decimal).
Its eval s/study is reported beside it, and a variant that slows eval by more than it speeds training is
not adopted.
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
SLUG = "rsna-knee-speed-bench"
EMBED = ["fusion/train_windows.py", "fusion/orthofoundation_model.py", "fusion/qwen_vlm.py", "extractor/metrics.py"]
STAGE_B_SHARDS = [f"{OWNER}/rsna-knee-stage-b-s{i}of8" for i in range(8)]
# v1: {"base": [], "ck12": [...12], "ck0": [...0], "comp": ["--compile"], "comp12": [...]} -- fusion/runs_speed_bench/
VARIANTS = {"base": [], "compT": ["--compile"], "compT12": ["--compile", "--ckpt-blocks", "12"]}

CELL = r'''
import base64, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
T0 = time.time()
def log(m): print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)
REPO, OUT = Path("/tmp/sb_repo"), Path("/kaggle/working")
for rel, (b64, sha) in json.loads(base64.b64decode(__FILES__).decode()).items():
    p = REPO / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(base64.b64decode(b64))
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha: sys.exit(f"{rel}: hash mismatch")
log(f"built from __BUILT_FROM__; repo files written")
import torch; log(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
NGPU = torch.cuda.device_count()          # v2: DDP on every GPU, as the real runs
sys.path[:0] = [str(REPO / "fusion"), str(REPO / "extractor")]
from train_windows import Corpus, LABELS
ids = sorted(Corpus(Path("/kaggle/input"), "stage_b").ids)
src = next(Path(r) / "llm_labels_v2.csv" for r, d, f in os.walk("/kaggle/input") if "llm_labels_v2.csv" in f)
tg = pd.read_csv(src).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(ids)[LABELS].dropna()
tr, va = list(tg.index[:300]), list(tg.index[300:312])
D = Path("/tmp/sb"); D.mkdir(exist_ok=True)
tg.reset_index().to_csv(D / "targets.csv", index=False)
pd.Series(tr, name="StudyInstanceUID").to_csv(D / "train.csv", index=False)
pd.Series(va, name="StudyInstanceUID").to_csv(D / "val.csv", index=False)
COMMON = ["--arch", "ortho", "--corpus", "/kaggle/input", "--layout", "stage_b", "--train", str(D / "train.csv"),
          "--val", str(D / "val.csv"), "--labels", str(D / "targets.csv"), "--weights-dir", "/tmp", "--epochs", "1",
          "--max-steps", "40", "--micro", "2", "--bs", str(2 * NGPU), "--k", "12", "--res", "384", "--pos-emb",
          "--llrd", "0.9", "--bb-lr", "9e-5", "--drop-path", "0", "--eval-bs", "2", "--time-budget-h", "1", "--seed", "42"]
VARIANTS = __VARIANTS__
res = {}
for i, (name, extra) in enumerate(VARIANTS.items()):
    out = OUT / "runs" / name
    launch = ([sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--node_rank", "0", "--master_addr",
               "127.0.0.1", "--master_port", str(29521 + i), "--nproc_per_node", str(NGPU)] if NGPU > 1
              else [sys.executable, "-u"])
    cmd = launch + [str(REPO / "fusion" / "train_windows.py")] + COMMON + ["--out", str(out)] + extra
    t = time.time()
    rc = subprocess.call(cmd, env=dict(os.environ, TW_WORKERS=str(max(1, (os.cpu_count() or 4) // NGPU))))
    b = json.loads((out / "bench.json").read_text()) if (out / "bench.json").exists() else None
    res[name] = {"rc": rc, "wall_min": (time.time() - t) / 60, "bench": b}
    log(f"{name}: rc {rc}, {res[name]['wall_min']:.1f} min" + (f", {b['sec_per_micro_step_median']:.3f} s/step, "
        f"peak {b.get('peak_cuda_gb') or 0:.2f} GB, eval {b['eval_sec_per_study']:.2f} s/study" if b else " -- no bench (OOM?)"))
base = res["base"]["bench"]
rows = []
for name, r in res.items():
    b = r["bench"]
    if not b or not base:
        rows.append({"variant": name, "verdict": "FAILED (rc %s)" % r["rc"]}); continue
    sp = 1 - b["sec_per_micro_step_median"] / base["sec_per_micro_step_median"]
    n = min(len(b["loss_trace"]), len(base["loss_trace"]))
    dl = float(np.max(np.abs(np.array(b["loss_trace"][:n]) - np.array(base["loss_trace"][:n])))) if n else float("nan")
    ev = b["eval_sec_per_study"] / base["eval_sec_per_study"] - 1
    peak = b.get("peak_cuda_gb") or 0.0
    ok = name != "base" and sp >= 0.10 and peak < 14 and dl <= 0.02 and ev <= sp
    rows.append({"variant": name, "s_per_step": round(b["sec_per_micro_step_median"], 3), "train_speedup": round(sp, 3),
                 "peak_gb": round(peak, 2), "max_dloss": round(dl, 4), "eval_s_study": round(b["eval_sec_per_study"], 3),
                 "eval_change": round(ev, 3), "data_wait_frac": round(b["data_wait_frac"] or 0, 4),
                 "verdict": "base" if name == "base" else ("ADOPT" if ok else "no")})
t = pd.DataFrame(rows)
print(t.to_string(index=False), flush=True)
(OUT / "speed_bench.json").write_text(json.dumps({"rows": rows, "raw": res, "built_from": "__BUILT_FROM__"}, indent=1))
log("speed bench complete")
'''


def _git_head() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(PROJ), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(PROJ), "status", "--porcelain", "--"] + EMBED, capture_output=True,
                               text=True, check=True).stdout.strip()
        return sha + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = {rel: [base64.b64encode((PROJ / rel).read_bytes()).decode(),
                   hashlib.sha256((PROJ / rel).read_bytes()).hexdigest()] for rel in EMBED}
    built = _git_head()
    cell = (CELL.replace("__FILES__", repr(base64.b64encode(json.dumps(files).encode()).decode()))
                .replace("__BUILT_FROM__", built).replace("__VARIANTS__", repr(VARIANTS)))
    compile(cell, "cell", "exec")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": cell.strip("\n").splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
    (a.out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "keywords": [], "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels"], "kernel_sources": STAGE_B_SHARDS,
        "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": []}, indent=2))
    print(f"built {a.out / (SLUG + '.ipynb')} from {built} | variants {list(VARIANTS)}")
    if "uncommitted" in built:
        print("  ⚠️ embedded files differ from HEAD -- commit first")


if __name__ == "__main__":
    main()
