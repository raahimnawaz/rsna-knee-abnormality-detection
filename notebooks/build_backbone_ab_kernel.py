"""The backbone A/B: one encoder per kernel, on the frontier's public corpus and recipe.

    python notebooks/build_backbone_ab_kernel.py --arch ortho --out /tmp/ab --smoke   # CPU, wiring
    python notebooks/build_backbone_ab_kernel.py --arch ortho --out /tmp/ab --bench   # T4, 1 epoch
    python notebooks/build_backbone_ab_kernel.py --arch ortho --out /tmp/ab --epochs N
    python -m kaggle kernels push -p /tmp/ab

THE QUESTION. OrthoFoundation was only ever tested FROZEN, at 160 mm, on the old 6-tile cache (C-5,
0.7710 -- still the best number any encoder trained here has produced). This is its first fair
test: fully fine-tuned, in the recipe behind the public 0.924-0.928 single model (`MODEL.md` §0),
on the corpus that recipe trained on, with everything but the encoder held fixed:

    ortho     OrthoFoundation-L                        the arm
    dinov3l   stock DINOv3-L, Ortho's own architecture  ortho - dinov3l = what knee pretraining is worth
    coatnet   the frontier's encoder                    ortho - coatnet = whether Ortho is the best encoder here

HELD FIXED ACROSS ARMS: the corpus (`dreaddevelopment/knee-raptor-corpus` + `-ext`, CC0, 44 slices,
140 mm, 15-85% -- their STARTING geometry, not the 0.928 one, and the same for every arm); the
training studies (a seeded draw of 2,871, the count `runs_port_repro` and the sampling test used,
from fold != 0, not gold, complete `steven_v2` targets); `steven_v2` targets; the recipe in
`fusion/train_windows.py`; and the validation set -- the SAME 691 fold-0 studies `runs_port_repro`
(0.7597), pilkwang (0.8453) and the sampling test are scored on, by the same `score_oof.py`. The
site-grouped partition is held to `provenance/inputs.json`, so a changed upstream stops the run.

THE RULE, PRE-REGISTERED HERE BEFORE ANY ARM HAS A NUMBER (one seed each; the paired SD on this
instrument has been ~0.009):
    knee pretraining pays        ortho - dinov3l > 0 paired, P(delta>0) >= 0.95
    Ortho is the encoder to use  ortho - coatnet > 0 paired, P(delta>0) >= 0.95
    a null is a result           neither -> record it as a null, not a refutation (§3 rule 4)
The primary read is the LAST epoch. `avg3/` (mean of the last three epochs' weights) is secondary,
fixed in advance, and is not a second chance to pass.

--bench runs ONE full epoch on the T4 and scores it. That prices the full run from a measurement
(PREMORTEM §1's rule) and gives a first, weak number. It is not the A/B.

STAGE B (`--corpus stage_b`, 2026-09-25). One package run against step 5's model (`c6_ortho`, 0.8298):
the Stage B cache (96 slices, 384 px, canonical orientation, two extra series slots -- `DATA.md` §10b)
AND the model changes (layer-wise LR decay 0.8, drop-path 0.1, a slot + depth embedding per window,
both T4s under DDP, epochs fitted to the session). Same 3,502 training studies, same `steven_v2`
targets, same 691 validation studies. It launches under torchrun on every visible GPU (two gloo
processes on CPU for --smoke, so the DDP path is exercised before any GPU hour is spent).
THE RULE, PRE-REGISTERED 2026-09-25 BEFORE THE RUN: the package pays if `c7_ortho - c6_ortho > 0`
paired on the same 691, P(delta>0) >= 0.95, primary = last epoch; `avg3/` is secondary and not a
second chance. A pass says the PACKAGE works, not which part of it; a loss ships nothing from it.

ALL DATA (`--all-data`, 2026-09-29). Train on EVERY non-gold study with complete targets: the 3,502 pool,
the 691 and the rest of fold 0, 4,349 in all. Nothing is held out. It is for a model whose start already
trained on those studies (the public CoAtNet, `--init`, which saw all 4,349 plus gold x3), so its 691
read was never a clean check and the LB is its only judge. The val list is the gold studies: train_windows
needs one disjoint from train, and gold is never trained on. No score_oof pairing. Slug suffix `-all`.

QWEN (`--arch qwen35vl --corpus stage_a`, 2026-09-26). Qwen3.5-2B as a VLM study reader
(`fusion/qwen_vlm.py`): the vision encoder fine-tuned, LoRA on the language model, every window of a
study one image in one sequence. Same Stage A cache, the same 3,502 training studies as step 5
(`--n-train 0` is the default for this arch), `steven_v2` targets, the same 691, both T4s under DDP.
`--bench` here is SHORT -- 40 optimiser steps and 60 validation studies at every window -- because it
answers cost and fp16 finiteness on a real T4, not accuracy. Its full run's rule is in the RUNBOOK,
written before that run.
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
EMBED = [
    "fusion/train_windows.py",
    "fusion/orthofoundation_model.py",
    "fusion/qwen_vlm.py",
    "fusion/score_oof.py",
    "fusion/folds.py",
    "pipeline/site_fingerprint.py",
    "pipeline/provenance.py",
    "extractor/metrics.py",
    "provenance/inputs.json",
    "fusion/runs_port_repro/oof_all.csv",      # the control: 691 fold-0 studies, 0.7597
    "fusion/runs_c6_ortho/oof_all.csv",        # Stage B's rule is scored against step 5's model, 0.8298
]
DATASETS = ["dreaddevelopment/knee-raptor-corpus",           # 3,200 studies, CC0
            "dreaddevelopment/knee-raptor-corpus-ext",       # 1,207 studies, CC0
            "stevenleehans/rsna-knee-llm-report-labels",     # llm_labels_v2.csv -> targets.csv
            "lixin73/rsna-knee-llm-report-labels-sol56"]     # the scoring reference
KERNELS = ["zhukovoleksiy/rsna-metadata-probe"]              # headers.parquet -> site folds
STAGE_A_SHARDS = [f"{OWNER}/rsna-knee-stage-a-s{i}of4" for i in range(4)]   # build_stage_a_kernel.py
STAGE_B_SHARDS = [f"{OWNER}/rsna-knee-stage-b-s{i}of8" for i in range(8)]   # ... --stage b
# Stage B's package: the model half of the run. The data half is the cache itself.
# Layer-wise decay REDISTRIBUTES the learning rate toward the top blocks; it must not cut it. Step 5 ran a
# flat 3e-5 on all 24 blocks (sum 7.2e-4). With decay 0.9, block i of 24 gets top * 0.9^(25 - i), which sums
# to top * 8.28, so top = 9e-5 keeps the same total (7.5e-4): upper blocks ~8e-5, the middle ~2.3e-5, the
# patch embedding ~6e-6. (0.8 on a 3e-5 top, as first written, cut the total 6x and left the embedding at
# 1.1e-7 -- read off the 09-25 smoke log, before any result.)
PACKAGE = ["--res", "384", "--drop-path", "0.1", "--llrd", "0.9", "--bb-lr", "9e-5", "--pos-emb", "--fit-budget"]
# Speed, adopted 09-27 by the pre-registered bench (RUNBOOK, speed bench v2; fusion/runs_speed_bench_v2): the TRAINING
# step compiled (eval stays eager) and the first 12 of 24 blocks checkpointed -- -28.6% per step on T4x2 DDP,
# loss unchanged, 13.3 GB peak. Fallback if a long run ever OOMs: drop "--ckpt-blocks", "12" (7.0 GB, -19%).
SPEED = ["--compile", "--ckpt-blocks", "12"]
N_TRAIN, DRAW_SEED = 2871, 20260923
# The VLM: every visible GPU, the epoch count fitted to the session, and the short bench.
# It trains on 12-image sequences, so it is read on 12 evenly spaced windows, not every window: a
# language model's length generalisation from 12 images to ~50 is a train/eval skew (AUDIT.md, 09-26).
QWEN_EXTRA = ["--fit-budget", "--k", "12", "--eval-windows", "12"]
# The versions this arch was built and tested against locally; the image may carry neither.
QWEN_TRANSFORMERS = "5.17.0"
QWEN_BENCH = ["--max-steps", "40", "--limit-val", "60", "--eval-bs", "1"]


SETUP = r'''
# ============================================================================ #
# Backbone A/B -- setup. Built by notebooks/build_backbone_ab_kernel.py; read
# that docstring for the question and the rule fixed before the run.
# ============================================================================ #
import base64, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ARCH = __ARCH__
MODE = __MODE__
LAYOUT = __LAYOUT__
EXTRA = __EXTRA__
EPOCHS = __EPOCHS__
BUILT_FROM = __BUILT_FROM__
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


import torch, timm
if ARCH.startswith("qwen"):
    # Qwen3.5 needs a transformers that knows `qwen3_5`; the image may predate it. Internet is on here.
    # Checked in a FRESH interpreter: training runs in a subprocess, and a package upgraded under this
    # notebook's own process cannot be re-imported here (the 09-26 smoke: pip succeeded, reload did not).
    _probe = [sys.executable, "-c", "import transformers as t; from transformers import Qwen3_5Model; print(t.__version__)"]
    _r = subprocess.run(_probe, capture_output=True, text=True)
    if _r.returncode:
        _old = subprocess.run([sys.executable, "-c", "import transformers as t; print(t.__version__)"],
                              capture_output=True, text=True).stdout.strip()
        log(f"transformers {_old or '(none)'} has no qwen3_5 -> installing transformers==__QWEN_TF__")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers==__QWEN_TF__"])
        _r = subprocess.run(_probe, capture_output=True, text=True)
        if _r.returncode:
            sys.exit("qwen3_5 still unavailable after the install:\n" + _r.stderr[-2000:])
    log(f"transformers {_r.stdout.strip()} (qwen3_5 available to the training subprocess)")
log(f"arch {ARCH}  mode {MODE}  epochs {EPOCHS}  built from {BUILT_FROM}")
log(f"torch {torch.__version__}  timm {timm.__version__}  cuda {torch.cuda.is_available()}"
    + (f" {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
for _p in ("/kaggle/tmp", "/tmp", "/kaggle/working"):
    if os.path.isdir(_p):
        log(f"disk {_p}: {shutil.disk_usage(_p).free / 1e9:.1f} GB free")

SCRATCH = Path("/tmp")
REPO = SCRATCH / "ab_repo"                # outside /kaggle/working: it holds competition CSVs
OUT = Path("/kaggle/working")
RUN = OUT / f"runs_{ARCH}"
FILES = json.loads(base64.b64decode(__FILES_B64__).decode())
for _rel, (_b64, _sha) in FILES.items():
    _p = REPO / _rel
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_bytes(base64.b64decode(_b64))
    if hashlib.sha256(_p.read_bytes()).hexdigest() != _sha:
        sys.exit(f"{_rel}: embedded file does not match its hash")
log(f"wrote {len(FILES)} repo files to {REPO}, all hashes match")

ROOT = None
for _d, _dirs, _fs in os.walk("/kaggle/input"):
    _dirs[:] = [x for x in _dirs if x not in ("train_series", "test_series", "train_images")]
    if "train.csv" in _fs and "train_series.csv" in _fs:
        ROOT = Path(_d)
        break
if ROOT is None:
    sys.exit("no competition root under /kaggle/input")
log(f"competition root: {ROOT}")
'''


# The sampling test's PREP, verbatim in what it computes: the same targets, reference, site folds,
# provenance checks and partition check, so the two runs sit on one partition.
PREP = r'''
# ============================================================================ #
# Labels, site folds, the partition check, then the train/val lists.
# ============================================================================ #
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
for _p in (REPO, REPO / "extractor", REPO / "pipeline", REPO / "fusion"):
    sys.path.insert(0, str(_p))
D = REPO / "data"
(D / "external").mkdir(parents=True, exist_ok=True)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


WANT = {"llm_labels_v2.csv", "labels_llm_gpt56sol.csv", "headers.parquet"}
FOUND = {}
for _root, _dirs, _files in os.walk("/kaggle/input"):
    _dirs[:] = [d for d in _dirs if d not in ("train_series", "test_series")]
    for _f in WANT.intersection(_files):
        FOUND.setdefault(_f, []).append(Path(_root) / _f)
for _f in sorted(WANT):
    if _f not in FOUND:
        sys.exit(f"{_f} not found under /kaggle/input -- is its source in kernel-metadata.json?")
    log(f"input {_f}: {FOUND[_f][0]}  sha256 {_sha(FOUND[_f][0])[:16]}")

from provenance import check as _check
_check(FOUND["llm_labels_v2.csv"][0], "stevenleehans/rsna-knee-llm-report-labels")

shutil.copy(ROOT / "train.csv", D / "train.csv")
_tr = pd.read_csv(D / "train.csv")
_v2 = (pd.read_csv(FOUND["llm_labels_v2.csv"][0])
       .drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID"))
_v2.reindex(_tr.StudyInstanceUID)[L].reset_index().to_csv(D / "targets.csv", index=False)
if __TARGETS_SHA__:
    # A different label table than steven_v2 (built with --targets): gzip-embedded, or -- too big for a
    # notebook -- read from an attached Kaggle dataset by name. Either way held to the hash the builder
    # computed locally and recorded in provenance.json: the run's ONLY change when judged against step 5.
    if __TARGETS_B64__:
        import gzip as _gz
        (D / "targets_run.csv").write_bytes(_gz.decompress(base64.b64decode(__TARGETS_B64__)))
        _how = "embedded"
    else:
        _hits = []
        for _r, _dd, _fs in os.walk("/kaggle/input"):   # not the image trees: 6 min walking DICOMs, 09-26
            _dd[:] = [x for x in _dd if x not in ("train_series", "test_series", "train_images", "test_images")]
            if __TARGETS_NAME__ in _fs:
                _hits.append(Path(_r) / __TARGETS_NAME__)
        _hits = [h for h in _hits if _sha(h) == __TARGETS_SHA__]
        if not _hits:
            sys.exit(f"no attached {__TARGETS_NAME__} with sha256 {__TARGETS_SHA__[:16]} -- is its dataset attached?")
        shutil.copy(_hits[0], D / "targets_run.csv")
        _how = f"from {_hits[0]}"
    if _sha(D / "targets_run.csv") != __TARGETS_SHA__:
        sys.exit("targets do not match their hash")
    log(f"targets: {__TARGETS_NAME__} ({_how}, sha256 {__TARGETS_SHA__[:16]}) -- NOT steven_v2")
# ⛔ targets.csv stays steven_v2: folds.py STRATIFIES on it, so it defines the partition every 691 read shares.
# The 09-26 dread run overwrote it, the rebuilt fold file hashed differently, and provenance stopped the run.
# The run's own labels go to targets_run.csv and reach nothing but the trainer and the pool filter.
LABELS_CSV = D / ("targets_run.csv" if __TARGETS_SHA__ else "targets.csv")
TARGETS_DF = pd.read_csv(LABELS_CSV).set_index("StudyInstanceUID")
if TARGETS_DF.index.duplicated().any() or not set(L) <= set(TARGETS_DF.columns):
    sys.exit("targets.csv: duplicate studies or missing label columns")

_ref = D / "public_llm_labels" / "lixin73_rsna-knee-llm-report-labels-sol56"
_ref.mkdir(parents=True, exist_ok=True)
shutil.copy(FOUND["labels_llm_gpt56sol.csv"][0], _ref / "labels_llm_gpt56sol.csv")
_check(_ref / "labels_llm_gpt56sol.csv", "lixin73/rsna-knee-llm-report-labels-sol56")

shutil.copy(FOUND["headers.parquet"][0], D / "external" / "dicom_headers_zhukovoleksiy.parquet")
_check(D / "external" / "dicom_headers_zhukovoleksiy.parquet",
       "kaggle kernels output zhukovoleksiy/rsna-metadata-probe")
import site_fingerprint as _sfp
_fp = _sfp.build()
_fp.to_csv(_sfp.OUT, index=False)
import folds as _folds
FOLDS = _folds.build(group_by="site")
FOLDS.to_csv(D / "folds_site.csv", index=False)
_check(D / "folds_site.csv", "fusion/folds.py --group-by site")     # held to the committed record

CTRL = pd.read_csv(REPO / "fusion" / "runs_port_repro" / "oof_all.csv").StudyInstanceUID.drop_duplicates().tolist()
_f0 = set(FOLDS.loc[FOLDS.fold == 0, "StudyInstanceUID"])
_drift = [u for u in CTRL if u not in _f0]
log(f"partition check: {len(CTRL) - len(_drift)}/{len(CTRL)} control studies are in fold 0 ({len(_f0)})")
if _drift:
    sys.exit(f"the rebuilt fold 0 lost {len(_drift)} control studies -- the paired read would be contaminated")

from train_windows import Corpus as _Corpus        # the trainer's own loader and its checks
CORPUS_IDS = set(_Corpus(Path("/kaggle/input"), LAYOUT).ids)
log(f"corpus ({LAYOUT}) holds {len(CORPUS_IDS)} studies")
VAL = [u for u in CTRL if u in CORPUS_IDS]
_pool = sorted(u for u in FOLDS.loc[(FOLDS.fold != 0) & (~FOLDS.is_gold), "StudyInstanceUID"]
               if u in CORPUS_IDS and u in TARGETS_DF.index and TARGETS_DF.loc[u, L].notna().all())
TEACHER_FOLD = __TEACHER_FOLD__
PREDICT_ALL = __PREDICT_ALL__
ALL_DATA = __ALL_DATA__
if ALL_DATA:
    # nothing held out: every non-gold study, fold 0 included. Val = gold (never trained on; disjoint from train).
    TRAIN = sorted(u for u in FOLDS.loc[~FOLDS.is_gold, "StudyInstanceUID"]
                   if u in CORPUS_IDS and u in TARGETS_DF.index and TARGETS_DF.loc[u, L].notna().all())
    VAL = sorted(u for u in FOLDS.loc[FOLDS.is_gold, "StudyInstanceUID"] if u in CORPUS_IDS)
    if set(TRAIN) & set(VAL) or len(VAL) < 50:
        sys.exit(f"all-data: train/gold overlap or only {len(VAL)} gold studies in the corpus")
elif PREDICT_ALL:
    # an outside teacher (e.g. trained on fastMRI+, loaded with --init, --epochs 0): label EVERY training study and the
    # 691; it never saw any of them. Gold is scored by --gold as always.
    VAL = sorted(set(_pool) | set(VAL))
    TRAIN = _pool[:2]                                      # train_windows needs a train list; 0 epochs never use it
elif TEACHER_FOLD:
    # teacher -> student: train on the other three training folds, label this one -- a study it never saw
    _fk = set(FOLDS.loc[FOLDS.fold == TEACHER_FOLD, "StudyInstanceUID"])
    VAL = [u for u in _pool if u in _fk]
    TRAIN = [u for u in _pool if u not in _fk]
else:
    TRAIN = _pool if not __N_TRAIN__ else sorted(np.random.default_rng(__DRAW_SEED__).choice(
        _pool, size=min(__N_TRAIN__, len(_pool)), replace=False).tolist())
if ALL_DATA:
    log(f"ALL DATA: train {len(TRAIN)} (every non-gold study with targets; the {len(_pool)} pool + fold 0) | "
        f"val = the {len(VAL)} gold studies, report only")
else:
    log(f"train {len(TRAIN)} (a seeded draw from a pool of {len(_pool)}) | val {len(VAL)} "
        f"({len(CTRL) - len(VAL)} control studies are not in the corpus)")
RUN.mkdir(parents=True, exist_ok=True)
pd.Series(TRAIN, name="StudyInstanceUID").to_csv(RUN / "train_studies.csv", index=False)
pd.Series(VAL, name="StudyInstanceUID").to_csv(RUN / "val_studies.csv", index=False)
(OUT / "provenance.json").write_text(json.dumps({
    "built_from": BUILT_FROM, "arch": ARCH, "mode": MODE,
    "inputs": {k: {"path": str(v[0]), "sha256": _sha(v[0])} for k, v in FOUND.items()},
    "folds_site_sha256": _sha(D / "folds_site.csv"), "targets_sha256": _sha(LABELS_CSV), "fold_targets_sha256": _sha(D / "targets.csv"),
    "n_train": len(TRAIN), "n_val": len(VAL), "draw_seed": __DRAW_SEED__, "teacher_fold": TEACHER_FOLD,
    "predict_all": PREDICT_ALL, "all_data": ALL_DATA}, indent=2))
'''


TRAIN = r'''
# ============================================================================ #
# Train: fusion/train_windows.py, the only place an arm differs is --arch.
# ============================================================================ #
# The VLM's CPU smoke is ONE process: two fp32 copies of a 2B model do not fit a CPU session's RAM, and
# its DDP path is exercised by the GPU bench's first steps instead.
NPROC = ((torch.cuda.device_count() if torch.cuda.is_available() else (1 if ARCH.startswith("qwen") else 2))
         if LAYOUT == "stage_b" or ARCH.startswith("qwen") else 1)
_launch = ([sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--node_rank", "0",
            "--master_addr", "127.0.0.1", "--master_port", "29511", "--nproc_per_node", str(NPROC)]
           if NPROC > 1 else [sys.executable, "-u"])
if NPROC > 1:
    os.environ["TW_WORKERS"] = str(max(1, (os.cpu_count() or 4) // NPROC))
    if not torch.cuda.is_available():          # CPU smoke: checkpoint like the GPU run, and use every core
        os.environ["TW_FORCE_CKPT"] = "1"
        os.environ["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 4) // NPROC))
if not torch.cuda.is_available():
    # EVERY CPU smoke takes the gradient-checkpointing branch the GPU run takes. 10-02: the CoAtNet student (Stage A,
    # one process, so the flag above was never set) died 8 min into its GPU run on a line only that branch reaches.
    os.environ["TW_FORCE_CKPT"] = "1"
if ARCH.startswith("qwen") and not torch.cuda.is_available():
    # The VLM's CPU smoke: checkpoint as the GPU run does (without it, activations of a 2B model on one
    # CPU process were SIGKILLed after the first step, smoke v3), with one loader worker.
    os.environ["TW_FORCE_CKPT"] = "1"
    os.environ["TW_WORKERS"] = "1"
log(f"launch: {NPROC} process(es), {os.environ.get('TW_WORKERS', '4')} loader workers each")
_cmd = _launch + [str(REPO / "fusion" / "train_windows.py"), "--arch", ARCH,
        "--corpus", "/kaggle/input", "--layout", LAYOUT, "--train", str(RUN / "train_studies.csv"),
        "--val", str(RUN / "val_studies.csv"), "--labels", str(LABELS_CSV),
        "--out", str(RUN), "--weights-dir", str(SCRATCH), "--epochs", str(EPOCHS)]
_cmd += EXTRA
if MODE == "smoke":
    _cmd += ["--limit-train", "16", "--limit-val", "2" if NPROC > 1 else "6", "--k", "2" if NPROC > 1 else "4",
             "--bs", str(2 * NPROC), "--micro", "1",
             "--eval-bs", "1"]
else:
    _cmd += ["--gold", str(D / "train.csv")]
if MODE == "full":
    _cmd += ["--save-weights"]
log("train: " + " ".join(_cmd))
_rc = subprocess.call(_cmd)
if _rc:
    sys.exit(f"train_windows exited {_rc}")
'''


SCORE = r'''
# ============================================================================ #
# Score: paired against runs_port_repro on the same 691 studies, by score_oof.py.
# ============================================================================ #
if MODE == "smoke":
    log("SMOKE: the whole path ran. No score -- 6 studies cannot carry one.")
elif ALL_DATA:
    log("ALL DATA: nothing held out, so no 691 score; gold-58 is in summary.json (report only); the LB judges.")
else:
    for _arm in [RUN] + ([RUN / "avg3"] if (RUN / "avg3" / "oof_all.csv").exists() else []):
        _ctrl = "runs_c6_ortho" if LAYOUT == "stage_b" or ARCH.startswith("qwen") else "runs_port_repro"
        _pair = [] if TEACHER_FOLD or PREDICT_ALL else [str(REPO / "fusion" / _ctrl)]   # not the 691 alone
        _res = subprocess.run([sys.executable, str(REPO / "fusion" / "score_oof.py")] + _pair + [str(_arm)],
                              capture_output=True, text=True, cwd=str(REPO))
        print(_res.stdout, _res.stderr)
        (OUT / f"score_{_arm.name}.txt").write_text(_res.stdout + _res.stderr)
log(f"backbone A/B {ARCH} {MODE} complete")
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
    ap.add_argument("--arch", choices=["ortho", "dinov3l", "coatnet", "resnet34", "qwen35vl"], required=True)
    ap.add_argument("--predict-all", action="store_true", help="eval-only (--epochs 0 with --init): predict every "
                    "training study and the 691 -- an outside teacher's labels")
    ap.add_argument("--teacher-fold", type=int, default=0, choices=[0, 1, 2, 3, 4],
                    help="teacher -> student: train on the other training folds, predict fold K (1-4); 0 = off")
    ap.add_argument("--extra-datasets", nargs="*", default=[], help="more Kaggle datasets to attach (e.g. public weights "
                    "for --init)")
    ap.add_argument("--cpu", action="store_true", help="a CPU session even in full mode (an eval-only check: no GPU quota)")
    ap.add_argument("--train-args", default="", help="extra train_windows flags, one string (e.g. '--bb-lr 3e-4'): "
                    "for an arm that differs from its recipe by a named setting")
    ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="CPU, 16 + 6 studies, one epoch. First.")
    g.add_argument("--bench", action="store_true", help="T4, ONE full epoch, scored. Prices the run.")
    ap.add_argument("--epochs", type=int, default=10, help="full run only")
    ap.add_argument("--time-budget-h", type=float, default=10.3, help="stage_b: training hours inside the 12 h session")
    ap.add_argument("--corpus", choices=["raptor44", "stage_a", "stage_b"], default="raptor44",
                    help="raptor44: the CC0 44-slice corpus (the A/B); stage_a: our 64-slice shards")
    ap.add_argument("--targets-from", default="", help="a Kaggle dataset holding --targets byte for byte: attach it "
                                                         "instead of embedding (notebooks over ~1 MB are refused)")
    ap.add_argument("--targets", type=Path, help="train on this table (StudyInstanceUID + 12 labels) instead of "
                                                 "steven_v2; embedded and hashed. Needs --tag.")
    ap.add_argument("--tag", default="", help="suffix for the kernel slug, so a variant does not overwrite a run")
    ap.add_argument("--all-data", action="store_true", help="train on every non-gold study (4,349), nothing held "
                    "out; val = gold. For a model whose --init already saw them; the LB is its judge")
    ap.add_argument("--n-train", type=int, default=None,
                    help=f"seeded draw from the pool ({N_TRAIN} = the A/B count, the default; qwen35vl: 0); "
                         "0 = the whole pool")
    a = ap.parse_args()
    if a.n_train is None:
        a.n_train = 0 if a.arch.startswith("qwen") else N_TRAIN
    mode = "smoke" if a.smoke else "bench" if a.bench else "full"
    epochs = 1 if mode != "full" else a.epochs
    if a.arch.startswith("qwen"):
        # the smoke skips the fit-budget probe (8 steps at ~90 s each on a CPU) and stops after 2 steps
        extra = ([x for x in QWEN_EXTRA if x != "--fit-budget"] + ["--max-steps", "2", "--eval-windows", "4"]
                 if mode == "smoke" else QWEN_EXTRA + (QWEN_BENCH if mode == "bench" else [])) + \
            ["--time-budget-h", str(a.time_budget_h if mode == "full" else 0.6)]
        epochs = a.epochs if mode == "full" else 1
    elif a.corpus == "stage_b":
        extra = PACKAGE + SPEED + ["--time-budget-h", str(a.time_budget_h if mode == "full" else 0.3)]
    else:
        extra = []
    extra += a.train_args.split()
    if a.all_data and (a.predict_all or a.teacher_fold or a.n_train or a.bench):
        raise SystemExit("--all-data trains on everything: not with --predict-all, --teacher-fold, a --n-train draw "
                         "(pass --n-train 0) or --bench")
    if a.predict_all:
        # An outside teacher labels studies with the weights it brings: nothing may train here, in any mode.
        # (train_windows lets the train list overlap val only at --epochs 0, and skips --fit-budget there.)
        # Read it on the corpus it trained on: --corpus stage_b adds PACKAGE, whose --res 384 sets the eval size.
        if "--init" not in extra:
            raise SystemExit("--predict-all needs the teacher's weights: --train-args '--init <file>' "
                             "(and --extra-datasets for the dataset that holds it)")
        if a.teacher_fold:
            raise SystemExit("--predict-all and --teacher-fold are different runs: pick one")
        epochs = 0

    files = {}
    for rel in EMBED:
        b = (PROJ / rel).read_bytes()
        files[rel] = [base64.b64encode(b).decode(), hashlib.sha256(b).hexdigest()]
    blob = base64.b64encode(json.dumps(files).encode()).decode()
    built_from = _git_head()
    setup = (SETUP.replace("__ARCH__", repr(a.arch)).replace("__MODE__", repr(mode))
                  .replace("__LAYOUT__", repr(a.corpus))
                  .replace("__EXTRA__", repr(extra))
                  .replace("__EPOCHS__", str(epochs)).replace("__BUILT_FROM__", repr(built_from))
                  .replace("__FILES_B64__", repr(blob)).replace("__QWEN_TF__", QWEN_TRANSFORMERS))
    if a.targets and not a.tag:
        raise SystemExit("--targets needs --tag: a run on other labels must not overwrite the steven_v2 run's slug")
    t_b64, t_sha, t_name = "", "", ""
    if a.targets:
        import pandas as pd
        L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
             "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
        t = pd.read_csv(a.targets)
        bad = t[L].isna().any(axis=1).sum() + (~t[L].stack().between(0, 1)).sum()
        if t.StudyInstanceUID.duplicated().any() or bad:
            raise SystemExit(f"{a.targets}: duplicate studies or {bad} missing/out-of-range cells")
        raw = a.targets.read_bytes()
        import gzip
        t_sha, t_name = hashlib.sha256(raw).hexdigest(), a.targets.name
        t_b64 = "" if a.targets_from else base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode()
        if len(t_b64) > 300_000:
            raise SystemExit(f"{a.targets}: {len(t_b64) / 1e3:.0f} kB embedded even gzipped -- the 09-26 push of a 2.1 MB "
                             "notebook was refused (400). Attach its dataset with --targets-from.")
        print(f"targets {a.targets} ({len(t)} studies, sha256 {t_sha[:16]}) replace steven_v2")
    prep = (PREP.replace("__DRAW_SEED__", str(DRAW_SEED)).replace("__N_TRAIN__", str(a.n_train))
                .replace("__TEACHER_FOLD__", str(a.teacher_fold)).replace("__PREDICT_ALL__", str(a.predict_all))
                .replace("__ALL_DATA__", str(a.all_data))
                .replace("__TARGETS_B64__", repr(t_b64)).replace("__TARGETS_SHA__", repr(t_sha))
                .replace("__TARGETS_NAME__", repr(t_name)))
    cells = [setup, prep, TRAIN, SCORE]
    for k, c in enumerate(cells):
        compile(c, f"cell{k}", "exec")              # fail here, not after the queue wait

    # the A/B keeps its slugs; a Stage A run is a different experiment and gets its own
    slug = {"raptor44": f"rsna-knee-backbone-ab-{a.arch}", "stage_a": f"rsna-knee-c6-train-{a.arch}",
            "stage_b": f"rsna-knee-c7-train-{a.arch}"}[a.corpus] + (f"-{a.tag}" if a.tag else "") \
        + (f"-t{a.teacher_fold}" if a.teacher_fold else "") + ("-pa" if a.predict_all else "") \
        + ("-all" if a.all_data else "")
    datasets = ((DATASETS if a.corpus == "raptor44" else DATASETS[2:]) + ([a.targets_from] if a.targets_from else [])
                + a.extra_datasets)
    kernels = KERNELS + {"raptor44": [], "stage_a": STAGE_A_SHARDS, "stage_b": STAGE_B_SHARDS}[a.corpus]
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": c.strip("\n").splitlines(keepends=True)} for c in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    body = json.dumps(nb, indent=1)
    if len(body.encode()) > 1_000_000:
        # Kaggle answers a bare "400 Bad Request" to a notebook over ~1 MB (09-26 at 2.1 MB; 09-28 at 1.06 MB, when
        # embedded student targets pushed an ~0.87 MB notebook over). Fail here, with the way out.
        raise SystemExit(f"notebook {len(body.encode()) / 1e6:.2f} MB > 1 MB: Kaggle refuses it with a 400"
                         + (" -- attach the targets with --targets-from <dataset>" if t_b64 else ""))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{slug}.ipynb").write_text(body, encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": mode != "smoke" and not a.cpu, "enable_tpu": False,
            # internet: Ortho's checkpoint comes from the authors' GitHub (sha-checked), the other
            # encoders from the HF Hub. A training run, not a submission.
            "enable_internet": True, "keywords": [],
            "dataset_sources": datasets, "kernel_sources": kernels,
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": []}
    if mode != "smoke" and not a.cpu:          # a machine_shape makes Kaggle count it as a GPU session, even with enable_gpu off
        meta["machine_shape"] = "NvidiaTeslaT4"
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (slug + '.ipynb')}  arch={a.arch} corpus={a.corpus} "
          f"n_train={'ALL (4,349, nothing held out)' if a.all_data else a.n_train or 'pool'} "
          f"mode={mode} epochs={epochs}  from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")
    print(f"push:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
