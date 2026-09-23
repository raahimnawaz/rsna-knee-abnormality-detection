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
    "fusion/score_oof.py",
    "fusion/folds.py",
    "pipeline/site_fingerprint.py",
    "pipeline/provenance.py",
    "extractor/metrics.py",
    "provenance/inputs.json",
    "fusion/runs_port_repro/oof_all.csv",      # the control: 691 fold-0 studies, 0.7597
]
DATASETS = ["dreaddevelopment/knee-raptor-corpus",           # 3,200 studies, CC0
            "dreaddevelopment/knee-raptor-corpus-ext",       # 1,207 studies, CC0
            "stevenleehans/rsna-knee-llm-report-labels",     # llm_labels_v2.csv -> targets.csv
            "lixin73/rsna-knee-llm-report-labels-sol56"]     # the scoring reference
KERNELS = ["zhukovoleksiy/rsna-metadata-probe"]              # headers.parquet -> site folds
STAGE_A_SHARDS = [f"{OWNER}/rsna-knee-stage-a-s{i}of4" for i in range(4)]   # build_stage_a_kernel.py
N_TRAIN, DRAW_SEED = 2871, 20260923


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
EPOCHS = __EPOCHS__
BUILT_FROM = __BUILT_FROM__
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


import torch, timm
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
TARGETS_DF = pd.read_csv(D / "targets.csv").set_index("StudyInstanceUID")

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
TRAIN = _pool if not __N_TRAIN__ else sorted(np.random.default_rng(__DRAW_SEED__).choice(
    _pool, size=min(__N_TRAIN__, len(_pool)), replace=False).tolist())
log(f"train {len(TRAIN)} (a seeded draw from a pool of {len(_pool)}) | val {len(VAL)} "
    f"({len(CTRL) - len(VAL)} control studies are not in the corpus)")
RUN.mkdir(parents=True, exist_ok=True)
pd.Series(TRAIN, name="StudyInstanceUID").to_csv(RUN / "train_studies.csv", index=False)
pd.Series(VAL, name="StudyInstanceUID").to_csv(RUN / "val_studies.csv", index=False)
(OUT / "provenance.json").write_text(json.dumps({
    "built_from": BUILT_FROM, "arch": ARCH, "mode": MODE,
    "inputs": {k: {"path": str(v[0]), "sha256": _sha(v[0])} for k, v in FOUND.items()},
    "folds_site_sha256": _sha(D / "folds_site.csv"), "targets_sha256": _sha(D / "targets.csv"),
    "n_train": len(TRAIN), "n_val": len(VAL), "draw_seed": __DRAW_SEED__}, indent=2))
'''


TRAIN = r'''
# ============================================================================ #
# Train: fusion/train_windows.py, the only place an arm differs is --arch.
# ============================================================================ #
_cmd = [sys.executable, "-u", str(REPO / "fusion" / "train_windows.py"), "--arch", ARCH,
        "--corpus", "/kaggle/input", "--layout", LAYOUT, "--train", str(RUN / "train_studies.csv"),
        "--val", str(RUN / "val_studies.csv"), "--labels", str(D / "targets.csv"),
        "--out", str(RUN), "--weights-dir", str(SCRATCH), "--epochs", str(EPOCHS)]
if MODE == "smoke":
    _cmd += ["--limit-train", "16", "--limit-val", "6", "--k", "4", "--bs", "2", "--micro", "2",
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
else:
    for _arm in [RUN] + ([RUN / "avg3"] if (RUN / "avg3" / "oof_all.csv").exists() else []):
        _res = subprocess.run([sys.executable, str(REPO / "fusion" / "score_oof.py"),
                               str(REPO / "fusion" / "runs_port_repro"), str(_arm)],
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
    ap.add_argument("--arch", choices=["ortho", "dinov3l", "coatnet"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="CPU, 16 + 6 studies, one epoch. First.")
    g.add_argument("--bench", action="store_true", help="T4, ONE full epoch, scored. Prices the run.")
    ap.add_argument("--epochs", type=int, default=10, help="full run only")
    ap.add_argument("--corpus", choices=["raptor44", "stage_a"], default="raptor44",
                    help="raptor44: the CC0 44-slice corpus (the A/B); stage_a: our 64-slice shards")
    ap.add_argument("--n-train", type=int, default=N_TRAIN,
                    help=f"seeded draw from the pool ({N_TRAIN} = the A/B count); 0 = the whole pool")
    a = ap.parse_args()
    mode = "smoke" if a.smoke else "bench" if a.bench else "full"
    epochs = 1 if mode != "full" else a.epochs

    files = {}
    for rel in EMBED:
        b = (PROJ / rel).read_bytes()
        files[rel] = [base64.b64encode(b).decode(), hashlib.sha256(b).hexdigest()]
    blob = base64.b64encode(json.dumps(files).encode()).decode()
    built_from = _git_head()
    setup = (SETUP.replace("__ARCH__", repr(a.arch)).replace("__MODE__", repr(mode))
                  .replace("__LAYOUT__", repr(a.corpus))
                  .replace("__EPOCHS__", str(epochs)).replace("__BUILT_FROM__", repr(built_from))
                  .replace("__FILES_B64__", repr(blob)))
    prep = PREP.replace("__DRAW_SEED__", str(DRAW_SEED)).replace("__N_TRAIN__", str(a.n_train))
    cells = [setup, prep, TRAIN, SCORE]
    for k, c in enumerate(cells):
        compile(c, f"cell{k}", "exec")              # fail here, not after the queue wait

    # the A/B keeps its slugs; a Stage A run is a different experiment and gets its own
    slug = f"rsna-knee-backbone-ab-{a.arch}" if a.corpus == "raptor44" else f"rsna-knee-c6-train-{a.arch}"
    datasets = DATASETS if a.corpus == "raptor44" else DATASETS[2:]
    kernels = KERNELS + ([] if a.corpus == "raptor44" else STAGE_A_SHARDS)
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": c.strip("\n").splitlines(keepends=True)} for c in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": mode != "smoke", "enable_tpu": False,
            # internet: Ortho's checkpoint comes from the authors' GitHub (sha-checked), the other
            # encoders from the HF Hub. A training run, not a submission.
            "enable_internet": True, "keywords": [],
            "dataset_sources": datasets, "kernel_sources": kernels,
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": []}
    if mode != "smoke":
        meta["machine_shape"] = "NvidiaTeslaT4"
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (slug + '.ipynb')}  arch={a.arch} corpus={a.corpus} n_train={a.n_train or 'pool'} "
          f"mode={mode} epochs={epochs}  from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")
    print(f"push:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
