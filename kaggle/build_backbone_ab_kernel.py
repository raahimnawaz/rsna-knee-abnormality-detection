"""Build a Kaggle GPU training kernel for one encoder: pinned inputs, embedded sources, paired scoring.

    python kaggle/build_backbone_ab_kernel.py --arch convnext_small --corpus stage_b --out /tmp/ab --smoke
    python kaggle/build_backbone_ab_kernel.py --arch convnext_small --corpus stage_b --folds 0 --n-train 0         --recipe v3 --targets <targets.csv> --tag <name> --out /tmp/ab
    python -m kaggle kernels push -p /tmp/ab
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
    "fusion/score_oof.py",
    "fusion/folds.py",
    "pipeline/site_fingerprint.py",
    "pipeline/provenance.py",
    "fusion/metrics.py",
    "provenance/inputs.json",
    "fusion/runs_port_repro/oof_all.csv",
    "fusion/runs_c6_ortho/oof_all.csv",
]
DATASETS = ["dreaddevelopment/knee-raptor-corpus",
            "dreaddevelopment/knee-raptor-corpus-ext",
            "stevenleehans/rsna-knee-llm-report-labels",
            "lixin73/rsna-knee-llm-report-labels-sol56"]
KERNELS = ["zhukovoleksiy/rsna-metadata-probe"]
STAGE_A_SHARDS = [f"{OWNER}/rsna-knee-stage-a-s{i}of4" for i in range(4)]
STAGE_B_SHARDS = [f"{OWNER}/rsna-knee-stage-b-s{i}of8" for i in range(8)]
K_SHARDS = [f"{OWNER}/rsna-knee-kcache-s{i}of5" for i in range(5)]
SMALL = ("convnext_tiny", "convnext_small")
SMALL_PACKAGE = ["--res", "384", "--drop-path", "0.1", "--bb-lr", "1e-4", "--pos-emb", "--fit-budget",
                 "--conf-weight", "none"]
COATNET_PACKAGE = ["--res", "384", "--drop-path", "0.1", "--bb-lr", "3e-5", "--pos-emb", "--fit-budget",
                   "--conf-weight", "none"]
V3_TRAIN_ARGS = ["--allow-missing-labels", "--spatial-attn", "--mirror", "0.5", "--mirror-tta", "--k", "24", "--micro", "1"]
V3_EPOCHS = 16
K_TRAIN_ARGS = ["--allow-missing-labels", "--only-weak4", "--spatial-attn", "--k", "24", "--micro", "1"]
N_TRAIN, DRAW_SEED = 2871, 20260923


SETUP = r'''
import base64, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ARCH = __ARCH__
MODE = __MODE__
LAYOUT = __LAYOUT__
EXTRA = __EXTRA__
EPOCHS = __EPOCHS__
FOLD_LIST = __FOLD_LIST__
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
REPO = SCRATCH / "ab_repo"
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


PREP = r'''
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
for _p in (REPO, REPO / "pipeline", REPO / "fusion"):
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
    if __TARGETS_B64__:
        import gzip as _gz
        (D / "targets_run.csv").write_bytes(_gz.decompress(base64.b64decode(__TARGETS_B64__)))
        _how = "embedded"
    else:
        _hits = []
        for _r, _dd, _fs in os.walk("/kaggle/input"):
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
_check(D / "folds_site.csv", "fusion/folds.py --group-by site")

CTRL = pd.read_csv(REPO / "fusion" / "runs_port_repro" / "oof_all.csv").StudyInstanceUID.drop_duplicates().tolist()
_f0 = set(FOLDS.loc[FOLDS.fold == 0, "StudyInstanceUID"])
_drift = [u for u in CTRL if u not in _f0]
log(f"partition check: {len(CTRL) - len(_drift)}/{len(CTRL)} control studies are in fold 0 ({len(_f0)})")
if _drift:
    sys.exit(f"the rebuilt fold 0 lost {len(_drift)} control studies -- the paired read would be contaminated")

from train_windows import Corpus as _Corpus
CORPUS_IDS = set(_Corpus(Path("/kaggle/input"), LAYOUT).ids)
log(f"corpus ({LAYOUT}) holds {len(CORPUS_IDS)} studies")
VAL = [u for u in CTRL if u in CORPUS_IDS]
MASKED = "--allow-missing-labels" in EXTRA
_ok = TARGETS_DF[L].notna().any(axis=1) if MASKED else TARGETS_DF[L].notna().all(axis=1)
HAS_TARGETS = set(_ok.index[_ok])
_pool = sorted(u for u in FOLDS.loc[(FOLDS.fold != 0) & (~FOLDS.is_gold), "StudyInstanceUID"]
               if u in CORPUS_IDS and u in HAS_TARGETS)
TEACHER_FOLD = __TEACHER_FOLD__
PREDICT_ALL = __PREDICT_ALL__
ALL_DATA = __ALL_DATA__
if ALL_DATA:
    TRAIN = sorted(u for u in FOLDS.loc[~FOLDS.is_gold, "StudyInstanceUID"] if u in CORPUS_IDS and u in HAS_TARGETS)
    VAL = sorted(u for u in FOLDS.loc[FOLDS.is_gold, "StudyInstanceUID"] if u in CORPUS_IDS)
    if set(TRAIN) & set(VAL) or len(VAL) < 50:
        sys.exit(f"all-data: train/gold overlap or only {len(VAL)} gold studies in the corpus")
elif PREDICT_ALL:
    VAL = sorted(u for u in FOLDS.loc[~FOLDS.is_gold, "StudyInstanceUID"] if u in CORPUS_IDS)
    TRAIN = _pool[:2]
elif TEACHER_FOLD:
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
FOLD_RUNS = {}
for _k in FOLD_LIST:
    _fk = set(FOLDS.loc[FOLDS.fold == _k, "StudyInstanceUID"])
    _nong = [u for u in FOLDS.loc[~FOLDS.is_gold, "StudyInstanceUID"] if u in CORPUS_IDS]
    _tr = sorted(u for u in _nong if u not in _fk and u in HAS_TARGETS)
    _va = sorted(u for u in _nong if u in _fk)
    if set(_tr) & set(_va) or not _tr or not _va:
        sys.exit(f"fold {_k}: train/val overlap or an empty list")
    _d = RUN / f"fold{_k}"
    _d.mkdir(parents=True, exist_ok=True)
    pd.Series(_tr, name="StudyInstanceUID").to_csv(_d / "train_studies.csv", index=False)
    pd.Series(_va, name="StudyInstanceUID").to_csv(_d / "val_studies.csv", index=False)
    FOLD_RUNS[_k] = _d
    log(f"fold {_k}: train {len(_tr)} | val {len(_va)}")
(OUT / "provenance.json").write_text(json.dumps({
    "built_from": BUILT_FROM, "arch": ARCH, "mode": MODE,
    "inputs": {k: {"path": str(v[0]), "sha256": _sha(v[0])} for k, v in FOUND.items()},
    "folds_site_sha256": _sha(D / "folds_site.csv"), "targets_sha256": _sha(LABELS_CSV), "fold_targets_sha256": _sha(D / "targets.csv"),
    "n_train": len(TRAIN), "n_val": len(VAL), "draw_seed": __DRAW_SEED__, "teacher_fold": TEACHER_FOLD,
    "predict_all": PREDICT_ALL, "all_data": ALL_DATA, "folds": FOLD_LIST, "masked_targets": MASKED}, indent=2))
'''


TRAIN = r'''
NPROC = (torch.cuda.device_count() if torch.cuda.is_available() else 2) if LAYOUT in ("stage_b", "stage_k") else 1
_launch = ([sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--node_rank", "0",
            "--master_addr", "127.0.0.1", "--master_port", "29511", "--nproc_per_node", str(NPROC)]
           if NPROC > 1 else [sys.executable, "-u"])
if NPROC > 1:
    os.environ["TW_WORKERS"] = str(max(1, (os.cpu_count() or 4) // NPROC))
    if not torch.cuda.is_available():
        os.environ["TW_FORCE_CKPT"] = "1"
        os.environ["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 4) // NPROC))
if not torch.cuda.is_available():
    os.environ["TW_FORCE_CKPT"] = "1"
log(f"launch: {NPROC} process(es), {os.environ.get('TW_WORKERS', '4')} loader workers each")
def _train(run_dir):
    _cmd = _launch + [str(REPO / "fusion" / "train_windows.py"), "--arch", ARCH,
            "--corpus", "/kaggle/input", "--layout", LAYOUT, "--train", str(run_dir / "train_studies.csv"),
            "--val", str(run_dir / "val_studies.csv"), "--labels", str(LABELS_CSV),
            "--out", str(run_dir), "--weights-dir", str(SCRATCH), "--epochs", str(EPOCHS)]
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
        sys.exit(f"train_windows exited {_rc} ({run_dir.name})")


if FOLD_RUNS:
    for _k, _d in FOLD_RUNS.items():
        log(f"===== fold {_k} of {sorted(FOLD_RUNS)} =====")
        _train(_d)
else:
    _train(RUN)
'''


SCORE = r'''
if MODE == "smoke":
    log("SMOKE: the whole path ran. No score -- 6 studies cannot carry one.")
elif FOLD_RUNS:
    for _sub in ("", "avg3"):
        _parts = [_d / _sub / "oof_all.csv" for _d in FOLD_RUNS.values()]
        if not all(p.exists() for p in _parts):
            continue
        _all = RUN / ("cv_" + (_sub or "last"))
        _all.mkdir(exist_ok=True)
        pd.concat([pd.read_csv(p) for p in _parts]).to_csv(_all / "oof_all.csv", index=False)
        _jobs = [[str(_all)]] + ([[str(REPO / "fusion" / "runs_c6_ortho"), str(FOLD_RUNS[0] / _sub)]] if 0 in FOLD_RUNS else [])
        for _j in _jobs:
            _res = subprocess.run([sys.executable, str(REPO / "fusion" / "score_oof.py")] + _j,
                                  capture_output=True, text=True, cwd=str(REPO))
            print(_res.stdout, _res.stderr)
            with open(OUT / f"score_cv_{_sub or 'last'}.txt", "a") as _f:
                _f.write(_res.stdout + _res.stderr)
    _gold = [pd.read_csv(_d / "oof_gold.csv") for _d in FOLD_RUNS.values() if (_d / "oof_gold.csv").exists()]
    if _gold:
        pd.concat(_gold).groupby("StudyInstanceUID", as_index=False)[L].mean().to_csv(RUN / "oof_gold_foldmean.csv", index=False)
elif ALL_DATA:
    log("ALL DATA: nothing held out, so no 691 score; gold-58 is in summary.json (report only); the LB judges.")
else:
    for _arm in [RUN] + ([RUN / "avg3"] if (RUN / "avg3" / "oof_all.csv").exists() else []):
        _ctrl = "runs_c6_ortho" if LAYOUT in ("stage_b", "stage_k") else "runs_port_repro"
        _pair = [] if TEACHER_FOLD or PREDICT_ALL else [str(REPO / "fusion" / _ctrl)]
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
    ap.add_argument("--arch", choices=["coatnet", "resnet34", *SMALL], required=True)
    ap.add_argument("--folds", default="", help="fold CV in one kernel, e.g. 0,1,2,3,4: model k trains on the non-gold "
                    "studies outside fold k and predicts fold k; outputs runs_<arch>/fold<k>/")
    ap.add_argument("--predict-all", action="store_true", help="eval-only (--epochs 0 with --init): predict every "
                    "training study and the 691 -- an outside teacher's labels")
    ap.add_argument("--teacher-fold", type=int, default=0, choices=[0, 1, 2, 3, 4],
                    help="teacher -> student: train on the other training folds, predict fold K (1-4); 0 = off")
    ap.add_argument("--extra-datasets", nargs="*", default=[], help="more Kaggle datasets to attach (e.g. public weights "
                    "for --init)")
    ap.add_argument("--cpu", action="store_true", help="a CPU session even in full mode (an eval-only check: no GPU quota)")
    ap.add_argument("--train-args", default="", help="extra train_windows flags, one string (e.g. '--bb-lr 3e-4'): "
                    "for an arm that differs from its recipe by a named setting")
    ap.add_argument("--recipe", choices=["v3", "k"], help=f"v3: {' '.join(V3_TRAIN_ARGS)} at {V3_EPOCHS} epochs "
                    "(convnext_small on stage_b; pass --targets and one fold per kernel)")
    ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="CPU, 16 + 6 studies, one epoch (two for the small conv "
                   "backbones, so the averaged weights are reached). First.")
    g.add_argument("--bench", action="store_true", help="T4, ONE full epoch, scored. Prices the run.")
    ap.add_argument("--epochs", type=int, default=10, help="full run only")
    ap.add_argument("--time-budget-h", type=float, default=10.3, help="stage_b: training hours inside the 12 h session")
    ap.add_argument("--corpus", choices=["raptor44", "stage_a", "stage_b", "stage_k"], default="raptor44",
                    help="raptor44: the CC0 44-slice corpus (the A/B); stage_a: our 64-slice shards")
    ap.add_argument("--targets-from", default="", help="a Kaggle dataset holding --targets byte for byte: attach it "
                                                         "instead of embedding (notebooks over ~1 MB are refused)")
    ap.add_argument("--targets", type=Path, help="train on this table (StudyInstanceUID + 12 labels) instead of "
                                                 "steven_v2; embedded and hashed. Needs --tag.")
    ap.add_argument("--tag", default="", help="suffix for the kernel slug, so a variant does not overwrite a run")
    ap.add_argument("--all-data", action="store_true", help="train on every non-gold study (4,349), nothing held "
                    "out; val = gold. For a model whose --init already saw them; the LB is its judge")
    ap.add_argument("--n-train", type=int, default=N_TRAIN,
                    help=f"seeded draw from the pool ({N_TRAIN} = the A/B count); 0 = the whole pool")
    a = ap.parse_args()
    if a.recipe == "k":
        if a.arch != "convnext_small" or a.corpus != "stage_k" or not a.targets:
            raise SystemExit("--recipe k is convnext_small on --corpus stage_k with --targets")
        a.train_args = " ".join(K_TRAIN_ARGS + a.train_args.split())
        a.epochs = V3_EPOCHS
    if a.recipe == "v3":
        if a.arch != "convnext_small" or a.corpus != "stage_b" or not a.targets:
            raise SystemExit("--recipe v3 is convnext_small on --corpus stage_b with --targets")
        a.train_args = " ".join(V3_TRAIN_ARGS + a.train_args.split())
        a.epochs = V3_EPOCHS
    mode = "smoke" if a.smoke else "bench" if a.bench else "full"
    epochs = a.epochs if mode == "full" else 2 if mode == "smoke" and a.arch in SMALL else 1
    if a.corpus in ("stage_b", "stage_k"):
        if not a.predict_all and a.arch not in (*SMALL, "coatnet"):
            raise SystemExit(f"--corpus {a.corpus}: convnext_* or coatnet")
        folds = [int(x) for x in a.folds.split(",") if x != ""]
        budget = (a.time_budget_h / max(len(folds), 1)) if mode == "full" else 0.3
        extra = (["--res", "384"] if a.predict_all else SMALL_PACKAGE if a.arch in SMALL else COATNET_PACKAGE) \
            + ["--time-budget-h", str(round(budget, 3))]
    else:
        extra = []
    extra += a.train_args.split()
    folds = [int(x) for x in a.folds.split(",") if x != ""]
    if folds and (a.corpus not in ("stage_b", "stage_k") or a.all_data or a.predict_all or a.teacher_fold or a.n_train
                  or sorted(set(folds)) != folds or not set(folds) <= set(range(5))):
        raise SystemExit("--folds: an increasing list from 0..4, on --corpus stage_b / stage_k with --n-train 0, and not with "
                         "--all-data, --predict-all or --teacher-fold")
    if a.all_data and (a.predict_all or a.teacher_fold or a.n_train or a.bench):
        raise SystemExit("--all-data trains on everything: not with --predict-all, --teacher-fold, a --n-train draw "
                         "(pass --n-train 0) or --bench")
    if a.predict_all:
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
                  .replace("__FOLD_LIST__", repr(folds))
                  .replace("__FILES_B64__", repr(blob)))
    if a.targets and not a.tag:
        raise SystemExit("--targets needs --tag: a run on other labels must not overwrite the steven_v2 run's slug")
    t_b64, t_sha, t_name = "", "", ""
    if a.targets:
        import numpy as np
        import pandas as pd
        L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
             "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
        t = pd.read_csv(a.targets)
        v = t[L].to_numpy(float)
        masked = "--allow-missing-labels" in extra
        bad = int(((v < 0) | (v > 1)).sum()) + int(np.isnan(v).all(axis=1).sum() if masked else np.isnan(v).any(axis=1).sum())
        if t.StudyInstanceUID.duplicated().any() or bad:
            raise SystemExit(f"{a.targets}: duplicate studies or {bad} missing/out-of-range cells"
                             + ("" if masked else " (masked targets need --train-args '--allow-missing-labels')"))
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
        compile(c, f"cell{k}", "exec")

    name = a.arch.replace("_", "-")
    slug = {"raptor44": f"rsna-knee-backbone-ab-{name}", "stage_a": f"rsna-knee-c6-train-{name}",
            "stage_b": f"rsna-knee-c7-train-{name}", "stage_k": f"rsna-knee-k-train-{name}"}[a.corpus] + (f"-{a.tag}" if a.tag else "") \
        + (f"-t{a.teacher_fold}" if a.teacher_fold else "") + ("-pa" if a.predict_all else "") \
        + ("-all" if a.all_data else "") + ("-cv" if folds else "")
    datasets = ((DATASETS if a.corpus == "raptor44" else DATASETS[2:]) + ([a.targets_from] if a.targets_from else [])
                + a.extra_datasets)
    kernels = KERNELS + {"raptor44": [], "stage_a": STAGE_A_SHARDS, "stage_b": STAGE_B_SHARDS,
                         "stage_k": K_SHARDS}[a.corpus]
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": c.strip("\n").splitlines(keepends=True)} for c in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    body = json.dumps(nb, indent=1)
    if len(body.encode()) > 1_000_000:
        raise SystemExit(f"notebook {len(body.encode()) / 1e6:.2f} MB > 1 MB: Kaggle refuses it with a 400"
                         + (" -- attach the targets with --targets-from <dataset>" if t_b64 else ""))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{slug}.ipynb").write_text(body, encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": mode != "smoke" and not a.cpu, "enable_tpu": False,
            "enable_internet": True, "keywords": [],
            "dataset_sources": datasets, "kernel_sources": kernels,
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": []}
    if mode != "smoke" and not a.cpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (slug + '.ipynb')}  arch={a.arch} corpus={a.corpus} "
          f"n_train={'ALL (4,349, nothing held out)' if a.all_data else a.n_train or 'pool'} "
          + (f"folds={folds} " if folds else "") +
          f"mode={mode} epochs={epochs}  from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")
    print(f"push:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
