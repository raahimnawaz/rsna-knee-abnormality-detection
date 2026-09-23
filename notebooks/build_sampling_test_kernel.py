"""The sampling test: is the INPUT why our own models never reach the downloaded ones?

    python notebooks/build_sampling_test_kernel.py --out /tmp/st --smoke    # CPU, no GPU quota
    python -m kaggle kernels push -p /tmp/st
    python notebooks/build_sampling_test_kernel.py --out /tmp/st            # the real run, T4
    python -m kaggle kernels push -p /tmp/st

THE QUESTION. Everything trained here scores 0.6924-0.7710 and everything downloaded 0.8469-0.8522
(`fusion/archive/pixel_fidelity.py`). `DECISIONS.md` D3 put the last 0.0856 of that on "our pixels
are a third-party NIfTI repackaging". The record does not support that reading:

  * the fork's own 20 members, run on OUR NIfTI files through their reader, score 0.8457 on the
    report instrument and 0.8516 on gold-47 -- the same as on their own pixels, 0.8453 / 0.8516
    (`docs/archive/IMPROVEMENTS.md` §3m). The files are fine.
  * the run D3 was measured on, `runs_port_repro`, trained on `tiles336_pilk`: shape
    (3599, 6, 3, 336, 336), i.e. ONE window of three adjacent slices per series, at depth 0.5
    (`slot_cache.py:130,291-293`). The fork samples TWELVE slices per series across 20-80% of the
    stack and trains on a random group of three per step (`pilkwang_pixels.py:66-70`, its
    notebook's training loop). `contract_audit.py` printed "slices 12 / band 0.2-0.8: PASS" for
    the port because it read those two fields off `pilkwang_pixels`, which `train_port` never
    imports -- the exact false PASS its own comment warns about for `crop_mm`.

So the input differed, and nothing has tested whether that difference is the gap.

WHAT THIS RUN CHANGES, AND WHAT IT HOLDS FIXED. One variable: the pixels. They are built by the
fork's OWN cells, lifted verbatim from `notebooks/submissions/rsna-knee-f6-two-arm.ipynb` (header
pass, slot choice, slice ordering, `read_slot`, laterality, `build_cache`), under the pixel contract
read from the members' own manifest via its own `adopt_config_globals` -- 336 px, 12 slices, groups
of 3, 130 mm, band 0.2-0.8, native rules. Everything else is `runs_port_repro` exactly: the same
`fusion/train_port.py` (whose multi-group branch only fires on a cache like this one), fold 0 of the
same site-grouped folds, `steven_v2` targets, gold excluded, 30 epochs, last epoch, one seed, one
model, 2,871 training studies, and the same 691 validation studies, scored paired on the same
`lixin_gpt56` instrument by `fusion/score_oof.py`.

Known residue, stated in advance: the decode is the fork's DICOM reader, not `slot_cache`'s NIfTI
path, so a gain here is "our cache", sampling and reader together. §3m says the source files cost
nothing measurable; separating reader from sampling is a follow-up only if this lands in between.

⛔ THE BAR IS OUR OWN RUN, NOT THEIR NUMBER. `0.8453` is a 4-seed mean per study; and per the fork's
notebook (whose loop is the best record of how the weights were trained, though not an exact one)
the holdout is report-hash grouped, NOT site-grouped (§2j measured ~0.024 of site leakage in
ungrouped folds), with the best epoch chosen on that same holdout. None of that is in this run, so matching it is not the
test. The test is the paired delta against `runs_port_repro` on the same studies.

THE RULE, PRE-REGISTERED HERE BEFORE THE RUN (gap = 0.8453 - 0.7597 = 0.0856, commit 5296563; the
paired SD on this instrument has been ~0.009):

    delta >= +0.043   the input is the main cause. Our own training reaches public strength on
                      public-style input; the C6 coverage direction is right, D3's reason is not.
    delta <= +0.015   the input is not the main cause. What remains is training and evaluation:
                      site-grouped folds, one seed vs four, last epoch vs best-on-holdout, labels.
    in between        both. Split reader from sampling next.

COST. The full run decodes ~3,560 studies x 6 series x 12 slices into a ~29 GB memmap on scratch
disk (it does not fit in RAM, which is the one patch to the fork's `build_cache`), then trains
`train_port` for 30 epochs on a T4 in fp32. Budget 4-6 GPU-hours. The kernel refuses up front if
scratch disk is short, if the rebuilt fold 0 does not contain every control study, or if slice
ordering or decode hit their time budgets -- each of those would make the delta mean something else.

**Smoke it first** (`--smoke`, CPU): the whole path on 64 + 16 studies and one epoch. It also prints
free scratch space, which is the one thing about the machine the full run depends on and nothing in
this repo has measured.
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
# Kaggle derives the slug from the title, so they are kept identical (build_f6_rad_kernel.py).
TITLE = SLUG = "rsna-knee-sampling-test"

SRC_NB = PROJ / "notebooks" / "submissions" / "rsna-knee-f6-two-arm.ipynb"
SRC_META = PROJ / "notebooks" / "submissions" / "f6-two-arm-kernel-metadata.json"

# Shipped into the kernel byte-for-byte, so the run uses this commit's code and not the stale
# `raahimnawaz/rsna-knee-code` Dataset. Hash-checked after they are written out.
EMBED = [
    "fusion/train_port.py",
    "fusion/pilkwang_model.py",
    "fusion/score_oof.py",
    "fusion/folds.py",
    "pipeline/site_fingerprint.py",
    "extractor/metrics.py",
    "pipeline/provenance.py",
    "provenance/inputs.json",                  # what the inputs must hash to, once recorded
    "fusion/runs_port_repro/oof_all.csv",      # the control: 691 fold-0 studies, 0.7597
]

# The fork cells this run needs, found by what they define. Each anchor must hit exactly one code
# cell or the build refuses -- the build_f6_rad_kernel.py rule. Kept in notebook order.
FORK_ANCHORS = [
    "T0 = time.time()",              # imports, constants, SLOTS
    "def find_root(",                # log, ROOT, plan_cache
    "def walk(",                     # header pass, laterality
    "def pick_slots(",
    "def read_slot(",                # order_slices, read_slot
    "def normalise_laterality(",
    "def build_cache(",
    "def adopt_config_globals(",     # find_weights, WeightsError
]

# The one change to the fork's code. At 12 slices the training corpus is ~29 GB, past a Kaggle
# machine's RAM, so the cache is a memmap on scratch disk: same dtype, shape and zero fill.
CACHE_OLD = "    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)\n"
CACHE_NEW = ("    # sampling test: ~29 GB at 12 slices does not fit in RAM, so this is a memmap on\n"
             "    # scratch disk. Same dtype, shape and zero fill; nothing else in this cell changed.\n"
             "    cache = np.lib.format.open_memmap(\n"
             "        CACHE_NPY, mode=\"w+\", dtype=np.uint8,\n"
             "        shape=(len(studies), N_SLOT, CACHE_SLICES, IMG, IMG))\n")

DATASETS = ["pilkwang/rsna-knee-weights",                  # the members' manifest = pixel contract
            "stevenleehans/rsna-knee-llm-report-labels",    # llm_labels_v2.csv -> targets.csv
            "lixin73/rsna-knee-llm-report-labels-sol56"]    # the scoring reference
KERNELS = ["zhukovoleksiy/rsna-metadata-probe"]             # headers.parquet -> site folds


SETUP = r'''
# ============================================================================ #
# Sampling test -- setup. Built by notebooks/build_sampling_test_kernel.py;
# read that docstring for the question and the pre-registered rule.
# ============================================================================ #
import base64, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

SMOKE = __SMOKE__
BUILT_FROM = __BUILT_FROM__
N_TRAIN = 2871            # runs_port_repro.log: "fold 0: train 2,871 / val 691"
N_VAL = 691
# 64, not 32: at batch 8 one epoch of 32 is 4 steps, and train_port's OneCycleLR (pct_start 0.25)
# divides by zero below 8. Found by the off-Kaggle smoke of this very file.
SMOKE_TRAIN, SMOKE_VAL = 64, 16
TAG = "forksample"
STUDY_BYTES = 6 * 12 * 336 * 336


def _free_gb(p):
    try:
        return shutil.disk_usage(p).free / 1e9
    except OSError:
        return -1.0


_need = (SMOKE_TRAIN + SMOKE_VAL if SMOKE else N_TRAIN + N_VAL) * STUDY_BYTES / 1e9
_cands = [p for p in ("/kaggle/tmp", "/kaggle/temp", "/tmp") if os.path.isdir(p)]
for _p in _cands + ["/kaggle/working"]:
    print(f"scratch candidate {_p}: {_free_gb(_p):.1f} GB free", flush=True)
SCRATCH = Path(max(_cands, key=_free_gb))
if _free_gb(SCRATCH) < 1.1 * _need + 5:
    sys.exit(f"the cache needs ~{_need:.1f} GB of scratch disk and the best candidate, "
             f"{SCRATCH}, has {_free_gb(SCRATCH):.1f} GB -- refusing before any work")
print(f"scratch: {SCRATCH}  (cache ~{_need:.1f} GB)  smoke={SMOKE}  built from {BUILT_FROM}")

REPO = SCRATCH / "st_repo"           # outside /kaggle/working: it holds competition CSVs
CACHE_DIR = SCRATCH / "st_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_NPY = str(CACHE_DIR / f"tiles_{TAG}.npy")
OUT = Path("/kaggle/working")
RUN = OUT / "runs_forksample"

FILES = json.loads(base64.b64decode(__FILES_B64__).decode())
for _rel, (_b64, _sha) in FILES.items():
    _p = REPO / _rel
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_bytes(base64.b64decode(_b64))
    if hashlib.sha256(_p.read_bytes()).hexdigest() != _sha:
        sys.exit(f"{_rel}: embedded file does not match its hash")
print(f"wrote {len(FILES)} repo files to {REPO}, all hashes match", flush=True)
'''


PREP = r'''
# ============================================================================ #
# Sampling test -- labels, site folds, and the partition check. All of it runs
# BEFORE the decode: a wrong partition found after four hours is four hours lost.
# ============================================================================ #
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
for _p in (REPO, REPO / "extractor", REPO / "pipeline", REPO / "fusion"):
    sys.path.insert(0, str(_p))
D = REPO / "data"
(D / "external").mkdir(parents=True, exist_ok=True)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# One pass over the inputs, never into the image trees (half a terabyte of files).
WANT = {"llm_labels_v2.csv", "labels_llm_gpt56sol.csv", "headers.parquet"}
FOUND = {}
for _root, _dirs, _files in os.walk("/kaggle/input"):
    _dirs[:] = [d for d in _dirs if d not in ("train_series", "test_series")]
    for _f in WANT.intersection(_files):
        FOUND.setdefault(_f, []).append(Path(_root) / _f)
for _f in sorted(WANT):
    if _f not in FOUND:
        sys.exit(f"{_f} not found under /kaggle/input -- is its source in kernel-metadata.json?")
    if len(FOUND[_f]) > 1:
        log(f"!! {_f} found {len(FOUND[_f])} times; using {FOUND[_f][0]}")
    log(f"input {_f}: {FOUND[_f][0]}  sha256 {_sha(FOUND[_f][0])[:16]}")

# Hold every third-party input to provenance/inputs.json: a file that differs from the one the
# project measured on stops the run HERE, before the decode. An empty record just records.
from provenance import check as _check, DIR as _PROV
_check(FOUND["llm_labels_v2.csv"][0], "stevenleehans/rsna-knee-llm-report-labels")

# targets.csv, exactly as README step 1 builds it
shutil.copy(ROOT / "train.csv", D / "train.csv")
_tr = pd.read_csv(D / "train.csv")
_v2 = (pd.read_csv(FOUND["llm_labels_v2.csv"][0])
       .drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID"))
_v2.reindex(_tr.StudyInstanceUID)[L].reset_index().to_csv(D / "targets.csv", index=False)
TARGETS_DF = pd.read_csv(D / "targets.csv").set_index("StudyInstanceUID")
log(f"targets.csv: {len(TARGETS_DF)} studies, "
    f"{int(TARGETS_DF[L].isna().any(axis=1).sum())} with a missing target")

# the scoring reference, where score_oof.py looks for it
_ref = D / "public_llm_labels" / "lixin73_rsna-knee-llm-report-labels-sol56"
_ref.mkdir(parents=True, exist_ok=True)
shutil.copy(FOUND["labels_llm_gpt56sol.csv"][0], _ref / "labels_llm_gpt56sol.csv")
_check(_ref / "labels_llm_gpt56sol.csv", "lixin73/rsna-knee-llm-report-labels-sol56")

# site fingerprints, then site-grouped folds -- the repo's own code, README step 2-3
shutil.copy(FOUND["headers.parquet"][0], D / "external" / "dicom_headers_zhukovoleksiy.parquet")
_check(D / "external" / "dicom_headers_zhukovoleksiy.parquet",
       "kaggle kernels output zhukovoleksiy/rsna-metadata-probe")
import site_fingerprint as _sfp
_fp = _sfp.build()
_fp.to_csv(_sfp.OUT, index=False)
_nfp = _fp.fingerprint.nunique()
log(f"site fingerprints: {_nfp} (README expects ~265; 200-340 is healthy)")
if not 200 <= _nfp <= 340:
    log("!! fingerprint count outside 200-340 -- the partition check below will say whether it matters")

import folds as _folds
FOLDS = _folds.build(group_by="site")
FOLDS.to_csv(D / "folds_site.csv", index=False)
if FOLDS.groupby("group").fold.nunique().max() > 1:
    sys.exit("a site group landed in two folds -- grouping is broken")

# THE PARTITION CHECK. The control's 691 studies must all sit in the rebuilt fold 0, or some of
# them are now in this run's TRAINING set and the paired delta is contaminated. folds_site.csv
# was never committed and its inputs are unpinned downloads (AUDIT.md), so this is not a formality.
CONTROL = Path(os.environ.get("SAMPLING_TEST_CONTROL",      # overridable ONLY for an off-Kaggle smoke
                              str(REPO / "fusion" / "runs_port_repro" / "oof_all.csv")))
CTRL = pd.read_csv(CONTROL).StudyInstanceUID.drop_duplicates().tolist()
_f0 = set(FOLDS.loc[FOLDS.fold == 0, "StudyInstanceUID"])
_drift = [u for u in CTRL if u not in _f0]
log(f"partition check: {len(CTRL) - len(_drift)}/{len(CTRL)} control studies are in the rebuilt "
    f"fold 0 ({len(_f0)} studies)")
if _drift:
    sys.exit(f"⛔ the rebuilt folds_site.csv does not reproduce the control's fold 0: {len(_drift)} "
             f"of its studies moved. An upstream input changed (targets or headers). Stop -- the "
             f"paired delta would be contaminated. First moved: {_drift[:3]}")
# The partition passed: keep it, so the next rebuild can be held to it (AUDIT.md). The record and
# the fold file land in /kaggle/working/provenance/ -- commit both to the repo's provenance/.
_check(D / "folds_site.csv", "fusion/folds.py --group-by site", keep_copy=True)
shutil.copytree(_PROV, OUT / "provenance", dirs_exist_ok=True)
PROVENANCE = {"built_from": BUILT_FROM,
              "inputs": {k: {"path": str(v[0]), "sha256": _sha(v[0])} for k, v in FOUND.items()},
              "folds_site_sha256": _sha(D / "folds_site.csv"),
              "targets_sha256": _sha(D / "targets.csv"),
              "n_fingerprints": int(_nfp)}
(OUT / "sampling_test_provenance.json").write_text(json.dumps(PROVENANCE, indent=2))
'''


BUILD = r'''
# ============================================================================ #
# Sampling test -- the fork's pixel contract, from its members' own manifest,
# then the fork's own header pass, slot choice and decode.
# ============================================================================ #
TIME_BUDGET = 10.5 * 3600    # the kernel dies at 12 h; training needs the rest
ORDER_BUDGET_S = 4 * 3600    # a ceiling, not a target -- see the check after build_cache

_pkg = find_weights()
if _pkg is None:
    sys.exit("pilkwang/rsna-knee-weights is not attached: its manifest IS the pixel contract")
_man = json.loads((_pkg / "manifest.json").read_text())
_cfgs = {json.dumps(m["config"], sort_keys=True) for m in _man["members"]}
if len(_cfgs) != 1:
    sys.exit(f"the members carry {len(_cfgs)} distinct configs; this test assumes one (§2y)")
MEMBER_CFG = json.loads(_cfgs.pop())
adopt_config_globals(MEMBER_CFG)
_expect = {"IMG": 336, "GROUP": 3, "CACHE_SLICES": 12, "CROP_MM": 130.0, "SLICE_BAND": (0.2, 0.8)}
_got = {"IMG": IMG, "GROUP": GROUP, "CACHE_SLICES": CACHE_SLICES, "CROP_MM": float(CROP_MM),
        "SLICE_BAND": tuple(float(x) for x in SLICE_BAND)}
if _got != _expect:
    sys.exit(f"the members' manifest now says {_got}, not the {_expect} recorded in "
             f"pilkwang_pixels.py:66-70 -- a different contract is a different test")
log(f"pixel contract = the members' own: {_got}, rules {RULES}, N_GROUP {N_GROUP}")

_ts = pd.read_csv(ROOT / "train_series.csv")
plane_map = dict(zip(_ts["SeriesInstanceUID"], _ts["Anatomical_Plane"]))
log("header pass: train")
htr = annotate(walk("train_series"))
slots_tr = pick_slots(htr, plane_map)
_usable = {s for s, v in slots_tr.items() if len(v)}
log(f"  {len(htr)} series, {len(_usable)} studies with at least one slot")

# Validation = the control's studies. Training = a seeded draw of the control's COUNT from the
# same pool (fold != 0, not gold, a complete target). The control trained on the studies its NIfTI
# mirror covered, a list that was deleted with data/; matching the count keeps "more data" out of
# the delta, and the draw is recorded below.
VAL = [u for u in CTRL if u in _usable]
_pool = sorted(u for u in FOLDS.loc[(FOLDS.fold != 0) & (~FOLDS.is_gold), "StudyInstanceUID"]
               if u in _usable and u in TARGETS_DF.index and TARGETS_DF.loc[u, L].notna().all())
TRAIN = sorted(np.random.default_rng(20260923).choice(
    _pool, size=min(N_TRAIN, len(_pool)), replace=False).tolist())
if SMOKE:
    TRAIN, VAL = TRAIN[:SMOKE_TRAIN], VAL[:SMOKE_VAL]
log(f"train {len(TRAIN)} (of a pool of {len(_pool)}) / val {len(VAL)} "
    f"({len(CTRL) - len([u for u in CTRL if u in _usable])} control studies have no slot)")

_chosen = sorted(set(TRAIN) | set(VAL))
slot_map = {s: slots_tr[s] for s in _chosen}
_t = time.time()
studies, _C, _M = build_cache(slot_map, plane_map, lat_of(htr, "train "), "train")
_recs = [rec for s in _chosen for rec in slot_map[s].values()]
_unordered = sum("ordered" not in rec for rec in _recs)
if _unordered:
    sys.exit(f"slice ordering hit its budget with {_unordered}/{len(_recs)} series unordered; "
             f"those keep file order, which is uncorrelated with anatomy -- refusing to train")
if time.time() - T0 > TIME_BUDGET:
    sys.exit("decode hit the time budget, so the cache is partial -- refusing to train")
log(f"cache built in {(time.time() - _t) / 60:.1f} min: {int(_M.sum())}/{len(_recs)} slots filled")

# train_port's TileStore format: tiles_<tag>.npy (this memmap), index_<tag>.csv, manifest_<tag>.json
_C.flush()
_idx = pd.DataFrame({"StudyInstanceUID": studies, "row": np.arange(len(studies))})
for _j, _s in enumerate(SLOTS):
    _idx[f"has_{_s[0]}"] = _M[:, _j] > 0
_idx.to_csv(CACHE_DIR / f"index_{TAG}.csv", index=False)
_cm = {"spec": "fork read_slot, 12 slices over 20-80%, groups of 3",
       "slots": [{"name": s[0], "plane": s[1]} for s in SLOTS],
       "preprocess_version": "fork-read_slot-" + hashlib.sha256(
           json.dumps(MEMBER_CFG, sort_keys=True).encode()).hexdigest()[:12],
       # the fork orders slices by geometry, so there is no K16 bit to carry; non-zero keeps
       # train_port from printing its "no direction bit" note, which is about tile caches
       "direction_bits": "geometry (fork order_slices)",
       "member_config": MEMBER_CFG, "rules": RULES,
       "n_train": len(TRAIN), "n_val": len(VAL), "train_draw_seed": 20260923}
(CACHE_DIR / f"manifest_{TAG}.json").write_text(json.dumps(_cm, indent=2, default=str))
for _f in (f"index_{TAG}.csv", f"manifest_{TAG}.json"):
    shutil.copy(CACHE_DIR / _f, OUT / _f)
pd.Series(TRAIN, name="StudyInstanceUID").to_csv(OUT / "train_studies.csv", index=False)
del _C
gc.collect()
'''


TRAIN = r'''
# ============================================================================ #
# Sampling test -- train_port.py, fold 0, every default as in runs_port_repro.
# ============================================================================ #
_cmd = [sys.executable, "-u", str(REPO / "fusion" / "train_port.py"),
        "--cache", str(CACHE_DIR), "--tag", TAG,
        "--folds", str(D / "folds_site.csv"), "--labels", str(D / "targets.csv"),
        "--run-folds", "0", "--out", str(RUN), "--verbose",
        # runs_port_repro trained with both workers drawing identical augmentation streams (fixed
        # in train_port 09-23). Keep that here so the input stays the ONE variable.
        "--legacy-worker-rng"]
if SMOKE:
    _cmd += ["--epochs", "1"]
log("train: " + " ".join(_cmd))
_rc = subprocess.call(_cmd)
if _rc:
    sys.exit(f"train_port exited {_rc}")
'''


SCORE = r'''
# ============================================================================ #
# Sampling test -- the paired read, and the rule fixed before the run.
# ============================================================================ #
GAP, MAIN, NOT_MAIN = 0.0856, 0.043, 0.015
VERDICT = {"smoke": SMOKE, "rule": {"gap": GAP, "input_is_main_cause_if_delta_ge": MAIN,
                                    "input_is_not_main_cause_if_delta_le": NOT_MAIN},
           "provenance": PROVENANCE}
if SMOKE:
    log("SMOKE: the whole path ran. No score -- 16 studies cannot carry one. Push without --smoke.")
else:
    _res = subprocess.run([sys.executable, str(REPO / "fusion" / "score_oof.py"),
                           str(REPO / "fusion" / "runs_port_repro"), str(RUN)],
                          capture_output=True, text=True)
    print(_res.stdout)
    print(_res.stderr)
    (OUT / "score_oof.txt").write_text(_res.stdout + _res.stderr)
    _m = re.search(r"PAIRED delta \(.*?\) = ([+-]\d+\.\d+) \+-(\d+\.\d+)", _res.stdout)
    if not _m:
        sys.exit("could not read the paired delta from score_oof.py -- see score_oof.txt")
    _d, _sd = float(_m.group(1)), float(_m.group(2))
    if _d >= MAIN:
        _call = "THE INPUT IS THE MAIN CAUSE"
    elif _d <= NOT_MAIN:
        _call = "THE INPUT IS NOT THE MAIN CAUSE"
    else:
        _call = "BOTH CONTRIBUTE"
    VERDICT.update({"delta": _d, "sd": _sd, "share_of_gap": _d / GAP, "call": _call})
    log(f"VERDICT  delta {_d:+.4f} +-{_sd:.4f} = {_d / GAP:.0%} of the {GAP} gap  ->  {_call}")
(OUT / "sampling_test_verdict.json").write_text(json.dumps(VERDICT, indent=2))
log("sampling test complete")
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


def fork_cells(nb: dict) -> list[str]:
    """The fork's cells this run needs, verbatim except for the one anchored cache patch."""
    code = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    picked = []
    for a in FORK_ANCHORS:
        hits = [i for i, s in code if a in s]
        if len(hits) != 1:
            raise SystemExit(f"fork anchor {a!r} found in {len(hits)} cells, expected exactly 1 -- "
                             f"the submission notebook changed shape; refusing to build")
        picked.append(hits[0])
    src = dict(code)
    out = []
    for i in sorted(set(picked)):
        s = src[i]
        if "def build_cache(" in s:
            if s.count(CACHE_OLD) != 1:
                raise SystemExit("build_cache's allocation line is not there exactly once; refusing")
            s = s.replace(CACHE_OLD, CACHE_NEW)
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="CPU, 64 train + 16 val studies, one epoch. ALWAYS push this first.")
    a = ap.parse_args()

    files = {}
    for rel in EMBED:
        b = (PROJ / rel).read_bytes()
        files[rel] = [base64.b64encode(b).decode(), hashlib.sha256(b).hexdigest()]
    blob = base64.b64encode(json.dumps(files).encode()).decode()
    built_from = _git_head()

    setup = (SETUP.replace("__SMOKE__", repr(bool(a.smoke)))
                  .replace("__BUILT_FROM__", repr(built_from))
                  .replace("__FILES_B64__", repr(blob)))
    cells = [setup] + fork_cells(json.loads(SRC_NB.read_text(encoding="utf-8"))) \
        + [PREP, BUILD, TRAIN, SCORE]
    for k, c in enumerate(cells):
        compile(c, f"cell{k}", "exec")              # fail here, not after the queue wait

    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": c.strip("\n").splitlines(keepends=True)} for c in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

    src_meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta = {"id": f"{OWNER}/{SLUG}", "title": TITLE, "code_file": f"{SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": not a.smoke, "enable_tpu": False,
            # internet: train_port pulls facebook/dinov2-small from the Hub, as runs_port_repro did.
            # This is a training run, not a submission, so it is allowed to.
            "enable_internet": True, "keywords": [],
            "dataset_sources": DATASETS, "kernel_sources": KERNELS,
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
            # the image the fork's cells already ran under, in the F6 submission
            "docker_image": src_meta["docker_image"]}
    if not a.smoke:
        meta["machine_shape"] = "NvidiaTeslaT4"     # F6 drew a P100 once without it (§3d)
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"built {a.out / (SLUG + '.ipynb')}  ({len(cells)} cells, "
          f"{len(cells) - 5} lifted from the fork)  smoke={a.smoke}  from {built_from}")
    print(f"  embedded: {', '.join(EMBED)}")
    print(f"  sources:  {', '.join(DATASETS + KERNELS)}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")
    print(f"push:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
