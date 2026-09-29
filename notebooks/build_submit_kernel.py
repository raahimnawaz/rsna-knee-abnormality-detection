"""Submission B: our Stage A OrthoFoundation-L model rank-meaned into the public 0.943 stack.

    python -m kaggle kernels pull mattiaangeli/bend-the-knee-to-speedy-raptors-the-original -p <pub> -m
    python notebooks/build_submit_kernel.py --public-dir <pub> --out /tmp/subb
    python -m kaggle kernels push -p /tmp/subb          # commit run on the 3 visible test studies
    # then, only if the commit log shows CHECK PASSED and the ARMS line:
    python -m kaggle competitions submit rsna-knee-abnormality-detection -k raahimnawaz/<slug> -v <n> \
        -f submission.csv -m "..."
    python notebooks/build_submit_kernel.py --ours-only --n-windows 18 --out /tmp/suba   # Submission A

SUBMISSION A (`--ours-only`) is step 1 alone plus a cell that writes our predictions as submission.csv:
the model's own leaderboard number, and an efficiency-track entry. There the gate runs only in the
commit run (test.csv under 100 studies) -- the scored rerun is the same code in the same image, and
the gate's minute is runtime the efficiency score charges for.

WHAT IT RUNS, IN ORDER (Submission B).
  1. OURS. `fusion/infer_windows.py` in subprocesses, so the notebook process never imports torch
     before the public cells set their thread environment, and the GPUs are empty when they start.
     First the GATE: 8 of the 691 fold-0 studies are rebuilt from their train DICOMs and must
     reproduce `runs_ortho/oof_all.csv` (the predictions that scored 0.8298). A failure stops the
     notebook, so no submission.csv exists and nothing can be submitted by mistake. Then every
     test study, one process per T4.
  2. THE PUBLIC STACK, cells 0-22 of `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original`
     (Apache 2.0) unmodified. Its last cell leaves its final ranks in `_pipeline_stage.csv`.
  3. THE BLEND. Our model joins as a third FAMILY. The public stack is two families by its own
     accounting (transformer and CoAtNet, `family_mix=[.5, .5]`, outer weights tuned by its author);
     §5.4's rule is an equal rank-mean over families with nothing fitted, so ours is 1/3 and the
     public output, internal weights untouched, is 2/3. `--w-ours` exists to record a different
     choice, not to tune one: there is no instrument here to tune it on.
  4. THE PUBLIC PUBLISH CELL, unmodified: its validation, then submission.csv.

ORDER AND TIME. Ours runs first so a fault surfaces in minutes, not after the public stack's hours.
The public stack describes itself as "comfortably in less than 9 hrs" on the hidden 1,322 and
guards 8 h from its own start; ours is ~15 min of decode overlapped with ~17 min of T4 per GPU.
The commit run cannot price the hidden run (3 studies), so the margin is the one real risk here.

THE UPGRADED STACK (`--core-kernel K --core-sha S`, RUNBOOK "Pending reads" (A)): our CoAtNet student
REPLACES the stack's core instead of joining as a family.

    python notebooks/build_submit_kernel.py --public-dir <pub> --core-kernel raahimnawaz/<student kernel> \
        --core-sha <sha256 of its runs_coatnet/last.pt> --tag student-r1 --out /tmp/subcore

The core is `maxspan-v5` (w 0.60) and `maxspan-v5-reverse` (w 0.10, the slice-reversed pass). Both load
ONE file, `raptor_ft_coatnet_v5_full_swa.pt`, and `run_shared_maxspan` requires that they share it. The
student is that model, started from that file (`train_windows --init`, CC0) and fine-tuned on teacher
labels, so it drops into the slot at zero extra runtime. Three changes to what runs, nothing else:
  1. a cell after the public cell that defines `ASSET_ROOTS`, CPU only. It finds our `last.pt` by
     sha256, checks `last_meta.json` against the slot's preprocessing (CoAtNet, Stage A, 64 slices over
     2-98%, 336 px, no slot embedding, no zoom), and renames the keys back to the public names
     (`encoder.` -> `backbone.`, `w` -> `clsW`, `b` -> `clsb`). The key set must equal the public file's
     exactly. Each tensor is cast to the public file's dtype. It is written under a NEW name, and its
     folder is added to `ASSET_ROOTS`;
  2. in the public ARMS, those two entries' `file` is that new name. A new name, so a missing file
     fails the lookup (`_asset_find_asset` wants exactly one hit) instead of loading the public
     weights;
  3. a cell before the publish cell. The stack logs `raptor_checkpoint` (file, sha256) for every load.
     The cell refuses the run, so no submission.csv exists, unless the core loaded our file at its
     sha256 and the public v5 file was never loaded. Then it prints the ARMS line.
Every other member, weight, calibrator and the publish cell run as published. `--add-family` also adds
a `--weights-kernel` model as a family at 1/3, as Submission B does (RUNBOOK rule (C): the Ortho
student joins only at >= +0.004 over (A)). Without it, nothing of ours runs outside the core slot.
Tested off Kaggle by `notebooks/core_swap_fixture.py`.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
OWNER = "raahimnawaz"
SLUG = "rsna-knee-sub-b-ortho-blend"
PUBLIC = "mattiaangeli/bend-the-knee-to-speedy-raptors-the-original"
WEIGHTS_KERNEL = f"{OWNER}/rsna-knee-c6-train-ortho"
WEIGHTS_SHA256 = "1fe51be2ff4deaae4dfd5fafd2354f3b9bbeb0c26cda3fd918a00699fd481b98"   # runs_ortho/last.pt
# the public stack's pinned image (torch 2.10.0+cu128); Submission A uses it too, so A and B run one environment
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
EMBED = ["fusion/infer_windows.py", "fusion/train_windows.py", "fusion/window_curve.py",
         "pipeline/build_slots64.py", "pipeline/slots64.py", "pipeline/slots96.py",
         "fusion/orthofoundation_model.py", "extractor/metrics.py",
         "fusion/qwen_vlm.py"]   # build_slots64 imports slots96; train_windows imports qwen_vlm (qwen archs only)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


INTRO = """# RSNA Knee -- Submission B: OrthoFoundation-L (Stage A) + the public 0.943 stack

Our model -- OrthoFoundation-L fine-tuned on a 64-slice, 2-98% stack built from the DICOMs
(0.8298 on 691 held-out studies) -- joins the public stack below as a third family, equal
rank-mean over families, nothing fitted. Built by `notebooks/build_submit_kernel.py`
(__BUILT_FROM__).

The public stack is `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` (Apache 2.0),
run unmodified; its own description and credits follow."""


OURS = r'''
# ============================================================================ #
# Submission B, part 1 -- our model. Standard library only in this process;
# torch runs in subprocesses so the public cells start clean (see the builder).
# ============================================================================ #
import base64 as _ob64, hashlib as _ohash, json as _ojson, os as _oos, subprocess as _osp
import sys as _osys, time as _otime, csv as _ocsv, shutil as _oshutil
from pathlib import Path as _OPath

_O_T0 = _otime.time()
_O = _OPath("/tmp/ortho_sub")
_O_REPO = _O / "repo"
_O_W = __W_OURS__
_O_N_CHECK = __N_CHECK__
_O_N_WINDOWS = __N_WINDOWS__          # 0 = every window, as the OOF was scored
_O_CHECK_ALWAYS = __CHECK_ALWAYS__    # False: gate in the commit run only (efficiency entries)


def _olog(m):
    print(f"[ortho {_otime.time() - _O_T0:7.1f}s] {m}", flush=True)


for _rel, (_b, _sha) in _ojson.loads(_ob64.b64decode(__FILES_B64__).decode()).items():
    _p = _O_REPO / _rel
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_bytes(_ob64.b64decode(_b))
    if _ohash.sha256(_p.read_bytes()).hexdigest() != _sha:
        raise RuntimeError(f"{_rel}: embedded file does not match its hash")
_olog(f"built from __BUILT_FROM__; repo files written, hashes match")

_SKIP = {"train_series", "test_series", "train_images", "test_images"}
_O_ROOT, _O_CKPT = None, None
for _d, _dirs, _fs in _oos.walk("/kaggle/input"):
    _dirs[:] = [x for x in _dirs if x not in _SKIP]
    if _O_ROOT is None and {"test.csv", "test_series.csv", "train_series.csv"} <= set(_fs):
        _O_ROOT = _OPath(_d)
    # any train_windows run dir (runs_ortho, runs_coatnet, ...): the sha256 is what identifies the checkpoint
    if "last.pt" in _fs and "oof_all.csv" in _fs and _OPath(_d).name.startswith("runs_"):
        _c = _OPath(_d) / "last.pt"
        if _ohash.sha256(_c.read_bytes()).hexdigest() == "__WEIGHTS_SHA256__":
            _O_CKPT = _c
if _O_ROOT is None:
    raise RuntimeError("competition root not found under /kaggle/input")
if _O_CKPT is None:
    raise RuntimeError("runs_*/last.pt with the recorded sha256 not found -- is __WEIGHTS_KERNEL__ attached?")
# The arm's name for the ARMS line, from how it was trained. Step 5's run predates last_meta.json: Ortho, Stage A.
_O_ARM = "ortho_stage_a"
if (_O_CKPT.parent / "last_meta.json").exists():
    _om = _ojson.loads((_O_CKPT.parent / "last_meta.json").read_text())
    _O_ARM = f"{_om.get('arch', 'ortho')}_{str(_om.get('layout', 'stage_a')).split()[0]}"
_olog(f"competition root {_O_ROOT} | weights {_O_CKPT} (sha256 ok) | arm {_O_ARM}")

try:
    _O_NGPU = len([l for l in _osp.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                       check=True).stdout.splitlines() if l.startswith("GPU ")])
except Exception as _e:                                    # noqa: BLE001
    _olog(f"nvidia-smi failed ({_e}); assuming one GPU")
    _O_NGPU = 1
_olog(f"GPUs: {_O_NGPU}")
_O_PY = [_osys.executable, "-u", str(_O_REPO / "fusion" / "infer_windows.py"), "--root", str(_O_ROOT),
         "--weights", str(_O_CKPT)]


def _orun(args, gpu, tag):
    env = dict(_oos.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    logf = open(_O / f"{tag}.log", "w")
    return _osp.Popen(_O_PY + args, env=env, stdout=logf, stderr=_osp.STDOUT), logf


def _owait(procs, what):
    bad = []
    for tag, (p, logf) in procs.items():
        rc = p.wait()
        logf.close()
        lines = (_O / f"{tag}.log").read_text().splitlines()
        print("\n".join(lines[-(60 if rc else 8):]), flush=True)
        if rc:
            bad.append(f"{tag} exited {rc}")
    if bad:
        raise RuntimeError(f"{what} failed: {bad} -- stopping before any submission.csv exists")


# ---- 1. the gate: studies the OOF already scored, rebuilt here from DICOMs
_O.mkdir(parents=True, exist_ok=True)
_oof = _O_CKPT.parent / "oof_all.csv"
with open(_oof) as _f:
    _val = sorted({r["StudyInstanceUID"] for r in _ocsv.DictReader(_f)})
with open(_O_ROOT / "test.csv") as _f:
    _test_ids = [r["StudyInstanceUID"] for r in _ocsv.DictReader(_f)]
if (_O_ROOT / "train_series").is_dir() and (_O_CHECK_ALWAYS or len(_test_ids) < 100):
    with open(_O / "check_uids.csv", "w") as _f:
        _f.write("StudyInstanceUID\n" + "\n".join(_val[::max(len(_val) // _O_N_CHECK, 1)][:_O_N_CHECK]) + "\n")
    _owait({"check": _orun(["--split", "train", "--uids", str(_O / "check_uids.csv"), "--out",
                            str(_O / "check" / "check.csv"), "--check-against", str(_oof),
                            "--workers", "4"], 0, "check")}, "the OOF reproduction check")
    _olog("GATE PASSED: " + (_O / "check" / "check.check.json").read_text().split('"per_study_max"')[0])
elif not (_O_ROOT / "train_series").is_dir():
    _olog("train DICOMs are not mounted in this run; the gate ran in the commit run on this code")
else:
    _olog(f"{len(_test_ids)} test studies: the scored rerun -- the commit run carried the gate on this code")

# ---- 2. every test study, one process per GPU
_procs = {f"test{i}": _orun(["--split", "test", "--shard", str(i), "--n-shards", str(_O_NGPU),
                             "--workers", "2" if _O_NGPU > 1 else "4",
                             "--out", str(_O / f"preds_{i}.csv")]
                            + (["--n-windows", str(_O_N_WINDOWS)] if _O_N_WINDOWS else []), i, f"test{i}")
          for i in range(_O_NGPU)}
_owait(_procs, "test inference")

_rows = {}
for _i in range(_O_NGPU):
    with open(_O / f"preds_{_i}.csv") as _f:
        _rd = _ocsv.reader(_f)
        _hdr = next(_rd)
        for _r in _rd:
            if _r[0] in _rows:
                raise RuntimeError(f"study {_r[0]} predicted twice")
            _rows[_r[0]] = _r
if set(_rows) != set(_test_ids):
    raise RuntimeError(f"ortho predictions cover {len(_rows)} studies, test.csv has {len(_test_ids)}")
with open(_O / "ortho_preds.csv", "w", newline="") as _f:
    _w = _ocsv.writer(_f)
    _w.writerow(_hdr)
    for _u in _test_ids:
        _w.writerow(_rows[_u])
_O_MIN = (_otime.time() - _O_T0) / 60
_olog(f"OURS DONE: {len(_rows)} test studies in {_O_MIN:.1f} min on {_O_NGPU} GPU(s), "
      f"windows {_O_N_WINDOWS or 'all'} -> {_O / 'ortho_preds.csv'}")
'''


BLEND = r'''
# ============================================================================ #
# Submission B, part 3 -- our model joins the public stack as a THIRD FAMILY.
# Equal rank-mean over families, nothing fitted (RUNBOOK §5.4): ours 1/3, the
# public output (its own two families, internal weights untouched) 2/3.
# ============================================================================ #
import numpy as _bnp
import pandas as _bpd
from pathlib import Path as _BPath
import json as _bjson

_B_LAB = __LABELS__
_b_stage = _BPath("/kaggle/working/_pipeline_stage.csv")
_b_pub = _bpd.read_csv(_b_stage, dtype={"StudyInstanceUID": str})
if _b_pub.columns.tolist() != ["StudyInstanceUID"] + _B_LAB:
    raise RuntimeError(f"public stage schema drift: {_b_pub.columns.tolist()}")
_b_ours = (_bpd.read_csv("/tmp/ortho_sub/ortho_preds.csv", dtype={"StudyInstanceUID": str})
           .drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
           .reindex(_b_pub.StudyInstanceUID))
if _b_ours[_B_LAB].isna().any().any():
    raise RuntimeError("ortho predictions do not cover every study in the public stage")
_b_rp = _b_pub[_B_LAB].rank(method="average", pct=True).to_numpy(_bnp.float64)
_b_ro = _b_ours[_B_LAB].reset_index(drop=True).rank(method="average", pct=True).to_numpy(_bnp.float64)
_b_mix = (1.0 - _O_W) * _b_rp + _O_W * _b_ro
if not _bnp.isfinite(_b_mix).all():
    raise RuntimeError("blend is not finite")
_b_rho = {l: float(_bpd.Series(_b_rp[:, j]).corr(_bpd.Series(_b_ro[:, j]), method="spearman"))
          for j, l in enumerate(_B_LAB)}
_b_mrho = float(_bnp.nanmean(list(_b_rho.values())))
print(f"[blend] per-label Spearman(ours, public) mean {_b_mrho:.3f}: "
      + ", ".join(f"{l} {r:.2f}" for l, r in _b_rho.items()), flush=True)
if len(_b_pub) >= 100 and not _b_mrho > 0.2:
    raise RuntimeError(f"ours and the public stack barely agree (mean rho {_b_mrho:.3f}) -- a misalignment, not a model")
_b_out = _b_pub.copy()
_b_out[_B_LAB] = _b_mix
_b_out.to_csv(_b_stage, index=False)
_bpd.read_csv("/tmp/ortho_sub/ortho_preds.csv").to_csv("/kaggle/working/ortho_preds.csv", index=False)
_BPath("/kaggle/working/sub_b_receipt.json").write_text(_bjson.dumps({
    "rule": "equal rank-mean over families, nothing fitted", "w_ours": _O_W,
    "families": ["public: transformer", "public: CoAtNet", f"ours: {_O_ARM}"],
    "n_studies": len(_b_pub), "spearman_ours_vs_public": _b_rho, "ours_minutes": _O_MIN,
    "built_from": "__BUILT_FROM__"}, indent=1))
print(f"[blend] ARMS: public stack (2 families, as published) + {_O_ARM} = 3 families; "
      f"w_ours {_O_W:.4f}; {len(_b_pub)} studies -> {_b_stage}", flush=True)
'''


SUB_A = r'''
# ============================================================================ #
# Submission A -- our model alone. Its predictions ARE the submission.
# ============================================================================ #
from pathlib import Path as _APath
_a_src = _APath("/tmp/ortho_sub/ortho_preds.csv")
_a_hdr = _a_src.read_text().splitlines()[0].split(",")
if _a_hdr[0] != "StudyInstanceUID" or len(_a_hdr) != 13:
    raise RuntimeError(f"ortho predictions header {_a_hdr}")
_APath("/kaggle/working/submission.csv").write_text(_a_src.read_text())
print(f"[sub A] ARMS: {_O_ARM} only, windows {_O_N_WINDOWS or 'all'}; {len(_test_ids)} studies; "
      f"ours {_O_MIN:.1f} min of the run", flush=True)
'''

INTRO_A = """# RSNA Knee -- Submission A: OrthoFoundation-L (Stage A), alone

Our model alone -- OrthoFoundation-L fine-tuned on a 64-slice, 2-98% stack built from the DICOMs
(0.8298 on 691 held-out studies) -- at __WINDOWS__ windows per study, on both T4s. Its first
leaderboard number, and an efficiency-track entry. Built by `notebooks/build_submit_kernel.py
--ours-only` (__BUILT_FROM__)."""


CORE_PUBLIC = "raptor_ft_coatnet_v5_full_swa.pt"     # the public core's file: maxspan-v5 and its reverse
CORE_ARMS = ("maxspan-v5", "maxspan-v5-reverse")

CORE = r'''
# ============================================================================ #
# Upgraded stack, part 1 -- our CoAtNet student becomes the public stack's CORE
# (maxspan-v5 + maxspan-v5-reverse, one shared file). CPU only. Runs after the
# public cell that defines ASSET_ROOTS, before the stack starts its clock.
# ============================================================================ #
import gc as _cgc, hashlib as _chash, json as _cjson, os as _cos
from pathlib import Path as _CPath
import torch as _ctorch

_C_KERNEL = "__CORE_KERNEL__"
_C_SHA = "__CORE_SHA256__"          # the student run's runs_*/last.pt
_C_FILE = "__CORE_FILE__"           # a NEW name: a miss fails the stack's lookup, it never loads the public weights
_C_PUBLIC = "__CORE_PUBLIC__"
_C_INPUT = _CPath("/kaggle/input")
_C_DIR = _CPath("/tmp/student_core")
# the maxspan-v5 slot's model and preprocessing, as our Stage A cache + eval reproduced it (gold 0.9187 vs 0.9214)
_C_META = {"arch": "coatnet", "timm": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "pool": "avg",
           "layout": "stage_a span [0.02, 0.98] img 336 canon False", "img": 336,
           "slot_sizes": [18, 14, 12, 8, 12], "pos_emb": False, "n_slots": 0}


def _csha(p):
    h = _chash.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


_c_src = None
for _d, _dirs, _fs in _cos.walk(_C_INPUT):
    _dirs[:] = [x for x in _dirs if x not in {"train_series", "test_series", "train_images", "test_images"}]
    if "last.pt" in _fs and _CPath(_d).name.startswith("runs_") and _csha(_CPath(_d) / "last.pt") == _C_SHA:
        _c_src = _CPath(_d) / "last.pt"
if _c_src is None:
    raise RuntimeError(f"runs_*/last.pt with sha256 {_C_SHA[:16]} not found -- is {_C_KERNEL} attached?")
_c_meta = _cjson.loads((_c_src.parent / "last_meta.json").read_text())
_c_bad = {k: (_c_meta.get(k), v) for k, v in _C_META.items() if _c_meta.get(k) != v}
if _c_meta.get("zoom", 0.0):
    _c_bad["zoom"] = (_c_meta["zoom"], 0.0)
if _c_bad:
    raise RuntimeError(f"{_c_src} is not a model for the maxspan-v5 slot (got, want): {_c_bad}")
_C_ARM = f"{_c_meta['arch']}_{_c_meta['layout'].split()[0]}"

# The key set must equal the public file's EXACTLY; each tensor takes the public file's dtype
# (ours are fp16, `train_windows` saves them so; the cast up to fp32 is exact).
_c_ref = _ctorch.load(_asset_find_asset(_C_PUBLIC), map_location="cpu", weights_only=False)
_c_rsd = _c_ref["model"]
_c_ren = {"w": "clsW", "b": "clsb"}
_c_out = {("backbone." + k[len("encoder."):]) if k.startswith("encoder.") else _c_ren.get(k, k): v
          for k, v in _ctorch.load(_c_src, map_location="cpu", weights_only=True).items()}
_c_miss, _c_extra = sorted(set(_c_rsd) - set(_c_out)), sorted(set(_c_out) - set(_c_rsd))
_c_shape = sorted(k for k in set(_c_rsd) & set(_c_out) if tuple(_c_rsd[k].shape) != tuple(_c_out[k].shape))
if _c_miss or _c_extra or _c_shape:
    raise RuntimeError(f"student keys != the public core's: missing {_c_miss[:8]}, extra {_c_extra[:8]}, "
                       f"shapes {_c_shape[:8]}")
_c_out = {k: _c_out[k].to(_c_rsd[k].dtype) for k in _c_rsd}
if not all(bool(t.isfinite().all()) for t in _c_out.values() if t.is_floating_point()):
    raise RuntimeError("student checkpoint has non-finite weights")
# how far training moved it from the public start (fp16 storage alone reads ~1e-4); logged, not gated
_c_num = sum(float((_c_out[k].double() - _c_rsd[k].double()).pow(2).sum()) for k in _c_rsd if _c_rsd[k].is_floating_point())
_c_den = sum(float(_c_rsd[k].double().pow(2).sum()) for k in _c_rsd if _c_rsd[k].is_floating_point())
_C_MOVED = (_c_num / _c_den) ** 0.5

_C_DIR.mkdir(parents=True, exist_ok=True)
_c_path = _C_DIR / _C_FILE
_ctorch.save({"model": _c_out, "arch": _c_ref["arch"], "res": _c_ref["res"],
              "source_kernel": _C_KERNEL, "source_sha256": _C_SHA}, _c_path)
_C_OUT_SHA = _csha(_c_path)
ASSET_ROOTS.append(str(_C_DIR))
# Public cell 1 catalogues ASSET_ROOTS ONCE (`_speed_asset_catalogue`) and never walks it again, so a
# folder added after the first lookup is invisible -- and our lookup of the public file above made it.
# Drop the catalogue; the next lookup walks every root, ours included.
with _speed_asset_lock:
    _speed_asset_catalogue = None
if _CPath(_asset_find_asset(_C_FILE)).resolve() != _c_path.resolve():
    raise RuntimeError("the stack's asset lookup does not resolve our core file")
_c_n = len(_c_out)
del _c_ref, _c_rsd, _c_out
_cgc.collect()
print(f"[core] {_C_ARM} student {_c_src} (sha256 {_C_SHA[:16]} ok) -> {_c_path} (sha256 {_C_OUT_SHA[:16]}): "
      f"{_c_n} tensors, key set = the public core's, relative change vs the public weights {_C_MOVED:.4f}; "
      f"maxspan-v5 + maxspan-v5-reverse load it", flush=True)
'''

CORE_CHECK = r'''
# ============================================================================ #
# Upgraded stack, part 3 -- no submission.csv unless the core loaded OUR file.
# The stack logs raptor_checkpoint (file, sha256) at every load_model.
# ============================================================================ #
import json as _vjson
from pathlib import Path as _VPath

_v_loads = [e for e in _RSNA_AUDIT["events"] if e.get("kind") == "raptor_checkpoint"]
_v_names = [_VPath(str(e.get("file", ""))).name for e in _v_loads]
_v_ours = [e for e, n in zip(_v_loads, _v_names) if n == _C_FILE]
if _C_PUBLIC in _v_names:
    raise RuntimeError("the public maxspan-v5 weights were loaded: the core was NOT replaced -- stopping")
if not _v_ours:
    raise RuntimeError(f"no raptor_checkpoint event for {_C_FILE}: the core did not load our file -- stopping")
if any(e.get("sha256") != _C_OUT_SHA for e in _v_ours):
    raise RuntimeError(f"a file named {_C_FILE} loaded with a sha256 that is not ours -- stopping")
_v_other = sorted(set(_v_names) - {_C_FILE})
_VPath("/kaggle/working/core_receipt.json").write_text(_vjson.dumps({
    "rule": "our student replaces the public core (maxspan-v5 + maxspan-v5-reverse); everything else as published",
    "student_kernel": _C_KERNEL, "student_last_pt_sha256": _C_SHA, "core_file": _C_FILE,
    "core_file_sha256": _C_OUT_SHA, "loads": len(_v_ours), "relative_change_vs_public": _C_MOVED,
    "other_raptor_files": _v_other, "built_from": "__BUILT_FROM__"}, indent=1))
print(f"[core] ARMS: public stack as published, core maxspan-v5 + maxspan-v5-reverse (w 0.60 + 0.10) = our "
      f"{_C_ARM} student ({_C_KERNEL}, last.pt {_C_SHA[:16]}; loaded {len(_v_ours)}x, sha256 {_C_OUT_SHA[:16]}); "
      f"other Raptor files as published: {_v_other}__FAMILY__", flush=True)
'''

INTRO_CORE = """# RSNA Knee -- the upgraded stack: our CoAtNet student in the public stack's core slot

The public stack `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` (Apache 2.0) runs
as published but for ONE thing. Its core CoAtNet (`maxspan-v5`, weight 0.60, and its slice-reversed
pass, 0.10, which share one file) loads our student's weights. The student is the same model
(`dreaddevelopment/raptor-knee-maxspan`, CC0), fine-tuned on report labels corrected by an image
teacher. Every other member, weight and calibrator is untouched. A cell before the publish cell
refuses the run unless the stack's own load log shows our file at its sha256. Built by
`notebooks/build_submit_kernel.py` (__BUILT_FROM__)."""


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
        dirty = subprocess.run(["git", "-C", str(PROJ), "status", "--porcelain", "--"] + EMBED
                               + ["notebooks/build_submit_kernel.py"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--public-dir", type=Path, help=f"`kaggle kernels pull {PUBLIC} -m` (Submission B)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ours-only", action="store_true", help="Submission A: our model alone")
    ap.add_argument("--n-windows", type=int, default=0, help="thin to ~n windows per study; 0 = all (as scored)")
    ap.add_argument("--w-ours", type=float, default=1 / 3, help="the rule gives 1/3; see docstring")
    ap.add_argument("--n-check", type=int, default=8)
    # A different checkpoint: the kernel whose output holds runs_ortho/{last.pt, oof_all.csv, last_meta.json},
    # and last.pt's sha256. infer_windows reads the spec (Stage A/B, canon, 384, slot/depth) from last_meta.json,
    # and the gate checks against THAT run's oof_all.csv. Defaults: step 5's model, as submitted 09-26.
    ap.add_argument("--weights-kernel", default=WEIGHTS_KERNEL)
    ap.add_argument("--weights-sha", default=WEIGHTS_SHA256)
    ap.add_argument("--tag", default="", help="slug suffix for a submission on another checkpoint")
    ap.add_argument("--alone-lb", type=float, default=None,
                    help="another checkpoint in Submission B: its own LB number, alone -- the guard's precondition")
    ap.add_argument("--desc", default="", help="one line for the notebook's intro: what this checkpoint is")
    # The upgraded stack: a CoAtNet student REPLACES the public core (see the docstring).
    ap.add_argument("--core-kernel", default=None, help="the student's training kernel (holds runs_*/last.pt)")
    ap.add_argument("--core-sha", default=None, help="sha256 of that last.pt, all 64 hex digits")
    ap.add_argument("--add-family", action="store_true",
                    help="with --core-*: ALSO add the --weights-kernel model as a family at 1/3 (rule (C))")
    a = ap.parse_args()
    core = bool(a.core_kernel or a.core_sha)
    if core:
        if not (a.core_kernel and a.core_sha and re.fullmatch(r"[0-9a-f]{64}", a.core_sha)):
            raise SystemExit("--core-kernel and --core-sha go together; the sha is the full 64-hex sha256 of last.pt")
        if a.ours_only or not a.public_dir:
            raise SystemExit("the upgraded stack is the public stack: pass --public-dir, not --ours-only")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", a.tag or ""):
            raise SystemExit("the upgraded stack needs a --tag (lowercase, digits, '-'): it names the slug and the core file")
    elif a.add_family:
        raise SystemExit("--add-family only means something with --core-kernel (Submission B always adds a family)")
    if (a.weights_kernel != WEIGHTS_KERNEL) != (a.weights_sha != WEIGHTS_SHA256):
        raise SystemExit("--weights-kernel and --weights-sha move together: a new checkpoint needs both")
    if a.weights_kernel != WEIGHTS_KERNEL and not a.tag:
        raise SystemExit("another checkpoint needs a --tag, so its notebook does not overwrite step 5's")
    if a.weights_kernel != WEIGHTS_KERNEL and not a.ours_only and a.alone_lb is None:
        raise SystemExit("another checkpoint: build it as --ours-only first and pass its own LB as --alone-lb "
                         "before it joins any blend (RUNBOOK 09-26: blending a weaker member costs)")
    if not a.ours_only and not a.public_dir:
        raise SystemExit("Submission B needs --public-dir (or pass --ours-only for Submission A)")

    _check_import_closure(EMBED)
    files = {}
    for rel in EMBED:
        b = (PROJ / rel).read_bytes()
        files[rel] = [base64.b64encode(b).decode(), hashlib.sha256(b).hexdigest()]
    blob = base64.b64encode(json.dumps(files).encode()).decode()

    def code(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.strip("\n").splitlines(keepends=True)}

    def md(src):
        return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

    if a.ours_only:
        built_from = _git_head()
        pub = {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        pmeta = None
    else:
        pmeta = json.loads((a.public_dir / "kernel-metadata.json").read_text())
        if pmeta["id"] != PUBLIC:
            raise SystemExit(f"{a.public_dir} holds {pmeta['id']}, not {PUBLIC}")
        nb_path = a.public_dir / pmeta["code_file"]
        pub_sha = hashlib.sha256(nb_path.read_bytes()).hexdigest()
        pub = json.loads(nb_path.read_text())
        cells = pub["cells"]
        last = "".join(cells[-1]["source"])
        before = "".join(cells[-2]["source"])
        # the two seams this build relies on -- refuse an upstream version that moved them
        if not last.startswith("# Publish submission.csv only after"):
            raise SystemExit("public notebook's last cell is not its publish cell -- re-read it before building")
        if "_blend_output.to_csv(_ke_primary" not in before or "_pipeline_stage.csv" not in before:
            raise SystemExit("public notebook's cell -2 no longer writes the final stage -- re-read it")
        built_from = f"{_git_head()}, public notebook sha256 {pub_sha[:16]}"

    ours = (OURS.replace("__FILES_B64__", repr(blob)).replace("__BUILT_FROM__", built_from)
                .replace("__W_OURS__", repr(a.w_ours)).replace("__N_CHECK__", str(a.n_check))
                .replace("__N_WINDOWS__", str(a.n_windows)).replace("__CHECK_ALWAYS__", str(not a.ours_only))
                .replace("__WEIGHTS_SHA256__", a.weights_sha).replace("__WEIGHTS_KERNEL__", a.weights_kernel))
    blend = BLEND.replace("__LABELS__", repr(LABELS)).replace("__BUILT_FROM__", built_from)
    for k, c in enumerate([ours, blend, SUB_A]):
        compile(c, f"ours{k}", "exec")                  # fail here, not after the queue wait

    if a.ours_only:
        slug = f"rsna-knee-sub-a-ortho{'-' + a.tag if a.tag else ''}-n{a.n_windows or 'all'}"
        intro = INTRO_A.replace("__WINDOWS__", str(a.n_windows or "all")).replace("__BUILT_FROM__", built_from)
        if a.desc:
            intro += f"\n\n**This build:** {a.desc} Weights: `{a.weights_kernel}`, last.pt sha256 `{a.weights_sha[:16]}`."
        new = [md(intro), code(ours), code(SUB_A)]
        sources = {"dataset_sources": [], "kernel_sources": [a.weights_kernel], "model_sources": [],
                   "docker_image": DOCKER_IMAGE}
    elif core:
        core_file = f"student_core_{a.tag}_{a.core_sha[:12]}.pt"
        # the seams the swap relies on -- refuse an upstream version that moved any of them
        i_roots = [i for i, c in enumerate(cells) if c["cell_type"] == "code"
                   and re.search(r"^ASSET_ROOTS = \[", "".join(c["source"]), re.M)
                   and "def _asset_find_asset" in "".join(c["source"])]
        if len(i_roots) != 1 or i_roots[0] >= len(cells) - 2:
            raise SystemExit("public notebook: no single early cell defines ASSET_ROOTS and _asset_find_asset")
        roots_src = "".join(cells[i_roots[0]]["source"])
        if not all(re.search(p, roots_src, re.M) for p in (r"^_speed_asset_lock = ", r"^_speed_asset_catalogue = None$")):
            raise SystemExit("public cell 1's asset catalogue cache changed -- the core cell resets it; re-read it")
        if not any(re.search(r"^_RSNA_AUDIT = \{", "".join(c["source"]), re.M) for c in cells):
            raise SystemExit("public notebook no longer defines _RSNA_AUDIT -- the core check reads its load log")
        for seam in ("rsna_event('raptor_checkpoint'", "def run_shared_maxspan", "['find_weight_file'](first['file'])"):
            if seam not in before:
                raise SystemExit(f"public cell -2 lost `{seam}` -- re-read it before building")
        if before.count(CORE_PUBLIC) != 2:
            raise SystemExit(f"public cell -2 names {CORE_PUBLIC} {before.count(CORE_PUBLIC)} times, not twice "
                             f"(the two core arms) -- re-read it")
        patched = before
        for arm in CORE_ARMS:
            old = f'{{"name": "{arm}", "file": "{CORE_PUBLIC}",'
            if patched.count(old) != 1:
                raise SystemExit(f"public ARMS entry {arm} moved -- re-read cell -2 before building")
            patched = patched.replace(old, f'{{"name": "{arm}", "file": "{core_file}",')
        if patched.count(CORE_PUBLIC) or patched.count(core_file) != 2:
            raise SystemExit("the ARMS swap did not land on exactly the two core arms")
        family = (f"; + a family of ours at {a.w_ours:.4f} ([blend] ARMS line above)" if a.add_family
                  else "; nothing else of ours runs")
        core_src = (CORE.replace("__CORE_KERNEL__", a.core_kernel).replace("__CORE_SHA256__", a.core_sha)
                        .replace("__CORE_FILE__", core_file).replace("__CORE_PUBLIC__", CORE_PUBLIC))
        check_src = CORE_CHECK.replace("__BUILT_FROM__", built_from).replace("__FAMILY__", family)
        compile(core_src, "core", "exec")
        compile(check_src, "core_check", "exec")

        slug = f"rsna-knee-sub-core-{a.tag}" + ("-fam" if a.add_family else "")
        body = [dict(c, outputs=[], execution_count=None) if c["cell_type"] == "code" else c for c in cells]
        body[-2] = dict(body[-2], source=patched.splitlines(keepends=True))
        i = i_roots[0]
        intro = INTRO_CORE.replace("__BUILT_FROM__", built_from)
        intro += f"\n\n**Core:** `{a.core_kernel}`, last.pt sha256 `{a.core_sha[:16]}`, served as `{core_file}`."
        if a.desc:
            intro += f" {a.desc}"
        if a.add_family:
            intro += (f"\n\n**Plus a family of ours at {a.w_ours:.4f}:** `{a.weights_kernel}`, last.pt sha256 "
                      f"`{a.weights_sha[:16]}`, alone on the LB {a.alone_lb}.")
        new = ([md(intro)] + ([code(ours)] if a.add_family else []) + body[:i + 1] + [code(core_src)]
               + body[i + 1:-1] + ([code(blend)] if a.add_family else []) + [code(check_src), body[-1]])
        sources = {"dataset_sources": pmeta["dataset_sources"],
                   "kernel_sources": pmeta["kernel_sources"] + [a.core_kernel]
                   + ([a.weights_kernel] if a.add_family else []),
                   "model_sources": pmeta["model_sources"], "docker_image": pmeta["docker_image"]}
    else:
        slug = SLUG + (f"-{a.tag}" if a.tag else "")
        body = [dict(c, outputs=[], execution_count=None) if c["cell_type"] == "code" else c for c in cells]
        intro = INTRO.replace("__BUILT_FROM__", built_from)
        if a.desc:
            intro += (f"\n\n**This build:** {a.desc} Alone on the LB: {a.alone_lb}. Weights: `{a.weights_kernel}`, "
                      f"last.pt sha256 `{a.weights_sha[:16]}`.")
        new = [md(intro), code(ours)] + body[:-1] + [code(blend), body[-1]]
        sources = {"dataset_sources": pmeta["dataset_sources"],
                   "kernel_sources": pmeta["kernel_sources"] + [a.weights_kernel],     # was the constant: a latent bug
                   "model_sources": pmeta["model_sources"], "docker_image": pmeta["docker_image"]}
    nb = dict(pub, cells=new)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "competition_sources": ["rsna-knee-abnormality-detection"], **sources,
            "machine_shape": "NvidiaTeslaT4"}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if core:
        print(f"built {a.out / (slug + '.ipynb')}: {len(new)} cells ({len(cells)} public, core swapped; + intro, "
              f"core, core check" + (", ours, blend" if a.add_family else "") + ")")
        print(f"  core {a.core_kernel} last.pt {a.core_sha[:16]} -> {core_file} in {', '.join(CORE_ARMS)} | "
              + (f"+ family {a.weights_kernel} at w {a.w_ours:.4f} | " if a.add_family else "no family | ")
              + f"from {built_from}")
    else:
        print(f"built {a.out / (slug + '.ipynb')}: {len(new)} cells ("
              + ("Submission A: ours alone" if a.ours_only else f"{len(cells)} public + intro, ours, blend") + ")")
        print(f"  windows {a.n_windows or 'all'} | " + ("" if a.ours_only else f"w_ours {a.w_ours:.4f} | ")
              + f"check studies {a.n_check} | from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")
    print(f"push:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
