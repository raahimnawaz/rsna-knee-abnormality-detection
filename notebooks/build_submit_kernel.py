r"""Build the Kaggle submission kernel: our model alone, blended into the public ensemble, or as its core.

    python notebooks/build_submit_kernel.py --ours-only --out /tmp/sub                 # our model alone
    python notebooks/build_submit_kernel.py --public-dir <pub> --out /tmp/sub          # + the public ensemble, ours at 1/3
    python notebooks/build_submit_kernel.py --public-dir <pub> --core-kernel <kernel> \
        --core-sha <sha256> --tag <name> --out /tmp/sub                                # ours as the ensemble's core
    python notebooks/build_submit_kernel.py --public-dir <pub> --core-alone --out /tmp/sub   # the public core alone
    python -m kaggle kernels push -p /tmp/sub
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
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
EMBED = ["fusion/infer_windows.py", "fusion/train_windows.py", "pipeline/kcache.py",
         "pipeline/build_slots64.py", "pipeline/slots64.py", "pipeline/slots96.py", "pipeline/slots118.py",
         "fusion/orthofoundation_model.py", "extractor/metrics.py",
         "fusion/qwen_vlm.py"]
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
_O_SHAS = __WEIGHTS_SHAS__            # one checkpoint, or the members of an equal-weight ensemble
_O_ROOT, _o_found = None, {}
for _d, _dirs, _fs in _oos.walk("/kaggle/input"):
    _dirs[:] = [x for x in _dirs if x not in _SKIP]
    if _O_ROOT is None and {"test.csv", "test_series.csv", "train_series.csv"} <= set(_fs):
        _O_ROOT = _OPath(_d)
    if "last.pt" in _fs and "oof_all.csv" in _fs:          # a run dir: weights beside the predictions they scored
        _c = _OPath(_d) / "last.pt"
        _h = _ohash.sha256(_c.read_bytes()).hexdigest()
        if _h in _O_SHAS:
            _o_found[_h] = _c
if _O_ROOT is None:
    raise RuntimeError("competition root not found under /kaggle/input")
if set(_o_found) != set(_O_SHAS):
    raise RuntimeError(f"last.pt not found for sha256 {sorted(set(_O_SHAS) - set(_o_found))} -- is __WEIGHTS_KERNEL__ attached?")
_O_CKPTS = [_o_found[_h] for _h in _O_SHAS]
_O_CKPT = _O_CKPTS[0]
_O_ARM = "ortho_stage_a"
if (_O_CKPT.parent / "last_meta.json").exists():
    _oms = [_ojson.loads((_c.parent / "last_meta.json").read_text()) for _c in _O_CKPTS]
    _O_ARM = "+".join(sorted({f"{_m.get('arch', 'ortho')}_{str(_m.get('layout', 'stage_a')).split()[0]}" for _m in _oms}))
if len(_O_CKPTS) > 1:
    _O_ARM += f" x{len(_O_CKPTS)}"
_olog(f"competition root {_O_ROOT} | {len(_O_CKPTS)} checkpoint(s), sha256 ok: {[str(_c) for _c in _O_CKPTS]} | arm {_O_ARM}")

try:
    _O_NGPU = len([l for l in _osp.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                       check=True).stdout.splitlines() if l.startswith("GPU ")])
except Exception as _e:                                    # noqa: BLE001
    _olog(f"nvidia-smi failed ({_e}); assuming one GPU")
    _O_NGPU = 1
_olog(f"GPUs: {_O_NGPU}")
_O_PY = [_osys.executable, "-u", str(_O_REPO / "fusion" / "infer_windows.py"), "--root", str(_O_ROOT)]
_o_seg_w = _o_seg_v = None                                 # K checkpoints need the segmenter that finds the knee
for _d, _dirs, _fs in _oos.walk("/kaggle/input"):
    _dirs[:] = [x for x in _dirs if x not in _SKIP]
    if _o_seg_w is None and "anatomy.pt" in _fs:
        _o_seg_w = _oos.path.join(_d, "anatomy.pt")
    if _o_seg_v is None and _d.endswith(_oos.path.join("vendor_runtime", "monai060")):
        _o_seg_v = _d
if _o_seg_w and _o_seg_v:
    _O_PY += ["--seg-weights", _o_seg_w, "--seg-vendor", _o_seg_v]
    _olog(f"segmenter for K: {_o_seg_w} | {_o_seg_v}")
_O_ALL = ["--weights"] + [str(_c) for _c in _O_CKPTS]


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
with open(_O_ROOT / "test.csv") as _f:
    _test_ids = [r["StudyInstanceUID"] for r in _ocsv.DictReader(_f)]
if (_O_ROOT / "train_series").is_dir() and (_O_CHECK_ALWAYS or len(_test_ids) < 100):
    _o_n = max(3, _O_N_CHECK // len(_O_CKPTS))             # every member reproduces ITS OWN held-out predictions
    for _j, _c in enumerate(_O_CKPTS):
        _oof = _c.parent / "oof_all.csv"
        with open(_oof) as _f:
            _val = sorted({r["StudyInstanceUID"] for r in _ocsv.DictReader(_f)})
        with open(_O / f"check_uids_{_j}.csv", "w") as _f:
            _f.write("StudyInstanceUID\n" + "\n".join(_val[::max(len(_val) // _o_n, 1)][:_o_n]) + "\n")
        _owait({f"check{_j}": _orun(["--weights", str(_c), "--split", "train", "--uids", str(_O / f"check_uids_{_j}.csv"),
                                     "--out", str(_O / f"check{_j}" / "check.csv"), "--check-against", str(_oof),
                                     "--workers", "4"], _j % _O_NGPU, f"check{_j}")}, "the OOF reproduction check")
        _olog(f"GATE PASSED ({_j + 1}/{len(_O_CKPTS)}): "
              + (_O / f"check{_j}" / "check.check.json").read_text().split('"per_study_max"')[0])
elif not (_O_ROOT / "train_series").is_dir():
    _olog("train DICOMs are not mounted in this run; the gate ran in the commit run on this code")
else:
    _olog(f"{len(_test_ids)} test studies: the scored rerun -- the commit run carried the gate on this code")

# ---- 2. every test study, one process per GPU
_procs = {f"test{i}": _orun(_O_ALL + ["--split", "test", "--shard", str(i), "--n-shards", str(_O_NGPU),
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
# Equal rank-mean over families, nothing fitted: ours 1/3, the
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
_A_PROBE = __PROBE__      # one finding: every other column is constant, so the public score = (5.5 + its AUROC) / 12
if _A_PROBE:
    _a_rows = [l.split(",") for l in _a_src.read_text().splitlines()]
    _a_j = _a_hdr.index(_A_PROBE)
    _APath("/kaggle/working/submission.csv").write_text("\n".join(
        [",".join(_a_hdr)] + [",".join([r[0]] + [r[k] if k == _a_j else "0.5" for k in range(1, 13)])
                              for r in _a_rows[1:]]) + "\n")
    print(f"[sub A] PROBE: only {_A_PROBE} is the model's; the other 11 columns are 0.5", flush=True)
else:
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
_c_num = sum(float((_c_out[k].double() - _c_rsd[k].double()).pow(2).sum()) for k in _c_rsd if _c_rsd[k].is_floating_point())
_c_den = sum(float(_c_rsd[k].double().pow(2).sum()) for k in _c_rsd if _c_rsd[k].is_floating_point())
_C_MOVED = (_c_num / _c_den) ** 0.5

_C_DIR.mkdir(parents=True, exist_ok=True)
_c_path = _C_DIR / _C_FILE
_ctorch.save({"model": _c_out, "arch": _c_ref["arch"], "res": _c_ref["res"],
              "source_kernel": _C_KERNEL, "source_sha256": _C_SHA}, _c_path)
_C_OUT_SHA = _csha(_c_path)
ASSET_ROOTS.append(str(_C_DIR))
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

CORE_PUBLIC_SHA256 = "20d45e0732a0e9120025d0172a4d2dd0170900bc250b06742dfb7ae08a63786a"   # CC0, as attached

ALONE = r'''
# ============================================================================ #
# The core alone, part 2 -- ONLY the public stack's core: maxspan-v5 and its slice-reversed pass
# (w 0.60 + 0.10, one shared file), through the stack's own Raptor code. Public cell -2 follows
# verbatim, less its DINOv2 gate, its read of the stage file and its four-arm runner. Then our
# runner keeps the two core arms and writes their blend at the core's own weights, ranked.
# ============================================================================ #
__KE_IMPORTS__
__KE_DEFS__
__KE_STATE__
# ---- ours from here: the two core arms on both T4s, the studies split by index
import contextlib as _al_ctxlib
import time as _al_time

_AL_T0 = _al_time.time()
_AL_FILE = "__AL_FILE__"            # the file both core arms load: our student's, or the public v5 (the baseline)
_AL_PUBLIC = "__CORE_PUBLIC__"
_AL_ARMS = ("maxspan-v5", "maxspan-v5-reverse")
_al_torch = _KE_NS["torch"]
_al_arms = [a for a in _KE_NS["ARMS"] if a["name"] in _AL_ARMS]
if [a["name"] for a in _al_arms] != list(_AL_ARMS) or any(a["file"] != _AL_PUBLIC for a in _al_arms):
    raise RuntimeError(f"public ARMS changed: {[(a['name'], a['file']) for a in _KE_NS['ARMS']]}")
if (any(_al_arms[0][k] != _al_arms[1][k] for k in ("file", "arch", "res", "img", "slots", "span", "k_eval"))
        or _al_arms[0]["reverse"] or not _al_arms[1]["reverse"]):
    raise RuntimeError("the core's forward and reverse arms no longer share preprocessing")
for _a in _al_arms:
    _a["file"] = _AL_FILE
_KE_NS["ARMS"][:] = _al_arms                    # nothing else can load
_AL_W = _ke_np.asarray([float(a["w"]) for a in _al_arms], _ke_np.float64)
_AL_W /= _AL_W.sum()                            # 6/7 and 1/7, the core's share of the public Raptor blend

_al_root = _KE_NS["find_test_root"]()
_al_series_root = _al_root + "/test_series"
if not _ke_os.path.isdir(_al_series_root):
    _al_series_root = _al_root + "/test_images"
_al_test = _ke_pd.read_csv(_al_root + "/test.csv")
_al_ids = _al_test["StudyInstanceUID"].astype(str).tolist()
_al_ts = _ke_pd.read_csv(_al_root + "/test_series.csv")
_al_ts["StudyInstanceUID"] = _al_ts["StudyInstanceUID"].astype(str)
_al_ts["SeriesInstanceUID"] = _al_ts["SeriesInstanceUID"].astype(str)
_al_series = {k: f.to_dict("records") for k, f in _al_ts.groupby("StudyInstanceUID")}
_al_cols = ["StudyInstanceUID", *_KE_NS["LAB"]]
if _ke_os.path.exists(_al_root + "/sample_submission.csv"):
    _al_cols = list(_ke_pd.read_csv(_al_root + "/sample_submission.csv", nrows=1).columns)
if _al_cols[1:] != list(_KE_NS["LAB"]) or _al_cols[1:] != list(_RSNA_LABELS):
    raise RuntimeError(f"label order drift: sample {_al_cols[1:]}, Raptor {_KE_NS['LAB']}")
rsna_initialize(_al_ids)
_al_out = [_ke_np.full((len(_al_ids), len(_KE_NS["LAB"])), _ke_np.nan, _ke_np.float32) for _ in _al_arms]
_AL_DEVICES = [_al_torch.device(f"cuda:{_i}") for _i in range(2)]


def _al_ctx(device):
    return _al_torch.cuda.device(device) if device.type == "cuda" else _al_ctxlib.nullcontext()


def _al_run(device, part):
    """run_shared_maxspan on the studies `part` (indices into _al_ids): one decode, forward + reversed."""
    first = _al_arms[0]
    with _al_ctx(device):
        reader = _KE_NS["_make_reader"]()
        model, resolution = _KE_NS["load_model"](_KE_NS["find_weight_file"](first["file"]), first["arch"],
                                                 first["res"], device)
        if int(resolution) != int(first["res"]):
            raise RuntimeError(f"core checkpoint resolution {resolution} != {first['res']}")
        for j, uid, future in _ke_prefetched_windows(first, [_al_ids[i] for i in part], _al_series,
                                                     _al_series_root, reader):
            rsna_deadline("core-alone inference")
            i = part[j]
            try:
                windows = future.result()
                _al_out[0][i] = _KE_NS["infer_probs"](model, windows, device)
                _al_out[1][i] = _KE_NS["infer_probs"](model, windows.flip(1).contiguous(), device)
                del windows
            except Exception as error:
                rsna_event("raptor_study_failed", arm=str(first["name"]), study=str(uid),
                           error=f"{type(error).__name__}: {str(error)[:500]}")
        del model
        _ke_gc.collect()
        if device.type == "cuda":
            _al_torch.cuda.empty_cache()


rsna_phase("core_alone", "START")
with _KeThreadPool(max_workers=len(_AL_DEVICES)) as _al_ex:
    _al_fs = [_al_ex.submit(_al_run, d, list(range(k, len(_al_ids), len(_AL_DEVICES))))
              for k, d in enumerate(_AL_DEVICES)]
    for _al_f in _al_fs:
        _al_f.result()
rsna_phase("core_alone", "COMPLETE")
for _ai, _arr in enumerate(_al_out):            # the public runner's neutral fill, unchanged
    _bad = ~_ke_np.isfinite(_arr).all(axis=1)
    if _bad.any():
        _fill = (_ke_np.nanmean(_ke_np.where(_ke_np.isfinite(_arr), _arr, _ke_np.nan), axis=0) if (~_bad).any()
                 else _ke_np.full(_arr.shape[1], .5, _arr.dtype))
        _arr[_bad] = _ke_np.where(_ke_np.isfinite(_fill), _fill, .5)
        rsna_event("raptor_neutral_fill", arm=str(_al_arms[_ai]["name"]), studies=int(_bad.sum()))
rsna_finite(_ke_np.stack(_al_out), "the two core views", probability=True)
_al_blend = _ke_np.tensordot(_AL_W, _ke_np.stack([_ke_np.clip(v, 0, 1) for v in _al_out]), axes=(0, 0))
_al_sub = _ke_pd.DataFrame(_KE_NS["rankpct"](_al_blend).astype(_ke_np.float32), columns=_KE_NS["LAB"])
_al_sub.insert(0, "StudyInstanceUID", _al_ids)
_al_sub[_al_cols].to_csv("/kaggle/working/_core_alone.csv", index=False)
_ke_np.savez_compressed("/kaggle/working/core_alone_raw.npz", study_uids=_ke_np.asarray(_al_ids),
                        raw_probabilities=_ke_np.stack(_al_out))
_AL_MIN = (_al_time.time() - _AL_T0) / 60
print(f"[alone] {len(_al_ids)} studies, maxspan-v5 + reverse on {len(_AL_DEVICES)} device(s), "
      f"{_AL_MIN:.1f} min -> /kaggle/working/_core_alone.csv", flush=True)
'''

ALONE_CHECK = r'''
# ============================================================================ #
# The core alone, part 3 -- no submission.csv unless the core loaded THE ONE file it should
# (once per GPU, at its sha256) and nothing else did. The commit run also refuses a failed study.
# ============================================================================ #
import json as _xjson
from pathlib import Path as _XPath

_X_SHA = __X_SHA__
_X_WHO = __WHO__
_x_loads = [e for e in _RSNA_AUDIT["events"] if e.get("kind") == "raptor_checkpoint"]
_x_names = [_XPath(str(e.get("file", ""))).name for e in _x_loads]
if _x_names != [_AL_FILE] * len(_AL_DEVICES):
    raise RuntimeError(f"loads {_x_names}, want {_AL_FILE} once per device -- stopping")
if any(e.get("sha256") != _X_SHA for e in _x_loads):
    raise RuntimeError(f"{_AL_FILE} loaded with a sha256 that is not {_X_SHA[:16]} -- stopping")
_x_failed = [e for e in _RSNA_AUDIT["events"] if e.get("kind") == "raptor_study_failed"]
if _x_failed and len(_al_ids) < 100:
    raise RuntimeError(f"{len(_x_failed)} studies failed in the commit run: {_x_failed[:2]} -- stopping")
_x_frame = rsna_frame(_ke_pd.read_csv("/kaggle/working/_core_alone.csv", dtype={"StudyInstanceUID": str}),
                      _al_ids, _al_cols[1:], "core alone")
_x_frame.to_csv("/kaggle/working/submission.csv", index=False)
_XPath("/kaggle/working/alone_receipt.json").write_text(_xjson.dumps({
    "rule": "only the public stack's core (maxspan-v5 + maxspan-v5-reverse, w 6:1), the stack's own code",
    "weights": _X_WHO, "core_file": _AL_FILE, "core_file_sha256": _X_SHA, "loads": len(_x_loads),
    "studies": len(_al_ids), "failed_studies": len(_x_failed), "minutes": _AL_MIN,
    "built_from": "__BUILT_FROM__"}, indent=1))
print(f"[alone] ARMS: maxspan-v5 + maxspan-v5-reverse ONLY (w 6/7 + 1/7) = {_X_WHO}; {_AL_FILE} loaded "
      f"{len(_x_loads)}x, sha256 {_X_SHA[:16]}; {len(_al_ids)} studies, {len(_x_failed)} failed; "
      f"{_AL_MIN:.1f} min -> submission.csv", flush=True)
'''

INTRO_ALONE = """# RSNA Knee -- the public stack's core, alone: __WHO_MD__

Only the core of `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` (Apache 2.0): its
CoAtNet `maxspan-v5` and the slice-reversed pass of the same file, weighted 6:1 as the stack weights
them, run by the stack's own code (public cells 1-3 unmodified, and cell -2's Raptor code less its
DINOv2 gate, stage read and other two Raptor arms). Nothing else of the stack runs. A last cell refuses
the run unless the load log shows the one expected file at its sha256. Built by
`notebooks/build_submit_kernel.py --core-alone` (__BUILT_FROM__)."""

INTRO_CORE = """# RSNA Knee -- the upgraded stack: our CoAtNet student in the public stack's core slot

The public stack `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` (Apache 2.0) runs
as published but for ONE thing. Its core CoAtNet (`maxspan-v5`, weight 0.60, and its slice-reversed
pass, 0.10, which share one file) loads our student's weights. The student is the same model
(`dreaddevelopment/raptor-knee-maxspan`, CC0), fine-tuned on report labels corrected by an image
teacher. Every other member, weight and calibrator is untouched. A cell before the publish cell
refuses the run unless the stack's own load log shows our file at its sha256. Built by
`notebooks/build_submit_kernel.py` (__BUILT_FROM__)."""


def _check_import_closure(embed: list[str]) -> None:
    """Refuse to build if an embedded file imports a repo module that is not embedded."""
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
    ap.add_argument("--segmenter", action="store_true", help="mount heliosli's code dataset (the SKM-TEA segmenter and "
                    "its vendored MONAI, both sha-checked): needed by K checkpoints")
    ap.add_argument("--n-windows", type=int, default=0, help="thin to ~n windows per study; 0 = all (as scored)")
    ap.add_argument("--w-ours", type=float, default=1 / 3, help="the rule gives 1/3; see docstring")
    ap.add_argument("--n-check", type=int, default=8)
    ap.add_argument("--weights-kernel", default=WEIGHTS_KERNEL)
    ap.add_argument("--weights-dataset", default="", help="private dataset(s), comma-joined, holding run dirs (last.pt, "
                    "last_meta.json and oof_all.csv side by side): weights trained off Kaggle, from `notebooks/"
                    "offkaggle.py pack`. With it --weights-kernel is not needed; the sha256 and the gate are the same")
    ap.add_argument("--weights-sha", default=WEIGHTS_SHA256)
    ap.add_argument("--tag", default="", help="slug suffix for a submission on another checkpoint")
    ap.add_argument("--alone-lb", type=float, default=None,
                    help="another checkpoint in Submission B: its own LB number, alone -- the guard's precondition")
    ap.add_argument("--desc", default="", help="one line for the notebook's intro: what this checkpoint is")
    ap.add_argument("--probe", default="", choices=[""] + LABELS,
                    help="with --ours-only: submit this one finding and hold the other eleven at 0.5, so the public "
                         "score reads its AUROC alone: (5.5 + AUROC) / 12, one leaderboard tick = 0.012")
    # Core swap: a CoAtNet student replaces the public core.
    ap.add_argument("--core-kernel", default=None, help="the student's training kernel (holds runs_*/last.pt)")
    ap.add_argument("--core-sha", default=None, help="sha256 of that last.pt, all 64 hex digits")
    ap.add_argument("--add-family", action="store_true",
                    help="with --core-*: ALSO add the --weights-kernel model as a family at 1/3 (rule (C))")
    ap.add_argument("--core-alone", action="store_true",
                    help="only the stack's core, alone (rule (B)): with --core-*, our student; without, the public "
                         "weights (the paired baseline)")
    a = ap.parse_args()
    core = bool(a.core_kernel or a.core_sha)
    if a.core_alone and (a.ours_only or a.add_family or not a.public_dir):
        raise SystemExit("--core-alone runs the public stack's own core code: pass --public-dir, not --ours-only "
                         "or --add-family")
    if core:
        if not (a.core_kernel and a.core_sha and re.fullmatch(r"[0-9a-f]{64}", a.core_sha)):
            raise SystemExit("--core-kernel and --core-sha go together; the sha is the full 64-hex sha256 of last.pt")
        if a.ours_only or not a.public_dir:
            raise SystemExit("the upgraded stack is the public stack: pass --public-dir, not --ours-only")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", a.tag or ""):
            raise SystemExit("the upgraded stack needs a --tag (lowercase, digits, '-'): it names the slug and the core file")
    elif a.add_family:
        raise SystemExit("--add-family only means something with --core-kernel (Submission B always adds a family)")
    shas, wkernels = a.weights_sha.split(","), a.weights_kernel.split(",")
    wdatasets = [d for d in a.weights_dataset.split(",") if d]
    if wdatasets and a.weights_kernel == WEIGHTS_KERNEL:    # every checkpoint comes from the datasets
        wkernels = []
    custom = a.weights_kernel != WEIGHTS_KERNEL or bool(wdatasets)
    wsrc = ",".join(wkernels + wdatasets)
    if len(set(shas)) != len(shas) or not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in shas):
        raise SystemExit("--weights-sha: one full 64-hex sha256 of a last.pt, or several joined by commas (an ensemble)")
    if custom != (a.weights_sha != WEIGHTS_SHA256):
        raise SystemExit("--weights-kernel / --weights-dataset and --weights-sha move together: a new checkpoint needs both")
    if custom and not a.tag:
        raise SystemExit("another checkpoint needs a --tag, so its notebook does not overwrite the default one")
    if custom and not a.ours_only and a.alone_lb is None:
        raise SystemExit("another checkpoint: build it as --ours-only first and pass its own LB as --alone-lb "
                         "before it joins any blend (blending a weaker member costs)")
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
                .replace("__WEIGHTS_SHAS__", repr(shas)).replace("__WEIGHTS_KERNEL__", wsrc))
    blend = BLEND.replace("__LABELS__", repr(LABELS)).replace("__BUILT_FROM__", built_from)
    if a.probe and not a.ours_only:
        raise SystemExit("--probe reads one finding of our model alone: pass --ours-only")
    sub_a = SUB_A.replace("__PROBE__", repr(a.probe))
    for k, c in enumerate([ours, blend, sub_a]):
        compile(c, f"ours{k}", "exec")                  # fail here, not after the queue wait

    if a.ours_only:
        slug = (f"rsna-knee-sub-a-ortho{'-' + a.tag if a.tag else ''}-n{a.n_windows or 'all'}"
                + (f"-p-{re.sub('[^a-z]+', '-', a.probe.lower()).strip('-')}" if a.probe else ""))
        intro = INTRO_A.replace("__WINDOWS__", str(a.n_windows or "all")).replace("__BUILT_FROM__", built_from)
        if a.desc:
            intro += (f"\n\n**This build:** {a.desc} Weights: `{wsrc}`, last.pt sha256 "
                      + ", ".join(f"`{h[:16]}`" for h in shas) + ".")
        new = [md(intro), code(ours), code(sub_a)]
        sources = {"dataset_sources": wdatasets + (["heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]
                                                   if a.segmenter else []),
                   "kernel_sources": wkernels, "model_sources": [],
                   "docker_image": DOCKER_IMAGE}
    elif a.core_alone:
        # the seams this build relies on -- refuse an upstream version that moved any of them
        def one(pred, what):
            hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and pred("".join(c["source"]))]
            if len(hits) != 1:
                raise SystemExit(f"public notebook: {len(hits)} cells {what}, want one -- re-read it")
            return hits[0]
        i_roots = one(lambda s: bool(re.search(r"^ASSET_ROOTS = \[", s, re.M)) and "def _asset_find_asset" in s,
                      "define ASSET_ROOTS and _asset_find_asset")
        i_dense = one(lambda s: "def _dense_allocate" in s and "def _dense_unique_linspace" in s,
                      "define the dense sampler")
        i_rsna = one(lambda s: bool(re.search(r"^_RSNA_AUDIT = \{", s, re.M)) and all(
            f"def {f}(" in s for f in ("rsna_initialize", "rsna_frame", "rsna_deadline", "rsna_phase", "rsna_strict_load")),
                     "define the rsna_* audit")
        if [i_roots, i_dense, i_rsna] != [1, 2, 3]:
            raise SystemExit(f"public setup cells moved: assets {i_roots}, dense {i_dense}, rsna {i_rsna} (want 1, 2, 3)")
        gate = ("if globals().get('_DINOV2_MATCHED_MEMBERS') != 20:\n"
                "    raise RuntimeError('DINOv2 20/20 fingerprint gate did not pass')\n")
        marks = {"stage": "_ke_primary = _KePath('/kaggle/working/_pipeline_stage.csv')\n",
                 "src": "_KE_SRC = r'''", "runner": "\n\ndef _ke_run_raptor_arms():",
                 "state": "_ke_input_lock = _ke_threading.Lock()\n", "start": "rsna_phase('public_raptor', 'START')"}
        if not before.startswith(gate) or any(before.count(m) != 1 for m in marks.values()):
            raise SystemExit("public cell -2's Raptor seams moved (gate, stage read, _KE_SRC, runner, state) -- re-read it")
        at = {k: before.index(m) for k, m in marks.items()}
        if not at["stage"] < at["src"] < at["runner"] < at["state"] < at["start"]:
            raise SystemExit(f"public cell -2's Raptor seams are out of order: {at}")
        ke_imports = before[len(gate):at["stage"]]
        ke_defs = before[at["src"]:at["runner"]]
        ke_state = before[at["state"]:at["start"]]
        if any(not (ln.startswith(("import ", "from ")) or not ln.strip()) for ln in ke_imports.splitlines()):
            raise SystemExit("public cell -2 runs more than imports before its stage read -- re-read it")
        for name in ("_ke_ours", "_KE_LAB", "_ke_primary", "_DINOV2"):
            if name in ke_defs or name in ke_state:
                raise SystemExit(f"public cell -2's Raptor code now uses {name} -- re-read it")
        if ke_defs.count(CORE_PUBLIC) != 2:
            raise SystemExit(f"public cell -2 names {CORE_PUBLIC} {ke_defs.count(CORE_PUBLIC)} times, not twice")
        if "dreaddevelopment/raptor-knee-maxspan" not in pmeta["dataset_sources"]:
            raise SystemExit("the public stack no longer attaches dreaddevelopment/raptor-knee-maxspan")

        if core:
            al_file = f"student_core_{a.tag}_{a.core_sha[:12]}.pt"
            who = 'f"our {_C_ARM} student ({_C_KERNEL}, last.pt {_C_SHA[:16]})"'
            who_md = f"our CoAtNet student (`{a.core_kernel}`, last.pt sha256 `{a.core_sha[:16]}`)"
            x_sha = "_C_OUT_SHA"
            slug = f"rsna-knee-sub-alone-{a.tag}"
            core_src = (CORE.replace("__CORE_KERNEL__", a.core_kernel).replace("__CORE_SHA256__", a.core_sha)
                            .replace("__CORE_FILE__", al_file).replace("__CORE_PUBLIC__", CORE_PUBLIC))
            compile(core_src, "core", "exec")
        else:
            al_file = CORE_PUBLIC
            who = '"the public weights (dreaddevelopment/raptor-knee-maxspan, CC0): the paired baseline"'
            who_md = "the public weights (`dreaddevelopment/raptor-knee-maxspan`, CC0), the paired baseline"
            x_sha = repr(CORE_PUBLIC_SHA256)
            slug = "rsna-knee-sub-alone-public" + (f"-{a.tag}" if a.tag else "")
        alone_src = (ALONE.replace("__KE_IMPORTS__", ke_imports.strip("\n")).replace("__KE_DEFS__", ke_defs.strip("\n"))
                          .replace("__KE_STATE__", ke_state.strip("\n")).replace("__AL_FILE__", al_file)
                          .replace("__CORE_PUBLIC__", CORE_PUBLIC))
        check_src = (ALONE_CHECK.replace("__X_SHA__", x_sha).replace("__WHO__", who)
                                .replace("__BUILT_FROM__", built_from))
        compile(alone_src, "alone", "exec")
        compile(check_src, "alone_check", "exec")
        intro = INTRO_ALONE.replace("__WHO_MD__", who_md).replace("__BUILT_FROM__", built_from)
        if a.desc:
            intro += f"\n\n{a.desc}"
        body = [dict(c, outputs=[], execution_count=None) if c["cell_type"] == "code" else c for c in cells]
        new = ([md(intro), body[i_roots]] + ([code(core_src)] if core else [])
               + [body[i_dense], body[i_rsna], code(alone_src), code(check_src)])
        sources = {"dataset_sources": ["dreaddevelopment/raptor-knee-maxspan"],
                   "kernel_sources": [a.core_kernel] if core else [], "model_sources": [],
                   "docker_image": pmeta["docker_image"]}
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
            intro += (f"\n\n**This build:** {a.desc} Alone on the LB: {a.alone_lb}. Weights: `{wsrc}`, "
                      f"last.pt sha256 `{a.weights_sha[:16]}`.")
        new = [md(intro), code(ours)] + body[:-1] + [code(blend), body[-1]]
        sources = {"dataset_sources": pmeta["dataset_sources"] + wdatasets,
                   "kernel_sources": pmeta["kernel_sources"] + wkernels,
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
    if a.core_alone:
        print(f"built {a.out / (slug + '.ipynb')}: {len(new)} cells (intro, public cells 1-3"
              + (", core" if core else "") + ", the core alone, its check)")
        print(f"  maxspan-v5 + maxspan-v5-reverse load {al_file} | "
              + (f"student {a.core_kernel} last.pt {a.core_sha[:16]}" if core else "the public weights")
              + f" | from {built_from}")
    elif core:
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
