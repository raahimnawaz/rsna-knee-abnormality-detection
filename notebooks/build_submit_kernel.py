"""Submission B: our Stage A OrthoFoundation-L model rank-meaned into the public 0.943 stack.

    python -m kaggle kernels pull mattiaangeli/bend-the-knee-to-speedy-raptors-the-original -p <pub> -m
    python notebooks/build_submit_kernel.py --public-dir <pub> --out /tmp/subb
    python -m kaggle kernels push -p /tmp/subb          # commit run on the 3 visible test studies
    # then, only if the commit log shows CHECK PASSED and the ARMS line:
    python -m kaggle competitions submit rsna-knee-abnormality-detection -k raahimnawaz/<slug> -v <n> \
        -f submission.csv -m "..."

WHAT IT RUNS, IN ORDER.
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
SLUG = "rsna-knee-sub-b-ortho-blend"
PUBLIC = "mattiaangeli/bend-the-knee-to-speedy-raptors-the-original"
WEIGHTS_KERNEL = f"{OWNER}/rsna-knee-c6-train-ortho"
WEIGHTS_SHA256 = "1fe51be2ff4deaae4dfd5fafd2354f3b9bbeb0c26cda3fd918a00699fd481b98"   # runs_ortho/last.pt
EMBED = ["fusion/infer_windows.py", "fusion/train_windows.py", "pipeline/build_slots64.py",
         "pipeline/slots64.py"]
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
    if "last.pt" in _fs and "oof_all.csv" in _fs and _OPath(_d).name == "runs_ortho":
        _c = _OPath(_d) / "last.pt"
        if _ohash.sha256(_c.read_bytes()).hexdigest() == "__WEIGHTS_SHA256__":
            _O_CKPT = _c
if _O_ROOT is None:
    raise RuntimeError("competition root not found under /kaggle/input")
if _O_CKPT is None:
    raise RuntimeError("runs_ortho/last.pt with the recorded sha256 not found -- is __WEIGHTS_KERNEL__ attached?")
_olog(f"competition root {_O_ROOT} | weights {_O_CKPT} (sha256 ok)")

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
if (_O_ROOT / "train_series").is_dir():
    with open(_O / "check_uids.csv", "w") as _f:
        _f.write("StudyInstanceUID\n" + "\n".join(_val[::max(len(_val) // _O_N_CHECK, 1)][:_O_N_CHECK]) + "\n")
    _owait({"check": _orun(["--split", "train", "--uids", str(_O / "check_uids.csv"), "--out",
                            str(_O / "check" / "check.csv"), "--check-against", str(_oof),
                            "--workers", "4"], 0, "check")}, "the OOF reproduction check")
    _olog("GATE PASSED: " + (_O / "check" / "check.check.json").read_text().split('"per_study_max"')[0])
else:
    _olog("train DICOMs are not mounted in this run; the gate ran in the commit run on this code")

# ---- 2. every test study, one process per GPU
_procs = {f"test{i}": _orun(["--split", "test", "--shard", str(i), "--n-shards", str(_O_NGPU),
                             "--workers", "2" if _O_NGPU > 1 else "4",
                             "--out", str(_O / f"preds_{i}.csv")], i, f"test{i}") for i in range(_O_NGPU)}
_owait(_procs, "test inference")

with open(_O_ROOT / "test.csv") as _f:
    _test_ids = [r["StudyInstanceUID"] for r in _ocsv.DictReader(_f)]
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
_olog(f"OURS DONE: {len(_rows)} test studies in {_O_MIN:.1f} min on {_O_NGPU} GPU(s) -> {_O / 'ortho_preds.csv'}")
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
    "families": ["public: transformer", "public: CoAtNet", "ours: ortho_stage_a"],
    "n_studies": len(_b_pub), "spearman_ours_vs_public": _b_rho, "ours_minutes": _O_MIN,
    "built_from": "__BUILT_FROM__"}, indent=1))
print(f"[blend] ARMS: public stack (2 families, as published) + ortho_stage_a = 3 families; "
      f"w_ours {_O_W:.4f}; {len(_b_pub)} studies -> {_b_stage}", flush=True)
'''


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
    ap.add_argument("--public-dir", type=Path, required=True, help=f"`kaggle kernels pull {PUBLIC} -m`")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--w-ours", type=float, default=1 / 3, help="the rule gives 1/3; see docstring")
    ap.add_argument("--n-check", type=int, default=8)
    a = ap.parse_args()

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

    files = {}
    for rel in EMBED:
        b = (PROJ / rel).read_bytes()
        files[rel] = [base64.b64encode(b).decode(), hashlib.sha256(b).hexdigest()]
    blob = base64.b64encode(json.dumps(files).encode()).decode()
    built_from = f"{_git_head()}, public notebook sha256 {pub_sha[:16]}"
    ours = (OURS.replace("__FILES_B64__", repr(blob)).replace("__BUILT_FROM__", built_from)
                .replace("__W_OURS__", repr(a.w_ours)).replace("__N_CHECK__", str(a.n_check))
                .replace("__WEIGHTS_SHA256__", WEIGHTS_SHA256).replace("__WEIGHTS_KERNEL__", WEIGHTS_KERNEL))
    blend = BLEND.replace("__LABELS__", repr(LABELS)).replace("__BUILT_FROM__", built_from)
    for k, c in enumerate([ours, blend]):
        compile(c, f"ours{k}", "exec")                  # fail here, not after the queue wait

    def code(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.strip("\n").splitlines(keepends=True)}

    intro = {"cell_type": "markdown", "metadata": {},
             "source": INTRO.replace("__BUILT_FROM__", built_from).splitlines(keepends=True)}
    body = [dict(c, outputs=[], execution_count=None) if c["cell_type"] == "code" else c for c in cells]
    new = [intro, code(ours)] + body[:-1] + [code(blend), body[-1]]
    nb = dict(pub, cells=new)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "dataset_sources": pmeta["dataset_sources"],
            "kernel_sources": pmeta["kernel_sources"] + [WEIGHTS_KERNEL],
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "model_sources": pmeta["model_sources"],
            "docker_image": pmeta["docker_image"], "machine_shape": "NvidiaTeslaT4"}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (SLUG + '.ipynb')}: {len(new)} cells ({len(cells)} public + intro, ours, blend)")
    print(f"  w_ours {a.w_ours:.4f} | check studies {a.n_check} | from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")
    print(f"push:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
