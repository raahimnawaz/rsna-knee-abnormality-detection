r"""Build a Kaggle kernel that runs our models alone: the full submission, or a single-finding probe.

    python notebooks/build_submit_kernel.py --weights-kernel <owner/kernel> --weights-sha <sha256> \\
        --tag <name> [--segmenter] [--probe "Lateral OA"] --out <dir>
    python -m kaggle kernels push -p <dir>

`OURS` (the gate and the test inference) is also the stage that build_public_blend_kernel.py adds to a public blend.
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
CODE_DATASET = "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"     # the SKM-TEA segmenter K needs
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
EMBED = ["fusion/infer_windows.py", "fusion/train_windows.py", "pipeline/kcache.py",
         "pipeline/build_slots64.py", "pipeline/slots64.py", "pipeline/slots96.py", "extractor/metrics.py"]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


OURS = r'''
# Our models: the gate on held-out studies, then every test study. Torch runs in subprocesses.
import base64 as _ob64, hashlib as _ohash, json as _ojson, os as _oos, subprocess as _osp
import sys as _osys, time as _otime, csv as _ocsv, shutil as _oshutil
from pathlib import Path as _OPath

_O_T0 = _otime.time()
_O = _OPath("/tmp/ours_sub")
_O_REPO = _O / "repo"
_O_N_CHECK = __N_CHECK__
_O_N_WINDOWS = __N_WINDOWS__          # 0 = every window, as the OOF was scored
_O_CHECK_ALWAYS = __CHECK_ALWAYS__    # False: gate in the commit run only (efficiency entries)


def _olog(m):
    print(f"[ours {_otime.time() - _O_T0:7.1f}s] {m}", flush=True)


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
_oms = [_ojson.loads((_c.parent / "last_meta.json").read_text()) for _c in _O_CKPTS]
_O_ARM = "+".join(sorted({f"{_m['arch']}_{str(_m['layout']).split()[0]}" for _m in _oms}))
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
    raise RuntimeError(f"our predictions cover {len(_rows)} studies, test.csv has {len(_test_ids)}")
with open(_O / "ours_preds.csv", "w", newline="") as _f:
    _w = _ocsv.writer(_f)
    _w.writerow(_hdr)
    for _u in _test_ids:
        _w.writerow(_rows[_u])
_O_MIN = (_otime.time() - _O_T0) / 60
_olog(f"OURS DONE: {len(_rows)} test studies in {_O_MIN:.1f} min on {_O_NGPU} GPU(s), "
      f"windows {_O_N_WINDOWS or 'all'} -> {_O / 'ours_preds.csv'}")
'''


ALONE = r'''
from pathlib import Path as _APath
_a_src = _APath("/tmp/ours_sub/ours_preds.csv")
_a_hdr = _a_src.read_text().splitlines()[0].split(",")
if _a_hdr[0] != "StudyInstanceUID" or len(_a_hdr) != 13:
    raise RuntimeError(f"predictions header {_a_hdr}")
_A_PROBE = __PROBE__
if _A_PROBE:
    _a_rows = [l.split(",") for l in _a_src.read_text().splitlines()]
    _a_j = _a_hdr.index(_A_PROBE)
    _APath("/kaggle/working/submission.csv").write_text("\n".join(
        [",".join(_a_hdr)] + [",".join([r[0]] + [r[k] if k == _a_j else "0.5" for k in range(1, 13)])
                              for r in _a_rows[1:]]) + "\n")
    print(f"[ours] PROBE: only {_A_PROBE} is the model's; the other 11 columns are 0.5", flush=True)
else:
    _APath("/kaggle/working/submission.csv").write_text(_a_src.read_text())
print(f"[ours] ARMS: {_O_ARM} only, windows {_O_N_WINDOWS or 'all'}; {len(_test_ids)} studies; "
      f"ours {_O_MIN:.1f} min of the run", flush=True)
'''

INTRO = """# RSNA Knee -- our models alone

__DESC__ Every test study is read straight from its DICOMs, at __WINDOWS__ windows per study, on both T4s, after the
gate reproduces each checkpoint's held-out predictions. Built from commit `__BUILT_FROM__`."""


def _check_import_closure(embed: list[str]) -> None:
    """Refuse to build if an embedded file imports a repo module that is not embedded."""
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
    ap.add_argument("--weights-kernel", default="", help="training kernels holding the run dirs, comma-joined")
    ap.add_argument("--weights-dataset", default="", help="and / or private datasets holding run dirs, comma-joined")
    ap.add_argument("--weights-sha", required=True, help="each last.pt's sha256, comma-joined (an equal-weight ensemble)")
    ap.add_argument("--tag", required=True, help="slug suffix")
    ap.add_argument("--desc", default="", help="one line for the intro: what these checkpoints are")
    ap.add_argument("--segmenter", action="store_true", help="mount the SKM-TEA segmenter (K checkpoints)")
    ap.add_argument("--probe", default="", choices=[""] + LABELS,
                    help="submit this finding alone, the other eleven at 0.5: the score reads (5.5 + AUROC) / 12")
    ap.add_argument("--n-windows", type=int, default=0, help="thin to ~n windows per study; 0 = all (as scored)")
    ap.add_argument("--n-check", type=int, default=8, help="gate studies in all")
    a = ap.parse_args()
    shas = a.weights_sha.split(",")
    wkernels = [k for k in a.weights_kernel.split(",") if k]
    wdatasets = [d for d in a.weights_dataset.split(",") if d]
    if not wkernels and not wdatasets:
        raise SystemExit("pass --weights-kernel and / or --weights-dataset")
    if len(set(shas)) != len(shas) or not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in shas):
        raise SystemExit("--weights-sha: full 64-hex sha256 values of each last.pt, joined by commas")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", a.tag):
        raise SystemExit("--tag: lowercase, digits, '-'")

    _check_import_closure(EMBED)
    files = {}
    for rel in EMBED:
        b = (PROJ / rel).read_bytes()
        files[rel] = [base64.b64encode(b).decode(), hashlib.sha256(b).hexdigest()]
    blob = base64.b64encode(json.dumps(files).encode()).decode()
    built_from = _git_head()
    wsrc = ",".join(wkernels + wdatasets)
    ours = (OURS.replace("__FILES_B64__", repr(blob)).replace("__BUILT_FROM__", built_from)
            .replace("__N_CHECK__", str(a.n_check)).replace("__N_WINDOWS__", str(a.n_windows))
            .replace("__CHECK_ALWAYS__", "False").replace("__WEIGHTS_SHAS__", repr(shas))
            .replace("__WEIGHTS_KERNEL__", wsrc))
    alone = ALONE.replace("__PROBE__", repr(a.probe))
    for k, c in enumerate([ours, alone]):
        compile(c, f"cell{k}", "exec")

    slug = (f"rsna-knee-sub-{a.tag}-n{a.n_windows or 'all'}"
            + (f"-p-{re.sub('[^a-z]+', '-', a.probe.lower()).strip('-')}" if a.probe else ""))
    if len(slug) > 50:
        raise SystemExit(f"slug {slug}: Kaggle refuses more than 50 characters")
    intro = (INTRO.replace("__DESC__", a.desc or "Our checkpoints.").replace("__WINDOWS__", str(a.n_windows or "all"))
             .replace("__BUILT_FROM__", built_from)
             + f"\n\nWeights: `{wsrc}`, last.pt sha256 " + ", ".join(f"`{h[:16]}`" for h in shas) + ".")

    def code(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.strip("\n").splitlines(keepends=True)}

    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)},
                    code(ours), code(alone)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "keywords": [], "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": wdatasets + ([CODE_DATASET] if a.segmenter else []), "kernel_sources": wkernels,
            "model_sources": [], "docker_image": DOCKER_IMAGE, "machine_shape": "NvidiaTeslaT4"}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (slug + '.ipynb')} | {len(shas)} checkpoint(s) | windows {a.n_windows or 'all'} | "
          f"probe {a.probe or 'none'} | from {built_from}")
    if "uncommitted" in built_from:
        print("  embedded files differ from HEAD -- commit first so the run is reproducible")
    print(f"push:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
