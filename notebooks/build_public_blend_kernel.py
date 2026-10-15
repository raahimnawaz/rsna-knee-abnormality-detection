r"""Build the submission kernel that adds our folds to a public blend notebook, as one more stage.

    kaggle kernels pull heliosli/this-is-new-version-0-9576-in-one-click -p <pub> -m      # any slug in PUBLIC
    python notebooks/build_public_blend_kernel.py --public-dir <pub> --weights-kernel <k1>,<k2>,<k3> \
        --weights-sha <64 hex>,<64 hex>,... --tag pub957-teach1 --out <dir>
    python -m kaggle kernels push -p <dir>

The public notebook's cells run unchanged and write /kaggle/working/submission.csv (one of the free recipes in
PUBLIC; the build refuses another slug, and any version whose seams below have moved).
Two cells follow: our model's stage (build_submit_kernel's OURS cell, under a deadline) and a per-finding rank blend
at the weights fixed below. In the scored rerun a failure or a timeout of our stage leaves the public submission as
it is; in the commit run it stops the notebook, so a fault is seen before anything is submitted.

Read in the commit log before submitting: `sha256 ok` · `GATE PASSED (n/n)` · `OURS_STAGE ok=True` ·
`[public+ours] ARMS:` · 0 tracebacks.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

import build_submit_kernel as B

# The public blend notebooks this build has been checked against, and what each one is (for the intro cell).
PUBLIC = {
    "varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956":
        "the free 0.956 recipe: nartaa's OAI-trained CoAtNet (384 px, and its 224 px sibling at 35% of that slot), "
        "heliosli's R7 ConvNeXt-small (four folds), per-finding ranks 50/50",
    "heliosli/this-is-new-version-0-9576-in-one-click":
        "the free 0.957 recipe (10-14): nartaa's OAI-trained CoAtNet (384 px, and its 224 px sibling at 35% of that "
        "half), heliosli's R6 bundle (four folds) with its flip pass on, per-finding ranks 50/50",
}

# Our folds' share of each finding's rank, fixed 10-13 before any run and fitted to nothing. 0.25 where the
# leaderboard probes of 10-12 put our five folds at 0.96 or more (MCL 0.968; the seven unprobed findings 0.966 on
# average), 0.10 where they read 0.88-0.93 and two public members carry supervision we lack. Within-class rank
# correlation of our folds with the public CoAtNet, in our input: 0.85-0.88.
W_OURS = {"ACL": 0.25, "MCL": 0.25, "Medial Meniscus": 0.25, "Medial OA": 0.25, "Effusion": 0.25, "Baker's": 0.25,
          "Contusion": 0.25, "Fracture": 0.25,
          "Lateral Meniscus": 0.10, "Lateral OA": 0.10, "PF OA": 0.10, "Synovitis": 0.10}

STAGE = r'''
# ============================================================================ #
# Our folds, one more stage after the public blend above has written its
# submission.csv. The commit run (three test studies) carries the gate and
# stops on any fault; the scored rerun keeps the public submission instead.
# ============================================================================ #
import csv as _pcsv, time as _ptime, importlib.metadata as _pmd
_P_T0 = _ptime.time()


def _pver(n):
    try:
        return _pmd.version(n)
    except Exception:                                              # noqa: BLE001
        return "-"


# the public cells pin another image than our own kernels' and install wheels: what our reader will run on
print("OURS_ENV " + " ".join(f"{_n}={_pver(_n)}" for _n in (
    "torch", "timm", "pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm", "numpy",
    "pandas")), flush=True)
_P_LEFT = 9 * 3600 - (_P_T0 - NB_T0) - __MARGIN_MIN__ * 60        # of the 9 h, less the margin for the blend
_O_DEADLINE = _P_T0 + _P_LEFT                                      # the stage's subprocesses are killed past it
with open(DATA / 'test.csv') as _f:
    _P_COMMIT = sum(1 for _ in _pcsv.DictReader(_f)) < 100
OURS_OK = False
if _P_LEFT < __MIN_LEFT_MIN__ * 60 and not _P_COMMIT:
    print(f"OURS_SKIPPED only {_P_LEFT / 60:.0f} min left", flush=True)
else:
    try:
        exec(compile(__OURS_SRC__, "ours_stage", "exec"), globals())
        OURS_OK = True
    except Exception as _e:                                        # noqa: BLE001
        if _P_COMMIT:
            raise
        print("OURS_FAILED", repr(_e)[:300], "-- the public blend stands", flush=True)
print(f"OURS_STAGE ok={OURS_OK} seconds {_ptime.time() - _P_T0:.0f} left_at_start_min {_P_LEFT / 60:.0f}", flush=True)
'''

BLEND = r'''
# ============================================================================ #
# The public blend + our folds: per-finding percentile ranks at weights fixed
# before any run, nothing fitted. A study our reader could not build keeps the
# public rank. The public file is replaced only once the mix has passed its checks.
# ============================================================================ #
import json as _qjson, os as _qos
from pathlib import Path as _QPath
import numpy as _qnp
import pandas as _qpd

_Q_W = __W_OURS__
_q_sub = _QPath("/kaggle/working/submission.csv")
if OURS_OK:
    try:
        _q_pub = _qpd.read_csv(_q_sub, dtype={"StudyInstanceUID": str})
        _q_lab = [c for c in _q_pub.columns if c != "StudyInstanceUID"]
        if set(_q_lab) != set(_Q_W):
            raise RuntimeError(f"public submission columns {_q_lab}")
        _q_ours = (_qpd.read_csv("/tmp/ortho_sub/ortho_preds.csv", dtype={"StudyInstanceUID": str})
                   .drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(_q_pub.StudyInstanceUID))
        _q_bad = set()
        for _p in _QPath("/tmp/ortho_sub").glob("preds_*.timing.json"):
            _q_bad |= set(_qjson.loads(_p.read_text()).get("errors", {}))
        _q_ok = (~_q_pub.StudyInstanceUID.isin(_q_bad).to_numpy()) & _q_ours[_q_lab].notna().all(axis=1).to_numpy()
        if _q_ok.mean() < 0.9:
            raise RuntimeError(f"our folds scored only {int(_q_ok.sum())} of {len(_q_pub)} studies")
        _q_out, _q_rho = _q_pub.copy(), {}
        for _c in _q_lab:
            _rp = _q_pub[_c].rank(method="average", pct=True).to_numpy(_qnp.float64)
            _ro = _qpd.Series(_q_ours[_c].to_numpy(_qnp.float64)[_q_ok]).rank(method="average", pct=True).to_numpy()
            _mix = _rp.copy()
            _mix[_q_ok] = (1.0 - _Q_W[_c]) * _rp[_q_ok] + _Q_W[_c] * _ro
            _q_out[_c] = _mix
            _q_rho[_c] = float(_qpd.Series(_rp[_q_ok]).corr(_qpd.Series(_ro), method="spearman"))
        _q_v = _q_out[_q_lab].to_numpy(_qnp.float64)
        if not (_qnp.isfinite(_q_v).all() and (_q_v >= 0).all() and (_q_v <= 1).all()):
            raise RuntimeError("the mix is not finite in [0, 1]")
        _q_mrho = float(_qnp.nanmean(list(_q_rho.values())))
        if len(_q_pub) >= 100 and not _q_mrho > 0.2:
            raise RuntimeError(f"ours and the public blend barely agree (mean rho {_q_mrho:.3f}): a misalignment")
        _q_pub.to_csv("/kaggle/working/public_blend_sub.csv", index=False)
        _q_out.to_csv("/kaggle/working/_public_plus_ours.csv", index=False)
        _qos.replace("/kaggle/working/_public_plus_ours.csv", _q_sub)
        _QPath("/kaggle/working/public_plus_ours_receipt.json").write_text(_qjson.dumps({
            "rule": "per-finding percentile ranks, weights fixed before any run", "w_ours": _Q_W, "arm": _O_ARM,
            "n_studies": len(_q_pub), "ours_unbuilt": int((~_q_ok).sum()), "spearman_ours_vs_public": _q_rho,
            "ours_minutes": _O_MIN, "built_from": "__BUILT_FROM__"}, indent=1))
        print(f"[public+ours] ARMS: the public blend as published + {_O_ARM}; w_ours "
              + ", ".join(f"{_c} {_Q_W[_c]:.2f}" for _c in _q_lab) + f"; {len(_q_pub)} studies, "
              f"{int((~_q_ok).sum())} on the public rank alone; mean Spearman(ours, public) {_q_mrho:.3f}", flush=True)
    except Exception as _e:                                        # noqa: BLE001
        if _P_COMMIT:
            raise
        print("OURS_BLEND_FAILED", repr(_e)[:300], "-- the public blend stands", flush=True)
else:
    print("[public+ours] our stage did not run to the end: the public blend stands unchanged", flush=True)
'''

INTRO = """# RSNA Knee -- the free public blend, with our five folds added

Cells 1-__N_PUB__ are `__PUBLIC__` as published (notebook sha256 `__PUB_SHA__`), unchanged: __PUB_DESC__.
The two cells after them are ours: __DESC__ as one more stage, then a per-finding rank blend at weights
fixed before any run (`W_OURS` in `notebooks/build_public_blend_kernel.py`). If our stage fails or would run late in
the scored rerun, the public submission is kept as it is. Built from `__BUILT_FROM__`.
"""


WAIT = "        rc = p.wait()\n"
WAIT_DEADLINE = ("        try:\n"
                 "            rc = p.wait(timeout=max(30.0, _O_DEADLINE - _otime.time()))\n"
                 "        except _osp.TimeoutExpired:\n"
                 "            for _q, _lf in procs.values():\n"
                 "                _q.kill()\n"
                 "            raise RuntimeError(f\"{what} ran past the stage deadline -- killed\")\n")


def with_deadline(ours: str) -> str:
    """build_submit_kernel's OURS cell waits on its subprocesses for ever; here they are killed past _O_DEADLINE."""
    if ours.count(WAIT) != 1:
        raise SystemExit("build_submit_kernel.OURS: _owait's wait moved -- the deadline patch needs re-reading")
    return ours.replace(WAIT, WAIT_DEADLINE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--public-dir", type=Path, required=True, help="`kaggle kernels pull <one of PUBLIC> -m`")
    ap.add_argument("--weights-kernel", default="", help="our training kernels, comma-joined")
    ap.add_argument("--weights-dataset", default="", help="and / or private datasets holding run dirs trained off "
                    "Kaggle (notebooks/offkaggle.py pack), comma-joined")
    ap.add_argument("--weights-sha", required=True, help="the members' last.pt sha256, comma-joined (an ensemble)")
    ap.add_argument("--tag", required=True, help="slug suffix (the whole slug must stay under 50 characters)")
    ap.add_argument("--desc", default="our ConvNeXt-small folds", help="what our member is, for the intro")
    ap.add_argument("--n-check", type=int, default=15, help="gate studies in all (at least 3 a member)")
    ap.add_argument("--margin-min", type=int, default=25, help="kept of the 9 h after our stage's deadline")
    ap.add_argument("--min-left-min", type=int, default=60, help="skip our stage with less than this left")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    shas = a.weights_sha.split(",")
    wkernels = [k for k in a.weights_kernel.split(",") if k]
    wdatasets = [d for d in a.weights_dataset.split(",") if d]
    if not wkernels and not wdatasets:
        raise SystemExit("pass --weights-kernel and / or --weights-dataset: where the members' last.pt are")
    if len(set(shas)) != len(shas) or not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in shas):
        raise SystemExit("--weights-sha: full 64-hex sha256 values of each last.pt, joined by commas")
    slug = f"rsna-knee-sub-{a.tag}"
    if len(slug) > 50 or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", a.tag):
        raise SystemExit(f"slug {slug} ({len(slug)} characters): Kaggle refuses more than 50; lowercase, digits, '-'")
    if set(W_OURS) != set(B.LABELS) or not all(0.0 <= w <= 0.5 for w in W_OURS.values()):
        raise SystemExit("W_OURS must name the twelve findings, each weight in [0, 0.5]")

    pmeta = json.loads((a.public_dir / "kernel-metadata.json").read_text())
    if pmeta["id"] not in PUBLIC:
        raise SystemExit(f"{a.public_dir} holds {pmeta['id']}, not one of {list(PUBLIC)}: read it, then add it")
    nb_path = a.public_dir / pmeta["code_file"]
    pub_sha = hashlib.sha256(nb_path.read_bytes()).hexdigest()
    pub = json.loads(nb_path.read_text())
    cells = pub["cells"]
    src = ["".join(c["source"]) for c in cells]
    code_cells = [s for c, s in zip(cells, src) if c["cell_type"] == "code"]
    # the seams this build relies on -- refuse an upstream version that moved any of them
    if sum(bool(re.search(r"^NB_T0 = time\.time\(\)$", s, re.M)) for s in code_cells) != 1:
        raise SystemExit("public notebook: no single cell starts the clock with NB_T0 = time.time() -- re-read it")
    if sum(bool(re.search(r"^DATA = find_mount\(COMPETITION, 'test\.csv'\)$", s, re.M)) for s in code_cells) != 1:
        raise SystemExit("public notebook: DATA is no longer the competition mount -- re-read it")
    last = code_cells[-1]
    if "sub.to_csv(WORK / 'submission.csv', index=False)" not in last or "print('NOTEBOOK_COMPLETE'" not in last:
        raise SystemExit("public notebook's last code cell is not its blend cell -- re-read it before building")
    if any(n in s for s in code_cells for n in ("_O_DEADLINE", "OURS_OK", "/tmp/ortho_sub")):
        raise SystemExit("public notebook already uses one of our names -- re-read it")

    B._check_import_closure(B.EMBED)
    files = {}
    for rel in B.EMBED:
        b = (B.PROJ / rel).read_bytes()
        files[rel] = [base64.b64encode(b).decode(), hashlib.sha256(b).hexdigest()]
    blob = base64.b64encode(json.dumps(files).encode()).decode()
    built_from = f"{B._git_head()}, public notebook sha256 {pub_sha[:16]}"

    ours = (with_deadline(B.OURS)
            .replace("__FILES_B64__", repr(blob)).replace("__BUILT_FROM__", built_from)
            .replace("__W_OURS__", "None").replace("__N_CHECK__", str(a.n_check))
            .replace("__N_WINDOWS__", "0").replace("__CHECK_ALWAYS__", "False")
            .replace("__WEIGHTS_SHAS__", repr(shas)).replace("__WEIGHTS_KERNEL__", ",".join(wkernels + wdatasets)))
    stage = (STAGE.replace("__MARGIN_MIN__", str(a.margin_min)).replace("__MIN_LEFT_MIN__", str(a.min_left_min))
             .replace("__OURS_SRC__", repr(ours)))
    blend = BLEND.replace("__W_OURS__", repr(W_OURS)).replace("__BUILT_FROM__", built_from)
    for k, c in enumerate([ours, stage, blend]):
        compile(c, f"ours{k}", "exec")                  # fail here, not after the queue wait

    def code(s):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": s.strip("\n").splitlines(keepends=True)}

    intro = (INTRO.replace("__N_PUB__", str(len(cells))).replace("__PUBLIC__", pmeta["id"]).replace("__PUB_DESC__", PUBLIC[pmeta["id"]])
             .replace("__PUB_SHA__", pub_sha[:16]).replace("__DESC__", a.desc).replace("__BUILT_FROM__", built_from))
    body = [dict(c, outputs=[], execution_count=None) if c["cell_type"] == "code" else c for c in cells]
    new = ([{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)}]
           + body + [code(stage), code(blend)])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{slug}.ipynb").write_text(json.dumps(dict(pub, cells=new), indent=1), encoding="utf-8")
    meta = {"id": f"{B.OWNER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": pmeta["dataset_sources"] + wdatasets, "kernel_sources": wkernels, "model_sources": [],
            "docker_image": pmeta["docker_image"], "machine_shape": "NvidiaTeslaT4"}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (slug + '.ipynb')}: {len(new)} cells ({len(cells)} public, unchanged; + intro, our stage, "
          f"the blend)")
    print(f"  members {len(shas)} | w_ours " + ", ".join(f"{k} {v:.2f}" for k, v in W_OURS.items()))
    print(f"  from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")
    print(f"push:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
