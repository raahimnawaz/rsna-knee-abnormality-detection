r"""Build the submission kernel that adds our folds to a public blend notebook, as one more stage (and KDC as another).

    kaggle kernels pull heliosli/this-is-new-version-0-9576-in-one-click -p <pub> -m      # any slug in PUBLIC
    python kaggle/build_public_blend_kernel.py --public-dir <pub> --weights-kernel <k1>,<k2>,<k3> \
        --weights-sha <64 hex>,<64 hex>,... --tag pub957-teach1 --out <dir>
    python -m kaggle kernels push -p <dir>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

import build_submit_kernel as B

PUBLIC = {
    "varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956":
        "the free 0.956 recipe: nartaa's OAI-trained CoAtNet (384 px, and its 224 px sibling at 35% of that slot), "
        "heliosli's R7 ConvNeXt-small (four folds), per-finding ranks 50/50",
    "heliosli/this-is-new-version-0-9576-in-one-click":
        "the free 0.957 recipe (10-14): nartaa's OAI-trained CoAtNet (384 px, and its 224 px sibling at 35% of that "
        "half), heliosli's R6 bundle (four folds) with its flip pass on, per-finding ranks 50/50. Its R6 dataset "
        "was overwritten with R7 on 10-14 18:12 (a kernel cannot pin a dataset version): pass --replace-dataset with "
        "a copy of version 1 under the same slug",
    "heliosli/this-one-is-old-version-0-957-new-one-you-can-use":
        "the free 0.957 recipe with R7 (10-14 18:13): nartaa's OAI-trained CoAtNet (384 px, and its 224 px sibling at "
        "35% of that half), heliosli's R7 bundle 8c7b666e (four folds, no flip pass), per-finding ranks 50/50",
}

W_OURS = {"ACL": 0.25, "MCL": 0.25, "Medial Meniscus": 0.25, "Medial OA": 0.25, "Effusion": 0.25, "Baker's": 0.25,
          "Contusion": 0.25, "Fracture": 0.25,
          "Lateral Meniscus": 0.10, "Lateral OA": 0.10, "PF OA": 0.10, "Synovitis": 0.10}

STAGE = r'''
import csv as _pcsv, time as _ptime, importlib.metadata as _pmd
_P_T0 = _ptime.time()


def _pver(n):
    try:
        return _pmd.version(n)
    except Exception:                                              # noqa: BLE001
        return "-"


print("OURS_ENV " + " ".join(f"{_n}={_pver(_n)}" for _n in (
    "torch", "timm", "pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm", "numpy",
    "pandas")), flush=True)
_P_READ = __READ__
_P_LEFT = 9 * 3600 - (_P_T0 - NB_T0) - __MARGIN_MIN__ * 60
_O_DEADLINE = _P_T0 + _P_LEFT
with open(DATA / 'test.csv') as _f:
    _P_COMMIT = sum(1 for _ in _pcsv.DictReader(_f)) < 100
OURS_OK = False
if _P_LEFT < __MIN_LEFT_MIN__ * 60 and not _P_COMMIT:
    if _P_READ:
        raise RuntimeError(f"READ: our stage would be skipped ({_P_LEFT / 60:.0f} min left)")
    print(f"OURS_SKIPPED only {_P_LEFT / 60:.0f} min left", flush=True)
else:
    try:
        exec(compile(__OURS_SRC__, "ours_stage", "exec"), globals())
        OURS_OK = True
    except Exception as _e:                                        # noqa: BLE001
        if _P_COMMIT or _P_READ:
            raise
        print("OURS_FAILED", repr(_e)[:300], "-- the public blend stands", flush=True)
print(f"OURS_STAGE ok={OURS_OK} seconds {_ptime.time() - _P_T0:.0f} left_at_start_min {_P_LEFT / 60:.0f}", flush=True)
'''

BLEND = r'''
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
        _q_ours = (_qpd.read_csv("/tmp/ours_sub/ours_preds.csv", dtype={"StudyInstanceUID": str})
                   .drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(_q_pub.StudyInstanceUID))
        _q_bad = set()
        for _p in _QPath("/tmp/ours_sub").glob("preds_*.timing.json"):
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
        if _P_COMMIT or _P_READ:
            raise
        print("OURS_BLEND_FAILED", repr(_e)[:300], "-- the public blend stands", flush=True)
else:
    if _P_READ:
        raise RuntimeError("READ: our stage did not run to the end; no public-only score under our name")
    print("[public+ours] our stage did not run to the end: the public blend stands unchanged", flush=True)
'''

STAGE_K = r'''
_K_T0 = _ptime.time()
_K_LEFT = 9 * 3600 - (_K_T0 - NB_T0) - __MARGIN_MIN__ * 60
_O_DEADLINE = _K_T0 + _K_LEFT
K_OK = False
if _K_LEFT < __MIN_LEFT_MIN__ * 60 and not _P_COMMIT:
    if _P_READ:
        raise RuntimeError(f"READ: KDC's stage would be skipped ({_K_LEFT / 60:.0f} min left)")
    print(f"K_SKIPPED only {_K_LEFT / 60:.0f} min left", flush=True)
else:
    try:
        exec(compile(__K_SRC__, "k_stage", "exec"), globals())
        K_OK = True
    except Exception as _e:                                        # noqa: BLE001
        if _P_COMMIT or _P_READ:
            raise
        print("K_FAILED", repr(_e)[:300], "-- the submission stands", flush=True)
print(f"K_STAGE ok={K_OK} seconds {_ptime.time() - _K_T0:.0f} left_at_start_min {_K_LEFT / 60:.0f}", flush=True)
'''

BLEND_K = r'''
import json as _kjson, os as _kos
from pathlib import Path as _KPath
import numpy as _knp
import pandas as _kpd

_K_MIX = __K_MIX__
_k_sub = _KPath("/kaggle/working/submission.csv")
if K_OK:
    try:
        _k_pre = _kpd.read_csv(_k_sub, dtype={"StudyInstanceUID": str})
        _k_lab = [c for c in _k_pre.columns if c != "StudyInstanceUID"]
        if not set(_K_MIX) <= set(_k_lab):
            raise RuntimeError(f"submission columns {_k_lab} lack {sorted(set(_K_MIX) - set(_k_lab))}")
        _k_pred = (_kpd.read_csv("/tmp/k_sub/ours_preds.csv", dtype={"StudyInstanceUID": str})
                   .drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(_k_pre.StudyInstanceUID))
        _k_bad = set()
        for _p in _KPath("/tmp/k_sub").glob("preds_*.timing.json"):
            _k_bad |= set(_kjson.loads(_p.read_text()).get("errors", {}))
        _k_ok = (~_k_pre.StudyInstanceUID.isin(_k_bad).to_numpy()) & _k_pred[list(_K_MIX)].notna().all(axis=1).to_numpy()
        if _k_ok.mean() < 0.9:
            raise RuntimeError(f"KDC scored only {int(_k_ok.sum())} of {len(_k_pre)} studies")
        _k_out, _k_rho = _k_pre.copy(), {}
        for _c, _w in _K_MIX.items():
            _rp = _k_pre[_c].rank(method="average", pct=True).to_numpy(_knp.float64)
            _rk = _kpd.Series(_k_pred[_c].to_numpy(_knp.float64)[_k_ok]).rank(method="average", pct=True).to_numpy()
            _mix = _rp.copy()
            _mix[_k_ok] = (1.0 - _w) * _rp[_k_ok] + _w * _rk
            _k_out[_c] = _mix
            _k_rho[_c] = float(_kpd.Series(_rp[_k_ok]).corr(_kpd.Series(_rk), method="spearman"))
        _k_v = _k_out[list(_K_MIX)].to_numpy(_knp.float64)
        if not (_knp.isfinite(_k_v).all() and (_k_v >= 0).all() and (_k_v <= 1).all()):
            raise RuntimeError("the KDC mix is not finite in [0, 1]")
        if len(_k_pre) >= 100 and not min(_k_rho.values()) > 0.2:
            raise RuntimeError(f"KDC and the submission barely agree ({_k_rho}): a misalignment")
        _k_pre.to_csv("/kaggle/working/before_kdc_sub.csv", index=False)
        _k_out.to_csv("/kaggle/working/_plus_kdc.csv", index=False)
        _kos.replace("/kaggle/working/_plus_kdc.csv", _k_sub)
        _KPath("/kaggle/working/plus_kdc_receipt.json").write_text(_kjson.dumps({
            "rule": "per-finding percentile ranks, KDC's shares fixed before any run", "k_mix": _K_MIX, "arm": _O_ARM,
            "n_studies": len(_k_pre), "kdc_unbuilt": int((~_k_ok).sum()), "spearman_kdc_vs_submission": _k_rho,
            "kdc_minutes": _O_MIN, "built_from": "__BUILT_FROM__"}, indent=1))
        print(f"[+KDC] ARMS: the submission above + {_O_ARM} on " + ", ".join(f"{_c} {_w:.2f}" for _c, _w in _K_MIX.items())
              + f" only; {len(_k_pre)} studies, {int((~_k_ok).sum())} keep their rank; Spearman(KDC, submission) "
              + ", ".join(f"{_c} {_r:.3f}" for _c, _r in _k_rho.items()), flush=True)
    except Exception as _e:                                        # noqa: BLE001
        if _P_COMMIT or _P_READ:
            raise
        print("K_BLEND_FAILED", repr(_e)[:300], "-- the submission stands", flush=True)
else:
    if _P_READ:
        raise RuntimeError("READ: KDC's stage did not run to the end")
    print("[+KDC] KDC's stage did not run to the end: the submission stands unchanged", flush=True)
'''

K_WRITE_LOOP_OLD = """for _rel, (_b, _sha) in _ojson.loads(_ob64.b64decode(__FILES_B64__).decode()).items():
    _p = _O_REPO / _rel
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_bytes(_ob64.b64decode(_b))
    if _ohash.sha256(_p.read_bytes()).hexdigest() != _sha:
        raise RuntimeError(f"{_rel}: embedded file does not match its hash")
_olog(f"built from __BUILT_FROM__; repo files written, hashes match")"""
K_CHECK_LOOP = """for _rel, _sha in __FILE_SHAS__.items():
    _p = _O_REPO / _rel
    if not _p.is_file() or _ohash.sha256(_p.read_bytes()).hexdigest() != _sha:
        raise RuntimeError(f"{_rel}: not written by our stage, or changed since")
_olog(f"built from __BUILT_FROM__; repo files from our stage, hashes match")"""
MAX_SOURCE = 1_000_000

INTRO = """# RSNA Knee -- the free public blend, with our five folds added

Cells 1-__N_PUB__ are `__PUBLIC__` as published (notebook sha256 `__PUB_SHA__`), unchanged except any line listed at the end: __PUB_DESC__.
The two cells after them are ours: __DESC__ as one more stage, then a per-finding rank blend at weights
fixed before any run (`W_OURS` in `kaggle/build_public_blend_kernel.py`). If our stage fails or would run late in
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


def replaced(sources: list[str], copies: list[str]) -> list[str]:
    """The public notebook's dataset sources, each one we hold a copy of (same slug, another owner) swapped for it."""
    out = list(sources)
    for copy in copies:
        if "=" in copy:
            old, copy = copy.split("=", 1)
            if old not in out or copy.count("/") != 1:
                raise SystemExit(f"--replace-dataset {old}={copy}: {old} is not one of {sources}")
            out[out.index(old)] = copy
            continue
        hits = [i for i, s in enumerate(out) if s.split("/")[-1] == copy.split("/")[-1]]
        if copy.count("/") != 1 or len(hits) != 1:
            raise SystemExit(f"--replace-dataset {copy}: owner/slug, its slug that of exactly one of {sources}")
        out[hits[0]] = copy
    return out


def _patched(src: str, patches: list[tuple[str, str]]) -> str:
    for old, new in patches:
        src = src.replace(old, new)
    return src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--public-dir", type=Path, required=True, help="`kaggle kernels pull <one of PUBLIC> -m`")
    ap.add_argument("--weights-kernel", default="", help="our training kernels, comma-joined")
    ap.add_argument("--weights-dataset", default="", help="and / or private datasets holding run dirs trained off "
                    "Kaggle (kaggle/offkaggle.py pack), comma-joined")
    ap.add_argument("--weights-sha", required=True, help="the members' last.pt sha256, comma-joined (an ensemble)")
    ap.add_argument("--tag", required=True, help="slug suffix (the whole slug must stay under 50 characters)")
    ap.add_argument("--desc", default="our ConvNeXt-small folds", help="what our member is, for the intro")
    ap.add_argument("--n-check", type=int, default=15, help="gate studies in all (at least 3 a member)")
    ap.add_argument("--margin-min", type=int, default=25, help="kept of the 9 h after our stage's deadline")
    ap.add_argument("--min-left-min", type=int, default=60, help="skip our stage with less than this left")
    ap.add_argument("--read", action="store_true", help="a leaderboard read: in the scored rerun a skipped, late or "
                    "failed stage of ours raises (no score) instead of keeping the public file. Finals leave it off")
    ap.add_argument("--replace-dataset", action="append", default=[], help="owner/slug of our copy, in place of "
                    "the public source with the same slug (a dataset its author has since overwritten); or "
                    "public/slug=ours/slug for a copy under another name; repeatable")
    ap.add_argument("--cell-patch", action="append", default=[], help="OLD=>NEW: one exact line change in the public "
                    "cells; OLD must occur exactly once across them; repeatable. Every patch is listed in the intro")
    ap.add_argument("--drop-cells", default="", help="comma-joined indices of public cells to drop (e.g. a member's "
                    "stages); a clock cell (NB_T0) is put in their place if one of them held it")
    ap.add_argument("--final-from", default="", help="replace the public blend cell: the submission is this file of "
                    "/kaggle/working as one stage wrote it (e.g. ours_sub.csv, the R-family's own predictions)")
    ap.add_argument("--drop-dataset", action="append", default=[], help="owner/slug not to mount (a dropped member's "
                    "weights); repeatable")
    ap.add_argument("--w-ours", default="", help="STRONG,WEAK: our share on the eight findings / on Lateral Meniscus, "
                    "Lateral OA, PF OA, Synovitis, in place of W_OURS (0.25,0.10)")
    ap.add_argument("--k-weights-kernel", default="", help="KDC: its training kernels, comma-joined (a second stage "
                    "and a second mix after ours; nothing changes without the --k-* flags)")
    ap.add_argument("--k-weights-sha", default="", help="KDC: the members' last.pt sha256, comma-joined (an ensemble)")
    ap.add_argument("--k-mix", action="append", default=[], help="KDC: FINDING=SHARE, the only findings KDC enters, "
                    "at a share of their rank in (0, 0.5]; repeatable")
    ap.add_argument("--k-desc", default="KDC, our knee-crop model on decoupled Lateral OA labels",
                    help="what KDC is, for the intro")
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
    w_ours = dict(W_OURS)
    if a.w_ours:
        strong, weak = (float(x) for x in a.w_ours.split(","))
        w_ours = {c: (weak if W_OURS[c] < 0.2 else strong) for c in W_OURS}
    if set(w_ours) != set(B.LABELS) or not all(0.0 <= w <= 0.5 for w in w_ours.values()):
        raise SystemExit("W_OURS must name the twelve findings, each weight in [0, 0.5]")
    k_shas = [h for h in a.k_weights_sha.split(",") if h]
    k_kernels = [k for k in a.k_weights_kernel.split(",") if k]
    k_mix = {}
    for spec in a.k_mix:
        name, sep, w = spec.rpartition("=")
        if not sep or name not in B.LABELS or name in k_mix or not 0.0 < float(w) <= 0.5:
            raise SystemExit(f"--k-mix {spec!r}: FINDING=SHARE, each of the twelve findings at most once, in (0, 0.5]")
        k_mix[name] = float(w)
    if not (bool(k_shas) == bool(k_kernels) == bool(k_mix)):
        raise SystemExit("--k-weights-kernel, --k-weights-sha and --k-mix go together")
    if k_shas and (len(set(k_shas)) != len(k_shas) or set(k_shas) & set(shas)
                   or not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in k_shas)):
        raise SystemExit("--k-weights-sha: full 64-hex sha256 values, each once, none of them one of ours")

    pmeta = json.loads((a.public_dir / "kernel-metadata.json").read_text())
    if pmeta["id"] not in PUBLIC:
        raise SystemExit(f"{a.public_dir} holds {pmeta['id']}, not one of {list(PUBLIC)}: read it, then add it")
    nb_path = a.public_dir / pmeta["code_file"]
    pub_sha = hashlib.sha256(nb_path.read_bytes()).hexdigest()
    pub = json.loads(nb_path.read_text())
    cells = pub["cells"]
    src = ["".join(c["source"]) for c in cells]
    code_cells = [s for c, s in zip(cells, src) if c["cell_type"] == "code"]
    if sum(bool(re.search(r"^NB_T0 = time\.time\(\)$", s, re.M)) for s in code_cells) != 1:
        raise SystemExit("public notebook: no single cell starts the clock with NB_T0 = time.time() -- re-read it")
    if sum(bool(re.search(r"^DATA = find_mount\(COMPETITION, 'test\.csv'\)$", s, re.M)) for s in code_cells) != 1:
        raise SystemExit("public notebook: DATA is no longer the competition mount -- re-read it")
    last = code_cells[-1]
    if "sub.to_csv(WORK / 'submission.csv', index=False)" not in last or "print('NOTEBOOK_COMPLETE'" not in last:
        raise SystemExit("public notebook's last code cell is not its blend cell -- re-read it before building")
    if any(n in s for s in code_cells for n in ("_O_DEADLINE", "OURS_OK", "/tmp/ours_sub")):
        raise SystemExit("public notebook already uses one of our names -- re-read it")
    if k_shas and any(re.search(r"\bK_OK\b|/tmp/k_sub|\b_K_(MIX|LEFT|T0)\b", s) for s in code_cells):
        raise SystemExit("public notebook already uses one of KDC's names -- re-read it")

    patches = []
    for spec in a.cell_patch:
        old, sep, new = spec.partition("=>")
        n = sum(s.count(old) for s in code_cells)
        if not sep or not old or n != 1:
            raise SystemExit(f"--cell-patch {spec[:80]!r}: OLD occurs {n} times in the public code cells (must be 1)")
        patches.append((old, new))
    cells = [dict(c, source=_patched("".join(c["source"]), patches).splitlines(keepends=True))
             if c["cell_type"] == "code" else c for c in cells]

    notes = []
    drop = sorted({int(x) for x in a.drop_cells.split(",") if x.strip()})
    if drop:
        if any(not 0 <= i < len(cells) - 1 for i in drop):
            raise SystemExit(f"--drop-cells {drop}: indices of public cells before the last one")
        clock = any(re.search(r"^NB_T0 = time\.time\(\)$", "".join(cells[i]["source"]), re.M) for i in drop)
        if any("DATA = find_mount(COMPETITION" in "".join(cells[i]["source"]) for i in drop):
            raise SystemExit("--drop-cells: a dropped cell defines DATA, which our stage needs")
        kept = [c for i, c in enumerate(cells) if i not in drop]
        if clock:
            kept.insert(drop[0], {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import time\n", "NB_T0 = time.time()   # [ours] the notebook clock, from a dropped cell\n"]})
        notes.append(f"dropped public cells {drop}" + (" (the clock NB_T0 kept in a cell of its own)" if clock else ""))
        cells = kept
    if a.final_from:
        last = max(i for i, c in enumerate(cells) if c["cell_type"] == "code")
        cells[last] = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# [ours] the public blend cell is replaced: the submission is one stage's own predictions\n",
            "import json as _fj, shutil as _fs\n", "from pathlib import Path as _FP\n",
            f"_fs.copyfile(_FP('/kaggle/working') / {a.final_from!r}, _FP('/kaggle/working') / 'submission.csv')\n",
            f"print('NOTEBOOK_COMPLETE', _fj.dumps({{'submission_from': {a.final_from!r}}}), flush=True)\n"]}
        notes.append(f"the public blend cell replaced: the submission is `{a.final_from}`")
    if a.drop_dataset:
        notes.append("not mounted: " + ", ".join(a.drop_dataset))

    B._check_import_closure(B.EMBED)
    files = {}
    for rel in B.EMBED:
        b = (B.PROJ / rel).read_bytes()
        files[rel] = [base64.b64encode(b).decode(), hashlib.sha256(b).hexdigest()]
    blob = base64.b64encode(json.dumps(files).encode()).decode()
    built_from = f"{B._git_head()}, public notebook sha256 {pub_sha[:16]}"

    ours = (with_deadline(B.OURS)
            .replace("__FILES_B64__", repr(blob)).replace("__BUILT_FROM__", built_from)
            .replace("__N_CHECK__", str(a.n_check))
            .replace("__N_WINDOWS__", "0").replace("__CHECK_ALWAYS__", "False")
            .replace("__WEIGHTS_SHAS__", repr(shas)).replace("__WEIGHTS_KERNEL__", ",".join(wkernels + wdatasets)))
    stage = (STAGE.replace("__MARGIN_MIN__", str(a.margin_min)).replace("__MIN_LEFT_MIN__", str(a.min_left_min))
             .replace("__READ__", repr(bool(a.read)))
             .replace("__OURS_SRC__", repr(ours)))
    blend = BLEND.replace("__W_OURS__", repr(w_ours)).replace("__BUILT_FROM__", built_from)
    for k, c in enumerate([ours, stage, blend]):
        compile(c, f"ours{k}", "exec")
    cells_k = []
    if k_shas:
        k_src = with_deadline(B.OURS)
        if k_src.count('_OPath("/tmp/ours_sub")') != 1:
            raise SystemExit("build_submit_kernel.OURS: its output folder moved -- re-read it before building KDC's stage")
        write_loop = K_WRITE_LOOP_OLD
        if k_src.count(write_loop) != 1 or k_src.count('_O_REPO = _O / "repo"') != 1:
            raise SystemExit("build_submit_kernel.OURS: its file-writing block moved -- re-read it before building KDC's stage")
        k_src = (k_src.replace('_OPath("/tmp/ours_sub")', '_OPath("/tmp/k_sub")')
                 .replace('_O_REPO = _O / "repo"', '_O_REPO = _OPath("/tmp/ours_sub") / "repo"   # [KDC] the files our stage wrote')
                 .replace(write_loop, K_CHECK_LOOP.replace("__FILE_SHAS__", repr({r: v[1] for r, v in files.items()})))
                 .replace("__BUILT_FROM__", built_from)
                 .replace("__N_CHECK__", str(a.n_check))
                 .replace("__N_WINDOWS__", "0").replace("__CHECK_ALWAYS__", "False")
                 .replace("__WEIGHTS_SHAS__", repr(k_shas)).replace("__WEIGHTS_KERNEL__", ",".join(k_kernels)))
        stage_k = (STAGE_K.replace("__MARGIN_MIN__", str(a.margin_min)).replace("__MIN_LEFT_MIN__", str(a.min_left_min))
                   .replace("__K_SRC__", repr(k_src)))
        blend_k = BLEND_K.replace("__K_MIX__", repr(k_mix)).replace("__BUILT_FROM__", built_from)
        for k, c in enumerate([k_src, stage_k, blend_k]):
            compile(c, f"kdc{k}", "exec")
        cells_k = [stage_k, blend_k]

    def code(s):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": s.strip("\n").splitlines(keepends=True)}

    if patches or notes:
        intro_patch = "\n\nChanged in the public notebook (nothing else):\n" + "".join(
            f"\n- `{o.strip()[:90]}` → `{n.strip()[:90]}`" for o, n in patches) + "".join(f"\n- {n}" for n in notes)
    else:
        intro_patch = ""
    intro = (INTRO.replace("__N_PUB__", str(len(cells))).replace("__PUBLIC__", pmeta["id"]).replace("__PUB_DESC__", PUBLIC[pmeta["id"]])
             .replace("__PUB_SHA__", pub_sha[:16]).replace("__DESC__", a.desc).replace("__BUILT_FROM__", built_from)) + intro_patch
    if k_shas:
        intro += (f"\n\nThen two more cells: {a.k_desc} ({len(k_shas)} member(s)) as a stage of its own, with its own gate "
                  "and deadline, mixed into the submission above on "
                  + ", ".join(f"{c} at {w:.2f}" for c, w in k_mix.items())
                  + " only (shares fixed before any run). If it fails or would run late, the submission above stands.\n")
    body = [dict(c, outputs=[], execution_count=None) if c["cell_type"] == "code" else c for c in cells]
    new = ([{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)}]
           + body + [code(stage), code(blend)] + [code(c) for c in cells_k])
    a.out.mkdir(parents=True, exist_ok=True)
    nb_text = json.dumps(dict(pub, cells=new), indent=1)
    if len(nb_text.encode("utf-8")) >= MAX_SOURCE:
        raise SystemExit(f"the notebook is {len(nb_text.encode('utf-8')):,} bytes: Kaggle refuses a source of 1 MB or more")
    (a.out / f"{slug}.ipynb").write_text(nb_text, encoding="utf-8")
    meta = {"id": f"{B.OWNER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": [d for d in replaced(pmeta["dataset_sources"], a.replace_dataset)
                                if d not in a.drop_dataset] + wdatasets, "kernel_sources": wkernels + k_kernels, "model_sources": [],
            "docker_image": pmeta["docker_image"], "machine_shape": "NvidiaTeslaT4"}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (slug + '.ipynb')}: {len(new)} cells ({len(cells)} public, unchanged; + intro, our stage, "
          f"the blend)")
    print(f"  members {len(shas)} | w_ours " + ", ".join(f"{k} {v:.2f}" for k, v in w_ours.items()))
    if k_shas:
        print(f"  + KDC: {len(k_shas)} member(s) from {', '.join(k_kernels)} | k_mix "
              + ", ".join(f"{c} {w:.2f}" for c, w in k_mix.items()) + " | two more cells")
    for n in notes:
        print(f"  {n}")
    print(f"  from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ embedded files differ from HEAD -- commit first so the run is reproducible")
    print(f"push:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
