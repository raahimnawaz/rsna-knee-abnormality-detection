"""Stage 0 — measure the data before anything is designed on it. `PLAN-C6.md` Stage 0.

    python notebooks/build_stage0_kernel.py --out /tmp/s0 [--limit 200]
    python -m kaggle kernels push -p /tmp/s0

**CPU ONLY. No GPU, therefore no lottery, therefore no quota.** Headers only -- not one pixel is
decoded. Minutes, not a session.

WHAT THIS SETTLES, AND WHY EACH IS CURRENTLY AN ASSUMPTION.

  0.1  ANISOTROPY. r = through-plane sampling / in-plane pixel size, per plane. This is the fact
       that decides whether a 3D arm is admissible at all, and PLAN-C6 fixes its threshold
       (r >= 4 excludes isotropic 3D) BEFORE this number exists. FINDINGS.md 6.4 confirms
       PixelSpacing/SliceThickness/InstanceNumber/Rows/Columns are present 200/200, so this is a
       header pass -- but the ratio itself was never recorded anywhere.

       ⛔ SliceThickness is NOT the quantity that matters. It is the slab excited; the SAMPLING
       interval is thickness + inter-slice gap, and that is what an isotropic resampler has to
       interpolate across. A 3 mm slice with a 1 mm gap samples every 4 mm. So this measures
       BOTH, plus the true delta from ImagePositionPatient, and reports where they disagree.

  0.2  STACK DEPTH. Slices per series, per plane. `slots64.py` asks for 18 sagittal slices across
       6-94%; if the median sagittal stack is 20 that is nearly the whole stack and the spec is
       far cheaper than assumed. If it is 60, the 6-94% window is a real choice.

  0.3  PROTOCOL COMPOSITION. Series per study by (plane x fluid), from the competition's OWN
       `Anatomical_Plane` / `Fluid_Sensitive` columns -- slot_cache.py:175 is explicit that the
       assignment is the CSV's and never a rule, so no classifier is written here. `slots64.py`
       assumes two usable sagittals and two coronals per study. THAT IS UNTESTED and this is the
       table that tests it.

  ⛔ NO SLOT FILL IS COMPUTED HERE, DELIBERATELY. Stage 0 may overturn `slots64.py`, so baking
  that spec into the measurement would make the measurement agree with the thing it is checking.
  This emits raw per-series facts; the spec is derived from them afterwards, locally, for free.

  ⛔ NO AUC IS PRODUCED. Nothing here ranks a model (3b, [[never-select-on-a-tiny-set]]).

TWO PASSES, because reading every header is too slow and reading one is not enough.
  Pass 1, ALL series: one header each. PixelSpacing/SliceThickness/SpacingBetweenSlices/Rows/
       Columns are constant within a series, so one file settles them. Plus the file count.
  Pass 2, a SAMPLE of series: every header, sorted by projecting ImagePositionPatient onto the
       slice normal, to measure the TRUE spacing and check it against the tag. Their existing
       geometry work used n=396; this defaults to 400.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "raahimnawaz"
SLUG = "rsna-knee-c6-stage0"

WORK_SRC = r'''
import json, os, sys, time, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

t0 = time.time()
LIMIT = int(os.environ.get("LIMIT", "0"))          # studies; 0 = all
SAMPLE = int(os.environ.get("SAMPLE", "400"))      # series for the full-header pass
# Overridable ONLY so this worker can be smoke-tested off Kaggle against a synthetic tree.
# Defaults are the real mounts, so behaviour on Kaggle is unchanged.
IN_DIR = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


# --- locate the competition, without assuming the mount name -------------------------------
def find_root():
    base = IN_DIR
    cands = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    for p in cands:
        if (p / "train_series.csv").exists():
            return p
    # Fall back to any directory holding a train.csv, then report what was seen. A silent wrong
    # root is the failure mode here: it would produce a clean-looking table about nothing.
    for p in cands:
        if (p / "train.csv").exists():
            return p
    sys.exit(f"no competition root under {IN_DIR}; saw {[p.name for p in cands]}")


ROOT = find_root()
log(f"root: {ROOT}")

series_csv = ROOT / "train_series.csv"
if not series_csv.exists():
    sys.exit(f"train_series.csv not under {ROOT}")
ts = pd.read_csv(series_csv)
log(f"train_series.csv {ts.shape} cols={list(ts.columns)}")

# The competition's own plane/fluid assignment. slot_cache.py:175 -- never a rule of ours.
PLANE_COL = "Anatomical_Plane" if "Anatomical_Plane" in ts.columns else None
FLUID_COL = "Fluid_Sensitive" if "Fluid_Sensitive" in ts.columns else None
if PLANE_COL is None:
    log("!! Anatomical_Plane absent -- 0.3 will be blank; do not infer a plane here")

# --- find the image tree ---------------------------------------------------------------------
img_root = None
for cand in ("train_images", "train", "images"):
    if (ROOT / cand).is_dir():
        img_root = ROOT / cand
        break
if img_root is None:
    dirs = [p for p in ROOT.iterdir() if p.is_dir()]
    img_root = max(dirs, key=lambda p: sum(1 for _ in p.iterdir()), default=None)
if img_root is None:
    sys.exit("no image directory found")
log(f"images: {img_root}")

studies = sorted(p for p in img_root.iterdir() if p.is_dir())
if LIMIT:
    studies = studies[:LIMIT]
log(f"studies: {len(studies)}")

meta = {}
if PLANE_COL:
    for r in ts.itertuples(index=False):
        meta[getattr(r, "SeriesInstanceUID")] = (
            getattr(r, PLANE_COL, None),
            getattr(r, FLUID_COL, None) if FLUID_COL else None)


def num(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def first_of(ds, *names):
    for n in names:
        if n in ds and ds[n].value not in (None, ""):
            return ds[n].value
    return None


# --- PASS 1: one header per series ------------------------------------------------------------
rows, n_series = [], 0
for i, sd in enumerate(studies):
    for sed in sorted(p for p in sd.iterdir() if p.is_dir()):
        files = [f for f in sed.iterdir() if f.is_file()]
        if not files:
            continue
        n_series += 1
        try:
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
        except Exception as e:
            rows.append({"study": sd.name, "series": sed.name, "n_files": len(files),
                         "error": type(e).__name__})
            continue
        ps = first_of(ds, "PixelSpacing")
        inplane = num(ps[0]) if ps is not None and len(ps) else np.nan
        plane, fluid = meta.get(sed.name, (None, None))
        rows.append({
            "study": sd.name, "series": sed.name, "n_files": len(files),
            "plane": plane, "fluid": fluid,
            "in_plane_mm": inplane,
            "slice_thickness_mm": num(first_of(ds, "SliceThickness")),
            "spacing_between_mm": num(first_of(ds, "SpacingBetweenSlices")),
            "rows": num(first_of(ds, "Rows")), "cols": num(first_of(ds, "Columns")),
            "series_desc": str(first_of(ds, "SeriesDescription") or ""),
            "error": "",
        })
    if (i + 1) % 400 == 0:
        log(f"  pass1 {i+1}/{len(studies)} studies, {n_series} series")

df = pd.DataFrame(rows)
df.to_csv(OUT / "stage0_series.csv", index=False)
log(f"pass1 done: {len(df)} series -> stage0_series.csv")

# --- PASS 2: true through-plane spacing on a sample -------------------------------------------
# SpacingBetweenSlices is optional in DICOM and is frequently absent or wrong. The sampling
# interval that an isotropic resampler must interpolate across is the distance between adjacent
# slice POSITIONS, so measure that directly on a sample rather than trusting the tag everywhere.
random.seed(20260907)
ok = df[(df.error == "") & (df.n_files >= 3)]
pick = ok.sample(min(SAMPLE, len(ok)), random_state=20260907) if len(ok) else ok
log(f"pass2 sample: {len(pick)} series")

true_rows = []
for j, r in enumerate(pick.itertuples(index=False)):
    sed = img_root / r.study / r.series
    pos, normal = [], None
    for f in sorted(p for p in sed.iterdir() if p.is_file()):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        ipp = first_of(ds, "ImagePositionPatient")
        iop = first_of(ds, "ImageOrientationPatient")
        if ipp is None or iop is None:
            continue
        if normal is None:
            a = np.array([float(x) for x in iop[:3]])
            b = np.array([float(x) for x in iop[3:6]])
            normal = np.cross(a, b)
            n = np.linalg.norm(normal)
            if n == 0:
                break
            normal = normal / n
        pos.append(float(np.dot(np.array([float(x) for x in ipp]), normal)))
    if len(pos) < 3:
        continue
    d = np.diff(np.sort(np.asarray(pos)))
    d = d[d > 1e-6]
    if not len(d):
        continue
    true_rows.append({"study": r.study, "series": r.series, "plane": r.plane,
                      "n_slices": len(pos),
                      "true_spacing_mm": float(np.median(d)),
                      "spacing_cv": float(np.std(d) / (np.mean(d) + 1e-9)),
                      "extent_mm": float(np.ptp(pos)),
                      "tag_thickness_mm": r.slice_thickness_mm,
                      "tag_spacing_mm": r.spacing_between_mm,
                      "in_plane_mm": r.in_plane_mm})
    if (j + 1) % 100 == 0:
        log(f"  pass2 {j+1}/{len(pick)}")

tdf = pd.DataFrame(true_rows)
tdf.to_csv(OUT / "stage0_spacing.csv", index=False)
log(f"pass2 done: {len(tdf)} series -> stage0_spacing.csv")


# --- aggregate --------------------------------------------------------------------------------
def q(s, p):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(np.percentile(s, p)) if len(s) else None


summary = {"n_studies": len(studies), "n_series": int(len(df)),
           "n_errors": int((df.error != "").sum()), "sample_n": int(len(tdf))}

# 0.1 anisotropy, per plane, from the MEASURED spacing
aniso = {}
if len(tdf):
    tdf = tdf.assign(r=tdf.true_spacing_mm / tdf.in_plane_mm.replace(0, np.nan))
    for plane, g in tdf.groupby(tdf.plane.fillna("UNKNOWN")):
        aniso[str(plane)] = {
            "n": int(len(g)),
            "in_plane_mm_median": q(g.in_plane_mm, 50),
            "true_spacing_mm_median": q(g.true_spacing_mm, 50),
            "tag_thickness_mm_median": q(g.tag_thickness_mm, 50),
            "tag_spacing_mm_median": q(g.tag_spacing_mm, 50),
            "r_median": q(g.r, 50), "r_p10": q(g.r, 10), "r_p90": q(g.r, 90),
        }
    summary["r_median_overall"] = q(tdf.r, 50)
summary["anisotropy_by_plane"] = aniso

# 0.2 stack depth per plane
depth = {}
for plane, g in df.groupby(df.plane.fillna("UNKNOWN")):
    depth[str(plane)] = {"n_series": int(len(g)), "p10": q(g.n_files, 10),
                         "median": q(g.n_files, 50), "p90": q(g.n_files, 90),
                         "max": q(g.n_files, 100)}
summary["stack_depth_by_plane"] = depth

# 0.3 composition: series per study by (plane, fluid)
if PLANE_COL:
    per = df.groupby(["study", "plane", "fluid"], dropna=False).size().rename("n").reset_index()
    wide = per.pivot_table(index="study", columns=["plane", "fluid"], values="n",
                           aggfunc="sum", fill_value=0)
    comp = {}
    for col in wide.columns:
        v = wide[col]
        comp[f"{col[0]}|fluid={col[1]}"] = {
            "studies_with_ge1": int((v >= 1).sum()),
            "studies_with_ge2": int((v >= 2).sum()),
            "pct_ge1": round(100.0 * (v >= 1).mean(), 2)}
    summary["composition"] = comp
    # The specific claim slots64.py makes, stated so it can be read as pass/fail.
    sag = df[df.plane.astype(str).str.startswith("Sag")].groupby("study").size()
    cor = df[df.plane.astype(str).str.startswith("Cor")].groupby("study").size()
    ns = len(set(df.study))
    summary["slots64_assumption"] = {
        "studies": ns,
        "pct_with_2plus_sagittal": round(100.0 * (sag >= 2).sum() / max(ns, 1), 2),
        "pct_with_2plus_coronal": round(100.0 * (cor >= 2).sum() / max(ns, 1), 2),
        "pct_with_1plus_sagittal": round(100.0 * (sag >= 1).sum() / max(ns, 1), 2),
        "pct_with_1plus_coronal": round(100.0 * (cor >= 1).sum() / max(ns, 1), 2)}

(OUT / "stage0_summary.json").write_text(json.dumps(summary, indent=2))
print("\n" + json.dumps(summary, indent=2), flush=True)

# The pre-registered rule, applied. PLAN-C6 fixed the threshold before this number existed.
r = summary.get("r_median_overall")
if r is not None:
    verdict = ("r >= 4: isotropic 3D CONTRAINDICATED -- anisotropic kernels only (PLAN-C6 §2)"
               if r >= 4 else
               "r < 4: an isotropic 3D arm is admissible; PLAN-C6 §2's exclusion is void")
    log(f"0.1 VERDICT  r_median = {r:.2f}  ->  {verdict}")
log("stage 0 complete")
'''

PARENT = r'''
import os, subprocess, sys, time
t0 = time.time()
os.environ.setdefault("LIMIT", "__LIMIT__")
os.environ.setdefault("SAMPLE", "__SAMPLE__")
open("/kaggle/working/work.py", "w").write(WORK)
# Run as a subprocess so output flushes even if the worker dies, and so a non-zero exit is
# visible as a failed kernel rather than a notebook that "completed" with a traceback in a cell.
rc = subprocess.call([sys.executable, "-u", "/kaggle/working/work.py"])
print(f"[{time.time()-t0:7.1f}s] exited {rc}", flush=True)
sys.exit(rc)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="studies; 0 = all. USE 20 TO SMOKE.")
    ap.add_argument("--sample", type=int, default=400, help="series for the full-header pass")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    parent = PARENT.replace("__LIMIT__", str(a.limit)).replace("__SAMPLE__", str(a.sample))
    src = "WORK = " + json.dumps(WORK_SRC) + "\n\n" + parent
    compile(WORK_SRC, "work", "exec")     # fail here, not after the queue wait
    compile(src, "cell", "exec")

    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": src.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps(
        {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb",
         "language": "python", "kernel_type": "notebook", "is_private": True,
         # CPU ONLY -- this is the whole point. No GPU means no lottery and no quota.
         "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
         "keywords": [], "dataset_sources": [],
         "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"],
         "model_sources": []}, indent=2))
    print(f"built {out}/{SLUG}.ipynb   limit={a.limit or 'all'}  sample={a.sample}")
    print(f"push:  python -m kaggle kernels push -p {out}")


if __name__ == "__main__":
    main()
