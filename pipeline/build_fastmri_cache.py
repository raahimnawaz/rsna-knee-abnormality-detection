"""fastMRI+ knee volumes -> a Stage A shard (`c6-64slice`), so `train_windows` reads NYU knees unchanged.

    .venv/bin/python pipeline/build_fastmri_cache.py --h5 data/external/fastmri/h5 \
        --plus data/external/fastmri/fastmri_plus --out data/external/fastmri/cache_stage_a

WHAT IS REUSED, NOT REWRITTEN: slot assignment (`slots64.assign_series`), slice picking over the 2-98% span
(`slots64.pick_depths`, `SPAN`), the per-series 2-98th percentile window (`PCT`) and the 140 mm physical crop
(`build_slots64.crop_mm`) -- the exact functions that built the competition cache. Only the READER differs: an h5
`reconstruction_rss` stack instead of a DICOM series.

WHAT THE FILES ARE, READ OFF THEM 2026-09-27 (RUNBOOK, "HOW fastMRI+ FITS"): coronal only; PD (`CORPD_FBK`) and PD
fat-sat (`CORPDFS_FBK`) as SEPARATE files; 320 x 320 over 140 x 140 mm (read per file from the ISMRMRD header, not
assumed); a hashed `patient_id` per file. A study here is one patient: the fat-sat file fills `cor_1` (the slot
that prefers a fluid-sensitive series), the other `cor_2`, by the same `assign_series` rule as the competition.

⛔ ORIENTATION, MEASURED 2026-09-27: the stored images are UPSIDE DOWN relative to the competition cache. On the most
anterior coronal slice the tissue (kneecap) centre sits at 0.30 of the height in the competition cache, 0.67 in NYU
as stored and 0.33 flipped (60 files); and fastMRI+'s boxes land on anatomy only on the flipped image (effusion on
the suprapatellar fluid, a meniscus tear on the joint line). So every slice is `np.flipud`-ed, which also puts the
boxes in the image's frame. Slice order needs nothing: slice 0 is anterior in both.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slots64 as S                                                    # noqa: E402
from build_slots64 import PCT, SPAN, crop_mm, render_edges            # noqa: E402

FLUID = {"CORPDFS_FBK": 1, "CORPD_FBK": 0}


def header_mm(h) -> tuple[float, float]:
    """(row_mm, col_mm) of `reconstruction_rss` from the file's own ISMRMRD header: FOV / matrix, reconSpace."""
    x = h["ismrmrd_header"][()].decode()
    blk = re.search(r"<reconSpace>(.*?)</reconSpace>", x, re.S).group(1)
    fov = {k: float(re.search(rf"<fieldOfView_mm>.*?<{k}>(.*?)</{k}>", blk, re.S).group(1)) for k in "xy"}
    mat = {k: int(re.search(rf"<matrixSize>.*?<{k}>(.*?)</{k}>", blk, re.S).group(1)) for k in "xy"}
    return fov["y"] / mat["y"], fov["x"] / mat["x"]


def read_series(path: Path) -> dict:
    import h5py
    with h5py.File(path, "r") as h:
        acq = str(h.attrs["acquisition"])
        row_mm, col_mm = header_mm(h)
        n, hh, ww = h["reconstruction_rss"].shape
        return {"SeriesInstanceUID": path.stem, "plane": "Coronal", "fluid": FLUID.get(acq, 0), "acquisition": acq,
                "n_slices": n, "row_mm": row_mm, "col_mm": col_mm, "shape": (hh, ww),
                "patient_id": str(h.attrs["patient_id"]), "path": str(path)}


def build_study(series: list[dict], span=SPAN) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """One patient's h5 series -> (vol (64,336,336) uint8, mask (64,) uint8, one meta row per slot)."""
    import h5py
    pick = S.assign_series(series)
    vol = np.zeros((S.N_SLICES, S.IMG, S.IMG), np.uint8)
    mask = np.zeros(S.N_SLICES, np.uint8)
    meta = []
    for (a, b), slot in zip(S.slot_offsets(), S.SLOTS):
        ser = pick.get(slot.name)
        row = {"slot": slot.name, "series": None, "n_available": 0, "source_step": 0.0, "ordered_by": None,
               "row_mm": np.nan, "col_mm": np.nan, "picks": "", "decode_fail": 0}
        if ser is None:
            meta.append(row)
            continue
        n = ser["n_slices"]
        picks = S.pick_depths(n, slot.n_slices, *span)
        with h5py.File(ser["path"], "r") as h:
            stack = h["reconstruction_rss"]
            arrs = {p: np.flipud(np.asarray(stack[p], np.float32)) for p in sorted(set(picks.tolist()))}
        vals = np.concatenate([x.ravel() for x in arrs.values()])
        lo, hi = np.percentile(vals, PCT)
        for j, p in enumerate(picks):
            w = np.clip((arrs[p] - lo) / (hi - lo + 1e-6), 0, 1)
            vol[a + j] = np.round(crop_mm(w, ser["row_mm"], ser["col_mm"], img=S.IMG) * 255).astype(np.uint8)
            mask[a + j] = 1
        row.update(series=ser["SeriesInstanceUID"], fluid=ser["fluid"], n_available=n,
                   source_step=S.source_step(n, slot.n_slices, *span), ordered_by="h5 slice index (0 = anterior)",
                   row_mm=ser["row_mm"], col_mm=ser["col_mm"], picks=" ".join(map(str, picks.tolist())))
        meta.append(row)
    return vol, mask, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", type=Path, required=True, help="dir holding the unpacked singlecoil_{train,val}/*.h5")
    ap.add_argument("--plus", type=Path, required=True, help="fastMRI+ dir with knee_file_list.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N patients only (smoke)")
    a = ap.parse_args()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)

    listed = set(pd.read_csv(a.plus / "knee_file_list.csv", header=None)[0].astype(str).str.strip())
    files = {p.stem: p for p in sorted(a.h5.rglob("*.h5"))}
    use = sorted(listed & set(files))
    print(f"fastMRI+ lists {len(listed)} files; {len(files)} h5 on disk; converting {len(use)}", flush=True)
    series = [read_series(files[f]) for f in use]
    odd = [s["SeriesInstanceUID"] for s in series if s["acquisition"] not in FLUID]
    if odd:
        print(f"  {len(odd)} files with an unexpected acquisition (treated as not fluid-sensitive): {odd[:5]}")
    by_pt: dict[str, list[dict]] = {}
    for s in series:
        by_pt.setdefault(s["patient_id"], []).append(s)
    pts = sorted(by_pt)[: a.limit] if a.limit else sorted(by_pt)
    ids = [f"nyu_{p[:24]}" for p in pts]
    if len(set(ids)) != len(ids):
        raise SystemExit("patient-id prefix collision -- lengthen the prefix")
    print(f"{len(by_pt)} patients ({sum(len(v) == 2 for v in by_pt.values())} with two files); writing {len(pts)}",
          flush=True)

    vols = np.lib.format.open_memmap(a.out / "vols.npy", mode="w+", dtype=np.uint8,
                                     shape=(len(pts), S.N_SLICES, S.IMG, S.IMG))
    masks = np.zeros((len(pts), S.N_SLICES), np.uint8)
    rows = []
    for i, (p, uid) in enumerate(zip(pts, ids)):
        vols[i], masks[i], meta = build_study(by_pt[p])
        rows += [{"StudyInstanceUID": uid, **m} for m in meta]
        if i % 200 == 0 or i == len(pts) - 1:
            print(f"  {i + 1}/{len(pts)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    vols.flush()
    np.save(a.out / "masks.npy", masks)
    np.save(a.out / "ids.npy", np.asarray(ids, dtype=object), allow_pickle=True)
    pd.DataFrame(rows).to_csv(a.out / "slots.csv", index=False)
    pd.DataFrame([{"StudyInstanceUID": u, "patient_id": p, "files": " ".join(s["SeriesInstanceUID"] for s in by_pt[p])}
                  for p, u in zip(pts, ids)]).to_csv(a.out / "studies.csv", index=False)
    render_edges(vols, masks, ids, a.out)

    fill = {s.name: float(masks[:, x:y].any(1).mean()) for (x, y), s in zip(S.slot_offsets(), S.SLOTS)}
    edge = {}
    for (x, y), s in zip(S.slot_offsets(), S.SLOTS):
        ok = np.flatnonzero(masks[:, x])[:300]
        if len(ok):
            t = lambda k: float((np.asarray(vols[ok, k]) > 25).mean())   # noqa: E731
            edge[s.name] = {"first": t(x), "mid": t((x + y) // 2), "last": t(y - 1), "n": int(len(ok))}
    man = {**S.manifest(tuple(SPAN)), "canon": False, "split": "fastmri_plus", "shard": 0, "n_shards": 1,
           "n_studies": len(pts), "source": "fastMRI knee singlecoil train+val, reconstruction_rss, np.flipud",
           "slot_fill": fill, "edge_tissue": edge, "crop": "physical, padded (not clamped)",
           "window_pct": list(PCT), "wall_min": (time.time() - t0) / 60}
    (a.out / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
    print(f"slot fill {json.dumps({k: round(v, 3) for k, v in fill.items()})}")
    print("edge tissue (first/mid/last) " + " | ".join(
        f"{k} {v['first']:.2f}/{v['mid']:.2f}/{v['last']:.2f}" for k, v in edge.items()))
    print(f"done {len(pts)} studies in {man['wall_min']:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
