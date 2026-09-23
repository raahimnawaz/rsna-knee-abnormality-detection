"""Stage A: build the 64-slice C6 cache from the competition DICOMs. `slots64.py` is the spec.

    python pipeline/build_slots64.py --root <competition dir> --out <dir> --shard 0 --n-shards 4
    python pipeline/build_slots64.py --root /tmp/safix/input/... --out /tmp/safix/out --check  # fixture

WHAT A STUDY BECOMES. One (64, 336, 336) uint8 stack plus a (64,) mask, five slots in fixed order
(`slots64.SLOTS`: sag 18 / sag 14 / cor 12 / cor 8 / ax 12). Per slot, one series is chosen by
`slots64.assign_series` (plane is never substituted; fluid preference is a sort key); its slices
are ordered by position along the slice normal; `n_want` of them are taken evenly across
`--span` of the stack; each is intensity-windowed, cropped to a physical 140 mm box and resized.
A slot with no series stays zeros with its mask cleared. This is the frontier's geometry
(`MODEL.md` §0) built here, from DICOMs, with three things done differently on purpose:

  * Span 2-98% by default: their 0.928 corpus ("maxspan"). Their 64-slice 6-94% model read 0.924.
  * The crop is PHYSICAL both ways. They clamp the box to the image when the field of view is
    under 140 mm, which silently changes mm-per-pixel for exactly those scans; here the box is
    padded instead, so every slice is 140 mm across (6.2% of series are under 140 mm, `DATA.md`
    §10), and anisotropic PixelSpacing is honoured (0 of 20,792 series have any, so that one is moot).
  * Upsampling uses bilinear, not INTER_AREA (which is nearest-neighbour when enlarging).

Everything else follows their reader: IPP projected on IOP's normal for order (InstanceNumber only
when geometry is absent, and that is counted), modality LUT, MONOCHROME1 inverted, and a 2-98th
percentile window over the slices taken from each series.

OUTPUT per shard: `vols.npy` (n, 64, 336, 336) uint8, `masks.npy` (n, 64), `ids.npy`, `slots.csv`
(one row per study x slot: series, n_available, source_step, spacing, how it was ordered, decode
failures), `manifest.json`, and for GATE A `render_edges.png` -- the first and last slice of every
slot for 20 studies, to confirm the span's ends are anatomy and not wrap or air.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slots64 as S  # noqa: E402

SPAN = (0.02, 0.98)
PCT = (2.0, 98.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- reading
def order_series(series_dir: Path) -> tuple[list[tuple[Path, float, float]], str]:
    """-> ([(file, row_mm, col_mm)] sorted along the slice normal, how it was ordered)."""
    import pydicom
    recs, by_geom = [], True
    for f in sorted(series_dir.iterdir()):
        if f.name.startswith("."):
            continue
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:                                   # noqa: BLE001 -- unreadable header
            continue
        iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
        ps = getattr(h, "PixelSpacing", None)
        row_mm, col_mm = (float(ps[0]), float(ps[1])) if ps is not None and len(ps) == 2 else (np.nan, np.nan)
        if iop is not None and ipp is not None and len(iop) == 6:
            n = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
            key = float(np.dot(np.asarray(ipp, float), n))
        else:
            by_geom = False
            key = float(getattr(h, "InstanceNumber", 0) or 0)
        recs.append((key, f, row_mm, col_mm))
    recs.sort(key=lambda r: r[0])
    return [(f, r, c) for _, f, r, c in recs], ("normal" if by_geom else "instance_number")


def decode(path: Path) -> np.ndarray:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    d = pydicom.dcmread(path)
    a = d.pixel_array
    if a.ndim == 3 and a.shape[-1] in (3, 4):             # RGB-encoded greyscale
        a = a[..., :3].mean(-1)
    if a.ndim != 2:
        raise ValueError(f"pixel array shape {a.shape}")
    a = apply_modality_lut(a, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def crop_mm(a: np.ndarray, row_mm: float, col_mm: float, fov_mm: float = S.FOV_MM,
            img: int = S.IMG) -> np.ndarray:
    """Centre box of `fov_mm` x `fov_mm` in PHYSICAL units, padded with 0 where the scan is smaller."""
    import cv2
    h, w = a.shape
    ph, pw = int(round(fov_mm / row_mm)), int(round(fov_mm / col_mm))
    out = np.zeros((ph, pw), np.float32)
    sy, sx = max((h - ph) // 2, 0), max((w - pw) // 2, 0)          # source offsets
    dy, dx = max((ph - h) // 2, 0), max((pw - w) // 2, 0)          # destination offsets
    ch, cw = min(h, ph), min(w, pw)
    out[dy:dy + ch, dx:dx + cw] = a[sy:sy + ch, sx:sx + cw]
    interp = cv2.INTER_AREA if (ph >= img and pw >= img) else cv2.INTER_LINEAR
    return cv2.resize(out, (img, img), interpolation=interp)


def build_study(study_dir: Path, series: list[dict], span=SPAN) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """One study -> (vol (64,336,336) uint8, mask (64,) uint8, one meta dict per slot).

    `series`: dicts with SeriesInstanceUID, plane, fluid. n_slices is counted here from disk.
    """
    for s in series:
        d = study_dir / s["SeriesInstanceUID"]
        s["n_slices"] = sum(1 for f in d.iterdir() if not f.name.startswith(".")) if d.is_dir() else 0
    usable = [s for s in series if s["n_slices"] > 0]
    pick = S.assign_series(usable)
    vol = np.zeros((S.N_SLICES, S.IMG, S.IMG), np.uint8)
    mask = np.zeros(S.N_SLICES, np.uint8)
    meta = []
    for (a, b), slot in zip(S.slot_offsets(), S.SLOTS):
        ser = pick.get(slot.name)
        row = {"slot": slot.name, "series": None, "n_available": 0, "source_step": 0.0,
               "ordered_by": None, "row_mm": np.nan, "col_mm": np.nan, "picks": "", "decode_fail": 0}
        if ser is None:
            meta.append(row)
            continue
        files, how = order_series(study_dir / ser["SeriesInstanceUID"])
        n = len(files)
        picks = S.pick_depths(n, slot.n_slices, *span)
        row.update(series=ser["SeriesInstanceUID"], fluid=ser.get("fluid"), n_available=n,
                   source_step=S.source_step(n, slot.n_slices, *span), ordered_by=how,
                   picks=" ".join(map(str, picks.tolist())))
        arrs = {}
        for p in sorted(set(picks.tolist())):
            try:
                arrs[p] = decode(files[p][0])
            except Exception:                               # noqa: BLE001 -- counted, slot survives
                row["decode_fail"] += 1
        if arrs:
            vals = np.concatenate([x.ravel() for x in arrs.values()])
            lo, hi = np.percentile(vals, PCT)
            rmm = float(np.nanmedian([f[1] for f in files]))
            cmm = float(np.nanmedian([f[2] for f in files]))
            row.update(row_mm=rmm, col_mm=cmm)
            for j, p in enumerate(picks):
                if p not in arrs:
                    continue
                r, c = files[p][1], files[p][2]
                r, c = (rmm if not np.isfinite(r) else r), (cmm if not np.isfinite(c) else c)
                if not (np.isfinite(r) and np.isfinite(c)) or r <= 0 or c <= 0:
                    row["decode_fail"] += 1
                    continue
                w = np.clip((arrs[p] - lo) / (hi - lo + 1e-6), 0, 1)
                vol[a + j] = np.round(crop_mm(w, r, c) * 255).astype(np.uint8)
                mask[a + j] = 1
        meta.append(row)
    return vol, mask, meta


# ----------------------------------------------------------------------------- driver
def _one(args):
    uid, study_dir, series, span = args
    t = time.time()
    try:
        vol, mask, meta = build_study(Path(study_dir), series, span)
        err = None
    except Exception as e:                                  # noqa: BLE001 -- a study never kills the shard
        vol, mask, meta, err = (np.zeros((S.N_SLICES, S.IMG, S.IMG), np.uint8),
                                np.zeros(S.N_SLICES, np.uint8), [], f"{type(e).__name__}: {e}")
    return uid, vol, mask, meta, err, time.time() - t


def render_edges(vols: np.ndarray, masks: np.ndarray, ids, out: Path, n: int = 20, tile: int = 112) -> None:
    """GATE A: first and last slice of each slot, for `n` studies -- anatomy, not wrap or air."""
    import cv2
    offs = S.slot_offsets()
    rows = []
    for i in range(min(n, len(vols))):
        tiles = []
        for a, b in offs:
            for k in (a, b - 1):
                t = cv2.resize(vols[i, k], (tile, tile), interpolation=cv2.INTER_AREA)
                if not masks[i, k]:
                    t = np.full_like(t, 40)
                tiles.append(t)
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(str(out / "render_edges.png"), np.concatenate(rows, 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="competition dir (train_series.csv)")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--span", type=float, nargs=2, default=SPAN)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--check", action="store_true", help="assert the fixture's known answers")
    a = ap.parse_args()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)

    ser = pd.read_csv(a.root / f"{a.split}_series.csv")
    img_dir = a.root / f"{a.split}_series"
    studies = sorted(ser.StudyInstanceUID.astype(str).unique())
    mine = [u for i, u in enumerate(studies) if i % a.n_shards == a.shard]
    if a.limit:
        mine = mine[: a.limit]
    by_study = {u: [{"SeriesInstanceUID": str(r.SeriesInstanceUID), "plane": r.Anatomical_Plane,
                     "fluid": int(r.Fluid_Sensitive) if pd.notna(r.Fluid_Sensitive) else 0}
                    for r in g.itertuples()] for u, g in ser.groupby(ser.StudyInstanceUID.astype(str))}
    log(f"{a.split}: {len(studies)} studies, shard {a.shard}/{a.n_shards} -> {len(mine)} | "
        f"span {a.span[0]:.0%}-{a.span[1]:.0%} | {a.workers} workers | out {a.out}")

    vols = np.lib.format.open_memmap(a.out / "vols.npy", mode="w+", dtype=np.uint8,
                                     shape=(len(mine), S.N_SLICES, S.IMG, S.IMG))
    masks = np.zeros((len(mine), S.N_SLICES), np.uint8)
    rows, errs, secs = [], {}, []
    row_of = {u: i for i, u in enumerate(mine)}
    jobs = [(u, str(img_dir / u), by_study.get(u, []), tuple(a.span)) for u in mine]
    # --workers 1 runs in-process: debuggable, and the only way a monkeypatched function is seen
    # (macOS spawns workers, which re-import this module fresh).
    ex = ProcessPoolExecutor(max_workers=a.workers) if a.workers > 1 else None
    results = ex.map(_one, jobs, chunksize=2) if ex else map(_one, jobs)
    try:
        for k, (uid, vol, mask, meta, err, sec) in enumerate(results):
            i = row_of[uid]
            vols[i], masks[i] = vol, mask
            secs.append(sec)
            if err:
                errs[uid] = err
            for m in meta:
                rows.append({"StudyInstanceUID": uid, **m})
            if k % 100 == 0 or k == len(jobs) - 1:
                log(f"  {k + 1}/{len(jobs)}  {np.mean(secs):.2f} s/study/worker  "
                    f"elapsed {(time.time() - t0) / 60:.1f} min  errors {len(errs)}")
    finally:
        if ex:
            ex.shutdown()
    vols.flush()
    np.save(a.out / "masks.npy", masks)
    np.save(a.out / "ids.npy", np.asarray(mine, dtype=object), allow_pickle=True)
    meta = pd.DataFrame(rows)
    meta.to_csv(a.out / "slots.csv", index=False)
    if a.shard == 0:
        render_edges(vols, masks, mine, a.out)

    fill = {s.name: float(masks[:, x:y].any(1).mean()) for (x, y), s in zip(S.slot_offsets(), S.SLOTS)}
    # GATE A, measured over the whole shard rather than eyeballed: the share of pixels that are
    # tissue (> 25/255) on each slot's first, middle and last slice. The 09-23 smoke read the
    # sagittal edges at ~0.43 of pixels against ~0.82 mid-stack, and the anterior coronal at ~0.21.
    edge = {}
    for (x, y), s in zip(S.slot_offsets(), S.SLOTS):
        ok = np.flatnonzero(masks[:, x])[:300]
        if len(ok):
            t = lambda k: float((np.asarray(vols[ok, k]) > 25).mean())   # noqa: E731
            edge[s.name] = {"first": t(x), "mid": t((x + y) // 2), "last": t(y - 1), "n": int(len(ok))}
    ordered = meta.ordered_by.value_counts(dropna=True).to_dict() if len(meta) else {}
    man = {**S.manifest(tuple(a.span)), "split": a.split, "shard": a.shard, "n_shards": a.n_shards,
           "n_studies": len(mine), "errors": errs, "slot_fill": fill, "edge_tissue": edge,
           "ordered_by": ordered,
           "decode_fail": int(meta.decode_fail.sum()) if len(meta) else 0,
           "crop": "physical, padded (not clamped)", "window_pct": list(PCT),
           "sec_per_study_per_worker": float(np.mean(secs)) if secs else None,
           "wall_min": (time.time() - t0) / 60}
    (a.out / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
    log(f"slot fill {json.dumps({k: round(v, 3) for k, v in fill.items()})}")
    log("edge tissue (first/mid/last) " + " | ".join(
        f"{k} {v['first']:.2f}/{v['mid']:.2f}/{v['last']:.2f}" for k, v in edge.items()))
    log(f"ordered_by {ordered} | decode failures {man['decode_fail']} | study errors {len(errs)}")
    log(f"done {len(mine)} studies in {man['wall_min']:.1f} min")
    if a.check:
        _check_fixture(a.out)


def _check_fixture(out: Path) -> None:
    """The known answers stage_a_fixture.py builds in. See that file for why each one matters."""
    vols, masks = np.load(out / "vols.npy", mmap_mode="r"), np.load(out / "masks.npy")
    ids = np.load(out / "ids.npy", allow_pickle=True).astype(str).tolist()
    meta = pd.read_csv(out / "slots.csv")
    truth = json.loads((out.parent / "input" / "truth.json").read_text())
    offs = dict(zip([s.name for s in S.SLOTS], S.slot_offsets()))
    px = lambda mm: mm / S.FOV_MM * S.IMG                   # noqa: E731
    stripe_row = int(round(px(-45.5 + S.FOV_MM / 2)))
    want = int(round(px(80.0)))                             # the 80 mm square -> 192 px
    n_checks = 0
    for uid, t in truth.items():
        i = ids.index(uid)
        for slot, exp in t.items():
            a, b = offs[slot]
            if exp is None:
                assert masks[i, a:b].sum() == 0, f"{uid} {slot}: should be empty and masked"
                n_checks += 1
                continue
            assert masks[i, a:b].all(), f"{uid} {slot}: slices missing"
            m = meta[(meta.StudyInstanceUID == uid) & (meta.slot == slot)].iloc[0]
            assert int(m.n_available) == exp["n"], f"{uid} {slot}: took a {m.n_available}-slice series, want {exp['n']}"
            assert m.ordered_by == "normal", f"{uid} {slot}: ordered by {m.ordered_by}"
            # 1. ORDER: the stripe encodes true position; it must move left-to-right through the slot.
            #    Filenames and InstanceNumber are both shuffled, so either shortcut fails here.
            pos = [int(np.argmax(vols[i, k, stripe_row, :])) for k in range(a, b)]
            assert all(np.diff(pos) >= 0) and pos[-1] > pos[0], f"{uid} {slot}: not in geometric order {pos}"
            # 2. SCALE: 80 mm must be 192 px on BOTH axes, whatever the pixel spacing (incl. anisotropic)
            mid = np.asarray(vols[i, (a + b) // 2])
            w, h = int((mid[S.IMG // 2, :] > 200).sum()), int((mid[:, S.IMG // 2] > 200).sum())
            assert abs(w - want) <= 3 and abs(h - want) <= 3, f"{uid} {slot}: square {w}x{h} px, want {want}"
            # 3. PADDING: a scan narrower than 140 mm leaves black borders instead of being stretched
            if exp["kind"] == "small_fov":
                assert mid[S.IMG // 2, :8].max() == 0, f"{uid} {slot}: small FOV was stretched, not padded"
            else:
                assert mid[S.IMG // 2, :8].min() > 0, f"{uid} {slot}: a full-FOV scan came out padded"
            n_checks += 1
    print(f"FIXTURE CHECK PASSED: {n_checks} slot checks over {len(truth)} studies -- series choice, "
          f"geometric order, 140 mm scale on both axes, padding, masking, MONOCHROME1")


if __name__ == "__main__":
    main()
