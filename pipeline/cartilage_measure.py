"""Cartilage and meniscus measurements from the public SKM-TEA segmenter, one row per study (LOG 10-16 ~10:30).

    python pipeline/cartilage_measure.py --root <competition dir> --uids uids.json --weights anatomy.pt \
        --vendor <cloud_code/training/vendor_runtime/monai060> --out <dir> [--montages 6]
    python pipeline/cartilage_measure.py --check        # the measurement maths on synthetic masks, no model

The segmenter is heliosli's `anatomy.pt` (MONAI 0.6 2-D V-Net, SKM-TEA; channels patellar cartilage, femoral
cartilage, tibial cartilage, meniscus; sagittal only; whole-series mean/std; 0.3125 mm; rows S->I, columns A->P).
The series, its order, the knee's side and the in-plane orientation are ours (`build_slots64`: the `sag_1` choice and
the canonical frame, in which rows run S->I, columns A->P and slices run lateral -> medial for either knee).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

MM = 0.3125                       # the segmenter's pixel
TISSUES = ("patellar", "femoral", "tibial", "meniscus")
COMPARTMENT_MARGIN_MM = 4.0       # slices within this of the joint centre belong to neither tibiofemoral compartment
CENTRAL_MM = 12.0                 # central slices: patella and trochlea
TROCHLEA_MM = 20.0                # femoral cartilage within this of the patellar cartilage, on central slices
THIN_MM = 1.0
MIN_COMPONENT_PX = 5


# ----------------------------------------------------------------------------- the model
def load_segmenter(weights: str, vendor: str):
    """heliosli's `training.anatomy.load_segmenter`, without the rest of their package."""
    import torch
    sys.path.insert(0, str(Path(vendor).resolve()))
    from monai.networks.nets import VNet
    model = VNet(spatial_dims=2, in_channels=1, out_channels=4, act=("elu", {"inplace": True}),
                 dropout_prob=0., dropout_dim=2)
    state = torch.load(weights, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict({k.removeprefix("model.net."): v for k, v in state.items()}, strict=True)
    return model.eval().requires_grad_(False)


def segment(model, vol: np.ndarray, row_mm: np.ndarray, col_mm: np.ndarray) -> list[np.ndarray]:
    """vol [S, H, W] canonical sagittal -> per slice uint8 probabilities [4, h, w] on the 0.3125 mm grid."""
    import torch
    import torch.nn.functional as F
    mean, std = float(vol.mean()), max(float(vol.std()), 1.0)
    out = []
    with torch.inference_mode():
        for z in range(len(vol)):
            x = (vol[z].astype(np.float32) - mean) / std
            shape = (max(16, round(x.shape[1] * col_mm[z] / MM)), max(16, round(x.shape[0] * row_mm[z] / MM)))
            x = cv2.resize(x, shape, interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(np.ascontiguousarray(x))[None, None]
            h, w = t.shape[-2:]
            t = F.pad(t, (0, (-w) % 16, 0, (-h) % 16), mode="replicate")
            p = model(t).sigmoid()[0, :, :h, :w].numpy()
            out.append(np.rint(np.clip(p, 0, 1) * 255).astype(np.uint8))
    return out


# ----------------------------------------------------------------------------- reading one study
def read_sagittal(study_dir: Path, series: list[dict]):
    """-> (vol [S,H,W] float32 canonical, positions_mm [S] increasing lateral -> medial, row_mm, col_mm, info)."""
    import pydicom
    import build_slots64 as B
    import slots96
    for s in series:
        d = study_dir / s["SeriesInstanceUID"]
        s["n_slices"] = len(list(d.glob("*.dcm"))) if d.is_dir() else 0
    pick = slots96.assign_series([s for s in series if s["n_slices"] > 0]).get("sag_1")
    if pick is None:
        raise ValueError("no sagittal series")
    side, side_src = B.study_side(study_dir, [s["SeriesInstanceUID"] for s in series])
    recs, iop0 = [], None
    for f in sorted((study_dir / pick["SeriesInstanceUID"]).glob("*.dcm")):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:                                   # noqa: BLE001
            continue
        iop, ipp, ps = (getattr(h, k, None) for k in ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing"))
        if iop is None or ipp is None or ps is None or len(iop) != 6:
            continue
        if iop0 is None:
            iop0 = [float(v) for v in iop]
        recs.append((f, np.asarray(ipp, float), float(ps[0]), float(ps[1])))
    if iop0 is None or len(recs) < 5:
        raise ValueError(f"sagittal series without geometry ({len(recs)} slices)")
    ops = B.canon_ops(iop0, side)
    if ops["axis"] != "x":
        raise ValueError(f"the sag_1 series is not sagittal by geometry (normal along {ops['axis']})")
    recs.sort(key=lambda r: float(np.dot(r[1], ops["slice_dir"])))
    pos = np.array([float(np.dot(r[1], ops["slice_dir"])) for r in recs])
    sw = ops["transpose"]
    row_mm = np.array([r[3] if sw else r[2] for r in recs])
    col_mm = np.array([r[2] if sw else r[3] for r in recs])
    imgs = [B.apply_ops(B.decode(r[0]), ops) for r in recs]
    shp = imgs[0].shape
    keep = [i for i, a in enumerate(imgs) if a.shape == shp]
    vol = np.stack([imgs[i] for i in keep]).astype(np.float32)
    info = {"series": pick["SeriesInstanceUID"], "fluid": int(pick.get("fluid", 0)), "side": side or "",
            "side_src": side_src, "n_slices": len(keep), "native_row_mm": float(np.median(row_mm)),
            "spacing_mm": float(np.median(np.diff(pos[keep]))) if len(keep) > 1 else float("nan")}
    return vol, pos[keep], row_mm[keep], col_mm[keep], info


# ----------------------------------------------------------------------------- the measurements
def ridge_thickness(mask: np.ndarray) -> np.ndarray:
    """Local thickness (mm) on the ridge of the distance transform: 2 x distance at its local maxima."""
    if not mask.any():
        return np.zeros(0, np.float32)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ridge = mask & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6) & (dt > 0)
    return (2.0 * dt[ridge] - 0.5).clip(min=1.0) * MM      # exact to half a pixel for odd and even widths


def n_components(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX).sum())


def group_features(probs: list[np.ndarray], masks: list[np.ndarray], spacing: float, tag: str) -> dict:
    """probs / masks: the slices of one tissue in one region (float [h,w] in 0..1 / bool)."""
    f = {}
    px = sum(int(m.sum()) for m in masks)
    f[f"{tag}_vol"] = px * MM * MM * spacing
    present = [m for m in masks if m.sum() >= MIN_COMPONENT_PX]
    f[f"{tag}_nslices"] = len(present)
    th = np.concatenate([ridge_thickness(m) for m in masks]) if masks else np.zeros(0)
    f[f"{tag}_thick_med"] = float(np.median(th)) if th.size else np.nan
    f[f"{tag}_thick_p10"] = float(np.percentile(th, 10)) if th.size else np.nan
    f[f"{tag}_thin_frac"] = float((th < THIN_MM).mean()) if th.size else np.nan
    f[f"{tag}_ncomp"] = float(np.mean([n_components(m) for m in present])) if present else np.nan
    inside = np.concatenate([p[m] for p, m in zip(probs, masks)]) if px else np.zeros(0)
    f[f"{tag}_conf"] = float(inside.mean()) if inside.size else np.nan
    band = sum(int(((p > .2) & (p < .8)).sum()) for p in probs)
    f[f"{tag}_fuzzy"] = band / max(px, 1)
    return f


def measure(seg: list[np.ndarray], pos: np.ndarray, spacing: float) -> dict:
    """seg: per slice uint8 [4,h,w] (lateral -> medial); pos: slice positions (mm, increasing lateral -> medial)."""
    P = [s.astype(np.float32) / 255. for s in seg]
    M = [p > .5 for p in P]
    joint = np.array([float(np.maximum(np.maximum(m[1], m[2]), m[3]).sum()) for m in M])
    out = {"joint_px": float(joint.sum())}
    if joint.sum() < 200:
        out["seg_ok"] = 0
        return out
    out["seg_ok"] = 1
    c = float((pos * joint).sum() / joint.sum())
    rel = pos - c
    out["lat_extent_mm"] = float(-rel[joint > 50].min()) if (joint > 50).any() else np.nan
    out["med_extent_mm"] = float(rel[joint > 50].max()) if (joint > 50).any() else np.nan
    regions = {"lat": rel < -COMPARTMENT_MARGIN_MM, "med": rel > COMPARTMENT_MARGIN_MM}
    for rn, sel in regions.items():
        idx = np.flatnonzero(sel)
        for ti, tn in ((1, "femoral"), (2, "tibial"), (3, "meniscus")):
            out.update(group_features([P[i][ti] for i in idx], [M[i][ti] for i in idx], spacing, f"{rn}_{tn}"))
    central = np.flatnonzero(np.abs(rel) <= CENTRAL_MM)
    out.update(group_features([P[i][0] for i in range(len(P))], [M[i][0] for i in range(len(P))], spacing, "patellar"))
    tro_p, tro_m = [], []
    for i in central:
        pm = M[i][0]
        if pm.sum() >= MIN_COMPONENT_PX:
            d = cv2.distanceTransform((~pm).astype(np.uint8), cv2.DIST_L2, 5) * MM
            near = d < TROCHLEA_MM
        else:
            near = np.zeros_like(pm)
        tro_m.append(M[i][1] & near)
        tro_p.append(np.where(near, P[i][1], 0.))
    out.update(group_features(tro_p, tro_m, spacing, "trochlea"))
    for tn in ("femoral", "tibial", "meniscus"):
        lv, mv = out.get(f"lat_{tn}_vol", 0.), out.get(f"med_{tn}_vol", 0.)
        out[f"lm_{tn}_vol_ratio"] = lv / mv if mv > 0 else np.nan
    for tn in ("femoral", "tibial"):
        lt, mt = out.get(f"lat_{tn}_thick_med"), out.get(f"med_{tn}_thick_med")
        out[f"lm_{tn}_thick_ratio"] = lt / mt if (lt == lt and mt == mt and mt) else np.nan
    cart = sum(out.get(f"{r}_{t}_vol", 0.) for r in ("lat", "med") for t in ("femoral", "tibial"))
    out["lat_cart_share"] = (out.get("lat_femoral_vol", 0.) + out.get("lat_tibial_vol", 0.)) / cart if cart else np.nan
    return out


def montage(vol: np.ndarray, seg: list[np.ndarray], pos: np.ndarray, path: Path, uid: str) -> None:
    """Four slices (lateral, lateral-central, central, medial) with the four tissues outlined."""
    joint = np.array([float((s[1:4].max(0) > 127).sum()) for s in seg])
    c = float((pos * joint).sum() / max(joint.sum(), 1))
    picks = [int(np.argmin(np.abs(pos - (c + d)))) for d in (-22., -10., 0., 22.)]
    colors = [(255, 0, 255), (0, 255, 0), (0, 128, 255), (255, 255, 0)]  # patellar, femoral, tibial, meniscus (BGR)
    tiles = []
    for z in picks:
        h, w = seg[z].shape[1:]
        a = cv2.resize(vol[z], (w, h), interpolation=cv2.INTER_LINEAR)
        lo, hi = np.percentile(a, [1, 99])
        g = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for ch, col in enumerate(colors):
            cnts, _ = cv2.findContours((seg[z][ch] > 127).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cnts, -1, col, 1)
        rgb = cv2.resize(rgb, (320, int(320 * h / w)))
        cv2.putText(rgb, f"{pos[z] - c:+.0f} mm", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        tiles.append(rgb)
    hh = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
    img = np.hstack(tiles)
    cv2.putText(img, uid[-12:], (6, hh - 8), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    cv2.imwrite(str(path), img)


# ----------------------------------------------------------------------------- checks and the CLI
def _check() -> None:
    """Thickness and volume on synthetic ribbons whose answers are known."""
    h, w = 200, 200
    seg = []
    for z in range(20):
        s = np.zeros((4, h, w), np.uint8)
        s[1, 60:68, 20:180] = 255            # femoral: 8 px = 2.5 mm thick, everywhere
        s[2, 90:94, 20:180] = 255            # tibial: 4 px = 1.25 mm
        s[3, 80:86, 30:60] = 255             # meniscus
        if z >= 10:
            s[2, 90:94, 80:120] = 0          # medial half: a 40 px gap in the tibial cartilage
        seg.append(s)
    pos = np.arange(20) * 3.0                # 3 mm slices
    f = measure(seg, pos, 3.0)
    assert f["seg_ok"] == 1
    assert abs(f["lat_femoral_thick_med"] - 2.5) < 0.35, f["lat_femoral_thick_med"]
    assert abs(f["lat_tibial_thick_med"] - 1.25) < 0.35, f["lat_tibial_thick_med"]
    assert f["lat_tibial_ncomp"] == 1 and f["med_tibial_ncomp"] == 2, (f["lat_tibial_ncomp"], f["med_tibial_ncomp"])
    mass = np.array([float((s[1:4].max(0) > 127).sum()) for s in seg])   # the gap moves the centre laterally
    c = float((pos * mass).sum() / mass.sum())
    n_lat, n_med = int((pos - c < -COMPARTMENT_MARGIN_MM).sum()), int((pos - c > COMPARTMENT_MARGIN_MM).sum())
    want = n_lat * 8 * 160 * MM * MM * 3.0
    assert abs(f["lat_femoral_vol"] - want) < 1e-6 * want + 1e-6, (f["lat_femoral_vol"], want)
    gap_px = sum(1 for z in range(20) if z >= 10 and pos[z] - c > COMPARTMENT_MARGIN_MM) * 4 * 40
    want_med = (n_med * 4 * 160 - gap_px) * MM * MM * 3.0
    assert abs(f["med_tibial_vol"] - want_med) < 1e-6 * want_med + 1e-6, (f["med_tibial_vol"], want_med)
    print(f"check ok: {len(f)} features | lat femoral {f['lat_femoral_thick_med']:.2f} mm, lat tibial "
          f"{f['lat_tibial_thick_med']:.2f} mm, tibial components lat {f['lat_tibial_ncomp']:.0f} / med "
          f"{f['med_tibial_ncomp']:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, help="competition dir (train_series.csv, train_series/)")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--uids", type=Path, help="json list of StudyInstanceUIDs")
    ap.add_argument("--weights")
    ap.add_argument("--vendor", help="the vendored MONAI 0.6 (cloud_code/training/vendor_runtime/monai060)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--montages", type=int, default=0, help="save an outline montage for the first N studies")
    ap.add_argument("--deadline", type=float, default=0, help="unix time: stop starting new studies after it")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        _check()
        return
    import pandas as pd
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    a.out.mkdir(parents=True, exist_ok=True)
    uids = json.loads(a.uids.read_text())
    ser = pd.read_csv(a.root / f"{a.split}_series.csv")
    by_study = {u: [{"SeriesInstanceUID": str(r.SeriesInstanceUID), "plane": r.Anatomical_Plane,
                     "fluid": int(r.Fluid_Sensitive) if pd.notna(r.Fluid_Sensitive) else 0}
                    for r in g.itertuples()] for u, g in ser.groupby(ser.StudyInstanceUID.astype(str))}
    model = load_segmenter(a.weights, a.vendor)
    rows, t0 = [], time.time()
    for k, uid in enumerate(uids):
        if a.deadline and time.time() > a.deadline:
            print(f"deadline: stopped after {k} of {len(uids)} studies", flush=True)
            break
        t1 = time.time()
        row = {"StudyInstanceUID": uid}
        try:
            vol, pos, row_mm, col_mm, info = read_sagittal(a.root / f"{a.split}_series" / uid, by_study.get(uid, []))
            t2 = time.time()
            seg = segment(model, vol, row_mm, col_mm)
            t3 = time.time()
            row.update(info)
            row.update(measure(seg, pos, info["spacing_mm"]))
            row.update(decode_s=round(t2 - t1, 2), segment_s=round(t3 - t2, 2), measure_s=round(time.time() - t3, 2))
            if k < a.montages:
                montage(vol, seg, pos, a.out / f"montage_{k:02d}.png", uid)
        except Exception as e:                              # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rows.append(row)
        if k % 10 == 0 or k == len(uids) - 1:
            print(f"{k + 1}/{len(uids)} {time.time() - t1:.1f} s | mean {(time.time() - t0) / (k + 1):.1f} s/study | "
                  f"errors {sum('error' in r for r in rows)}", flush=True)
            pd.DataFrame(rows).to_csv(a.out / "measurements.csv", index=False)
    pd.DataFrame(rows).to_csv(a.out / "measurements.csv", index=False)
    print(f"done: {len(rows)} studies in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
