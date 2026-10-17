"""K: knee-centred region crops for the four weak findings (PLAN.md), one cache row per study.

    python pipeline/kcache.py --root <competition dir> --out <dir> --weights anatomy.pt --vendor <monai060> \
        [--shard 0 --n-shards 5] [--limit N] [--montages 6]
    python pipeline/kcache.py --check        # crop geometry on synthetic DICOM knees (no model)

Per study: the knee joint centre J and the patella centre P are found on the fluid-sensitive sagittal series with the
public SKM-TEA segmenter (heliosli's `anatomy.pt`; 9 slices rendered at its 0.3125 mm, S at the top, A at the left),
in patient coordinates (LPS mm). Four regions are then cut from the original series around them, each slice resampled
onto a fixed grid in the canonical frame (left knees mirrored: columns run lateral -> medial or anterior -> posterior):

  lat_sag   sagittal  the lateral compartment (6-42 mm lateral of J), 80 mm around J      Lateral OA, Lateral Meniscus
  lat_cor   coronal   slices 15 mm anterior to 20 mm posterior of J, the lateral half      Lateral OA, Lateral Meniscus
  pf_ax     axial     slices within 15 mm of P, 96 mm around the patellofemoral joint    PF OA
  ant_sag   sagittal  central slices (within 12 mm of J), 96 mm around the front          Synovitis, PF OA

Eight slices a region, 256 x 256 px (0.3125 mm; pf_ax and ant_sag 0.375 mm). The output is the trainer's cache contract
(`vols.npy` N x 32 x 256 x 256 uint8, `masks.npy`, `ids.npy`, `manifest.json` spec "k4-32slice-256", `slots.csv`).
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
import build_slots64 as B  # noqa: E402

SPEC, IMG, PER = "k4-32slice-256", 256, 8
SEG_MM, SEG_PX, N_ANCHOR = 0.3125, 512, 9
X, Y, Z = np.eye(3)
# name, plane, fluid preference, the slab along the slice normal (mm, relative to the anchor), spacing (mm/px)
REGIONS = (("lat_sag", "Sagittal", 1, (-42.0, -6.0), 0.3125),
           ("lat_cor", "Coronal", 1, (-15.0, 20.0), 0.3125),
           ("pf_ax", "Axial", 1, (-15.0, 15.0), 0.375),
           ("ant_sag", "Sagittal", 1, (-12.0, 12.0), 0.375))
# P - J when the patella is not found: 40 mm anterior (-y), 25 mm superior (+z) -- heliosli's anchor default ("+30 cut
# it" on the gold montages); 35 clipped the patella in K's smoke montages (10-16)
PATELLA_DEFAULT = np.array([0.0, -40.0, 25.0])
VENDOR_TREE_SHA = "1dcf1dabb83ab80eec50a5c0abfc407b3ba785bf9f30ae6c0e25da124ef87b05"
ANATOMY_SHA = "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- geometry
def read_series(d: Path) -> list[dict]:
    """Slice headers of one series, sorted along its normal: file, IPP, row / column direction, spacings."""
    import pydicom
    out = []
    for f in sorted(d.glob("*.dcm")):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:                                   # noqa: BLE001
            continue
        iop, ipp, ps = (getattr(h, k, None) for k in ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing"))
        if iop is None or ipp is None or ps is None or len(iop) != 6:
            continue
        r, c = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
        out.append({"file": f, "ipp": np.asarray(ipp, float), "r": r, "c": c, "n": np.cross(r, c),
                    "ps_row": float(ps[0]), "ps_col": float(ps[1]), "shape": (int(h.Rows), int(h.Columns))})
    if not out:
        return out
    n = out[0]["n"]
    out.sort(key=lambda s: float(s["ipp"] @ n))
    return out


def render(img: np.ndarray, s: dict, centre: np.ndarray, row_ax: np.ndarray, col_ax: np.ndarray, mm: float,
           size: int) -> np.ndarray:
    """Resample one slice onto a size x size grid of `mm` pixels centred on `centre` (projected onto the slice), rows
    along `row_ax` and columns along `col_ax` (both projected into the slice plane). Outside the image: 0."""
    n = s["n"] / np.linalg.norm(s["n"])
    c0 = centre - ((centre - s["ipp"]) @ n) * n
    ra = row_ax - (row_ax @ n) * n
    ca = col_ax - (col_ax @ n) * n
    ra, ca = ra / max(np.linalg.norm(ra), 1e-6), ca / max(np.linalg.norm(ca), 1e-6)
    h = (size - 1) / 2.0
    # output (x = column b, y = row a) -> source (x = column j, y = row i)
    jr, jc = s["r"] / s["ps_col"], s["c"] / s["ps_row"]
    rel = c0 - s["ipp"] - h * mm * (ra + ca)
    M = np.array([[mm * (ca @ jr), mm * (ra @ jr), rel @ jr],
                  [mm * (ca @ jc), mm * (ra @ jc), rel @ jc]], np.float32)
    return cv2.warpAffine(img.astype(np.float32), M, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def world_of(s_centre: np.ndarray, s: dict, row_ax: np.ndarray, col_ax: np.ndarray, mm: float, size: int,
             a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Grid pixels (a = row, b = column) of a `render` call -> patient coordinates."""
    n = s["n"] / np.linalg.norm(s["n"])
    c0 = s_centre - ((s_centre - s["ipp"]) @ n) * n
    ra = row_ax - (row_ax @ n) * n
    ca = col_ax - (col_ax @ n) * n
    ra, ca = ra / max(np.linalg.norm(ra), 1e-6), ca / max(np.linalg.norm(ca), 1e-6)
    h = (size - 1) / 2.0
    return c0[None] + ((a - h) * mm)[:, None] * ra[None] + ((b - h) * mm)[:, None] * ca[None]


def slice_centre(s: dict) -> np.ndarray:
    H, W = s["shape"]
    return s["ipp"] + (W - 1) / 2 * s["ps_col"] * s["r"] + (H - 1) / 2 * s["ps_row"] * s["c"]


def axes(plane: str, m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(normal used for the slab, row axis, column axis) of a region's grid in the canonical frame."""
    if plane == "Sagittal":
        return m, -Z, Y            # rows down, columns anterior -> posterior; slab lateral -> medial
    if plane == "Coronal":
        return Y, -Z, m            # rows down, columns lateral -> medial; slab anterior -> posterior
    return Z, Y, m                 # axial: rows anterior -> posterior, columns lateral -> medial; slab inferior -> superior


def pick_slab(series: list[dict], normal: np.ndarray, at: float, lo: float, hi: float, k: int) -> list[int]:
    """Indices of up to k slices whose centre lies in [at + lo, at + hi] along `normal`, evenly spread."""
    pos = np.array([slice_centre(s) @ normal for s in series])
    ok = np.flatnonzero((pos >= at + lo) & (pos <= at + hi))
    if len(ok) <= k:
        return sorted(ok.tolist(), key=lambda i: pos[i])
    want = np.linspace(at + lo, at + hi, k)
    order = ok[np.argsort(pos[ok])]
    chosen = sorted({int(order[np.argmin(np.abs(pos[order] - w))]) for w in want}, key=lambda i: pos[i])
    return chosen


# ----------------------------------------------------------------------------- the anchors
def tree_sha256(root: str) -> str:
    import hashlib
    lines = []
    for f in sorted(Path(root).rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            lines.append(f"{f.relative_to(root).as_posix()}\t{hashlib.sha256(f.read_bytes()).hexdigest()}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def load_segmenter(weights: str, vendor: str, dev: str = "cpu"):
    """heliosli's SKM-TEA V-Net (MONAI 0.6, vendored) with both its weights and the vendored code hash-checked."""
    import hashlib
    import torch
    if hashlib.sha256(Path(weights).read_bytes()).hexdigest() != ANATOMY_SHA:
        raise SystemExit(f"{weights}: not R6's anatomy.pt")
    if tree_sha256(vendor) != VENDOR_TREE_SHA:
        raise SystemExit(f"{vendor}: the vendored MONAI tree is not the pinned one")
    sys.path.insert(0, str(Path(vendor).resolve()))
    from monai.networks.nets import VNet
    model = VNet(spatial_dims=2, in_channels=1, out_channels=4, act=("elu", {"inplace": True}),
                 dropout_prob=0., dropout_dim=2)
    state = torch.load(weights, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict({k.removeprefix("model.net."): v for k, v in state.items()}, strict=True)
    return model.eval().requires_grad_(False).to(dev)


def anchors(model, sag: list[dict], imgs: dict, m: np.ndarray, dev: str = "cpu") -> tuple[np.ndarray, np.ndarray, str]:
    """(J, P, method) from the fluid-sensitive sagittal series: N_ANCHOR slices over its central 80 %, each rendered at
    the segmenter's 0.3125 mm with S at the top and A at the left; joint tissue (femoral / tibial cartilage, meniscus)
    and patellar cartilage centroids in patient coordinates, weighted by confident area, inside the central 70 %."""
    import torch
    import torch.nn.functional as F
    vals = np.concatenate([imgs[i].ravel() for i in imgs])
    mean, std = float(vals.mean()), max(float(vals.std()), 1.0)
    idx = sorted(imgs)
    pts_j, w_j, pts_p, w_p = [], [], [], []
    h = (SEG_PX - 1) / 2.0
    keep = np.zeros((SEG_PX, SEG_PX), bool)                 # joint tissue: the central 70 % (edge firing, 10-16)
    keep[int(.15 * SEG_PX):int(.85 * SEG_PX), int(.15 * SEG_PX):int(.85 * SEG_PX)] = True
    keep_p = np.zeros((SEG_PX, SEG_PX), bool)               # patella: everything but a 5 % border (it sits anterior)
    keep_p[int(.05 * SEG_PX):int(.95 * SEG_PX), int(.05 * SEG_PX):int(.95 * SEG_PX)] = True
    with torch.inference_mode():
        for i in idx:
            s, c = sag[i], slice_centre(sag[i])
            x = (render(imgs[i], s, c, -Z, Y, SEG_MM, SEG_PX) - mean) / std
            t = torch.from_numpy(x)[None, None].to(dev)
            p = model(t).sigmoid()[0].float().cpu().numpy()
            joint = p[1:4].max(0) > .5
            for mask, pts, ws, least in ((joint & keep, pts_j, w_j, 20), ((p[0] > .5) & keep_p, pts_p, w_p, 10)):
                a, b = np.nonzero(mask)
                if len(a) >= least:
                    pts.append(world_of(c, s, -Z, Y, SEG_MM, SEG_PX, a, b).mean(0))
                    ws.append(len(a))
    mass = sum(w_j)
    if mass < 250:
        centre = np.mean([slice_centre(sag[i]) for i in idx], 0)
        return centre, centre + PATELLA_DEFAULT, "fallback: no joint tissue"
    J = np.average(np.array(pts_j), 0, weights=np.array(w_j, float))
    if sum(w_p) >= 60:
        P = np.average(np.array(pts_p), 0, weights=np.array(w_p, float))
        if not (P[1] < J[1] and P[2] > J[2] - 10):        # the patella sits anterior to the joint and not below it
            return J, J + PATELLA_DEFAULT, "segmenter; patella implausible"
        return J, P, "segmenter"
    return J, J + PATELLA_DEFAULT, "segmenter; no patella"


# ----------------------------------------------------------------------------- one study
def build_study(study_dir: Path, series: list[dict], model=None, dev: str = "cpu",
                override: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> (vol (32, IMG, IMG) uint8, mask (32,) uint8, meta). `override` = (J, P) skips the segmenter (tests)."""
    import slots96
    vol = np.zeros((len(REGIONS) * PER, IMG, IMG), np.uint8)
    mask = np.zeros(len(REGIONS) * PER, np.uint8)
    for s in series:
        d = study_dir / s["SeriesInstanceUID"]
        s["n_slices"] = len(list(d.glob("*.dcm"))) if d.is_dir() else 0
    usable = [s for s in series if s["n_slices"] > 0]
    side, side_src = B.study_side(study_dir, [s["SeriesInstanceUID"] for s in usable])
    m = -X if side == "L" else X                      # lateral -> medial, as build_slots64.canon_ops
    meta = {"side": side or "", "side_src": side_src}
    by_plane = {}
    for plane in ("Sagittal", "Coronal", "Axial"):
        cands = sorted([s for s in usable if s["plane"] == plane], key=lambda s: (-int(s.get("fluid", 0)), -s["n_slices"]))
        by_plane[plane] = cands[0] if cands else None
    sag_pick = slots96.assign_series(usable).get("sag_1")
    if sag_pick is None:
        meta["anchor"] = "no sagittal series"
        return vol, mask, meta
    sag = read_series(study_dir / sag_pick["SeriesInstanceUID"])
    if len(sag) < 5:
        meta["anchor"] = "sagittal series without geometry"
        return vol, mask, meta
    cache = {}

    def img(series_hdr, i):
        key = (str(series_hdr[i]["file"]))
        if key not in cache:
            cache[key] = B.decode(series_hdr[i]["file"])
        return cache[key]

    if override is not None:
        J, P, how = override[0], override[1], "override"
    else:
        lo, hi = int(.1 * len(sag)), max(int(.9 * len(sag)) - 1, int(.1 * len(sag)))
        ai = sorted({int(round(v)) for v in np.linspace(lo, hi, N_ANCHOR)})
        J, P, how = anchors(model, sag, {i: img(sag, i) for i in ai}, m, dev)
    meta.update(anchor=how, J=[round(float(v), 2) for v in J], P=[round(float(v), 2) for v in P])
    for k, (name, plane, _, (lo, hi), mm) in enumerate(REGIONS):
        ser_row = sag_pick if plane == "Sagittal" else by_plane[plane]
        if ser_row is None:
            meta[name] = "no series"
            continue
        hdr = sag if ser_row is sag_pick else read_series(study_dir / ser_row["SeriesInstanceUID"])
        if not hdr:
            meta[name] = "no geometry"
            continue
        normal, row_ax, col_ax = axes(plane, m)
        if name == "lat_sag":
            ref, centre = J @ normal, J
        elif name == "lat_cor":
            ref, centre = J @ normal, J - 22.0 * m             # the lateral half of the joint
        elif name == "pf_ax":
            ref, centre = P @ normal, P + 12.0 * Y              # the patella, the joint and the trochlea
        else:
            ref, centre = J @ normal, P + 25.0 * Y              # the front: suprapatellar recess to Hoffa's fat pad
        picks = pick_slab(hdr, normal, ref, lo, hi, PER)
        if not picks:
            meta[name] = "no slice in the slab"
            continue
        tiles = []
        for i in picks:
            try:
                tiles.append(render(img(hdr, i), hdr[i], centre, row_ax, col_ax, mm, IMG))
            except Exception:                               # noqa: BLE001
                tiles.append(None)
        ok = [t for t in tiles if t is not None]
        if not ok:
            meta[name] = "decode failed"
            continue
        v = np.concatenate([t[t > 0].ravel() for t in ok]) if any((t > 0).any() for t in ok) else np.zeros(1)
        p_lo, p_hi = np.percentile(v, B.PCT) if v.size > 1 else (0.0, 1.0)
        for j, t in enumerate(tiles):
            if t is None:
                continue
            vol[k * PER + j] = np.round(np.clip((t - p_lo) / (p_hi - p_lo + 1e-6), 0, 1) * 255).astype(np.uint8)
            mask[k * PER + j] = 1
        meta[name] = f"{len(ok)} slices of {ser_row['SeriesInstanceUID'][-8:]}"
    return vol, mask, meta


def manifest() -> dict:
    return {"spec": SPEC, "img": IMG, "depth_range": [0.0, 1.0], "canon": True,
            "slots": [{"name": n, "plane": p, "n_slices": PER, "fluid": f} for n, p, f, _, _ in REGIONS],
            "regions": [{"name": n, "slab_mm": list(s), "mm_per_px": mm} for n, _, _, s, mm in REGIONS]}


def montage(vol: np.ndarray, mask: np.ndarray, path: Path, title: str) -> None:
    rows = []
    for k in range(len(REGIONS)):
        tiles = [cv2.resize(vol[k * PER + j], (128, 128)) if mask[k * PER + j] else np.zeros((128, 128), np.uint8)
                 for j in (0, 2, 5, 7)]
        row = np.hstack(tiles)
        cv2.putText(row, REGIONS[k][0], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .45, 255, 1)
        rows.append(row)
    img = np.vstack(rows)
    cv2.putText(img, title[-12:], (4, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, .45, 255, 1)
    cv2.imwrite(str(path), img)


# ----------------------------------------------------------------------------- the check
def _check() -> None:
    """canon_fixture's synthetic knees (left / right, side by tag / geometry, standard and reversed storage) with the
    anchor given. On the raw renders (before windowing): the lateral block (raw 700) sits where the canonical frame
    puts it in lat_sag and lat_cor, the medial block (raw 1000) never appears there and sits where it should in pf_ax,
    ant_sag's columns run anterior -> posterior; a left knee renders like a right one; build_study fills every region."""
    import tempfile
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
    import canon_fixture as CF
    root = Path(tempfile.mkdtemp()) / "fx"
    truth = CF.write(root)
    ser = pd.read_csv(root / "train_series.csv")
    cx = {name: c for name, (_, c, *_) in CF.STUDIES.items()}
    h = (IMG - 1) / 2.0
    # expected block centres (row, column) on the grid, from the phantom's knee-local boxes (canon_fixture.phantom)
    want = {"lat_sag": (h + 22.5 / 0.3125, h + 20.0 / 0.3125),           # lateral block: si -22.5, ap +20
            "lat_cor": (h + 22.5 / 0.3125, h + (-32.5 + 22.0) / 0.3125),  # si -22.5, mu -32.5 (centre at mu -22)
            "pf_ax": (h + (-30.0 - (PATELLA_DEFAULT[1] + 12.0)) / REGIONS[2][4], None)}   # medial block: ap -30
    renders = {}
    n = 0
    for suid, t in truth.items():
        rows = [{"SeriesInstanceUID": str(r.SeriesInstanceUID), "plane": r.Anatomical_Plane, "fluid": 1}
                for r in ser[ser.StudyInstanceUID == suid].itertuples()]
        J = np.array([cx[t["name"]], 0.0, 0.0])
        P = J + PATELLA_DEFAULT
        m = -X if t["side"] == "L" else X
        uid_of = {r["plane"]: r["SeriesInstanceUID"] for r in rows}
        for name, plane, _, (lo, hi), mm in REGIONS:
            hdr = read_series(root / "train_series" / suid / uid_of[plane])
            normal, row_ax, col_ax = axes(plane, m)
            ref = (P if name == "pf_ax" else J) @ normal
            centre = {"lat_sag": J, "lat_cor": J - 22.0 * m, "pf_ax": P + 12.0 * Y, "ant_sag": P + 25.0 * Y}[name]
            picks = pick_slab(hdr, normal, ref, lo, hi, PER)
            assert len(picks) >= 5, (t["name"], name, picks)
            tiles = np.stack([render(B.decode(hdr[i]["file"]), hdr[i], centre, row_ax, col_ax, mm, IMG) for i in picks])
            renders[(t["name"], name)] = tiles
            if name in ("lat_sag", "lat_cor"):
                assert not (tiles >= 999).any(), (t["name"], name, "the medial block is in a lateral region")
                blk = np.argwhere((tiles >= 699) & (tiles < 701))
                assert len(blk) > 500, (t["name"], name, "no lateral block")
                (ar, ac), (er, ec) = blk[:, 1:].mean(0), want[name]
                assert abs(ar - er) < 6 and abs(ac - ec) < 6, (t["name"], name, (ar, ac), (er, ec))
                n += 1
            elif name == "pf_ax":
                blk = np.argwhere(tiles >= 999)
                assert len(blk) > 300 and abs(blk[:, 1].mean() - want[name][0]) < 6 and blk[:, 2].mean() > h + 50, \
                    (t["name"], name, blk[:, 1:].mean(0) if len(blk) else None)
                n += 1
            else:
                mid = tiles[len(tiles) // 2]
                row = mid[int(h)]
                nz = np.flatnonzero((row > 250) & (row < 690))
                assert len(nz) > 100 and row[nz[-20:]].mean() > row[nz[:20]].mean(), (t["name"], name, "columns not A -> P")
                n += 1
        vol, mask, meta = build_study(root / "train_series" / suid, rows, override=(J, P))
        assert meta["side"] == t["side"] and mask.sum() >= 4 * 5, (t["name"], meta, mask)
        n += 1
    for name, *_ in REGIONS:
        d = np.abs(renders[("R_tag", name)] - renders[("L_tag", name)]).max()
        assert d < 1e-3, (name, "a left knee does not render like a right one", d)
        n += 1
    print(f"kcache check ok: {n} checks on {len(truth)} synthetic knees (block positions within 6 px of the analytic "
          f"ones, no medial block in the lateral regions, left = right)")


# ----------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--weights")
    ap.add_argument("--vendor")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--uids", type=Path, help="json list: only these studies (in this order)")
    ap.add_argument("--montages", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        _check()
        return
    import pandas as pd
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count() or 4)
    a.out.mkdir(parents=True, exist_ok=True)
    ser = pd.read_csv(a.root / f"{a.split}_series.csv")
    if a.uids:
        mine = json.loads(a.uids.read_text())
    else:
        studies = sorted(ser.StudyInstanceUID.astype(str).unique())
        mine = [u for i, u in enumerate(studies) if i % a.n_shards == a.shard]
    if a.limit:
        mine = mine[:a.limit]
    by_study = {u: [{"SeriesInstanceUID": str(r.SeriesInstanceUID), "plane": r.Anatomical_Plane,
                     "fluid": int(r.Fluid_Sensitive) if pd.notna(r.Fluid_Sensitive) else 0}
                    for r in g.itertuples()] for u, g in ser.groupby(ser.StudyInstanceUID.astype(str))}
    model = load_segmenter(a.weights, a.vendor, dev)
    log(f"{a.split}: shard {a.shard}/{a.n_shards} -> {len(mine)} studies | device {dev} | out {a.out}")
    vols = np.lib.format.open_memmap(a.out / "vols.npy", mode="w+", dtype=np.uint8,
                                     shape=(len(mine), len(REGIONS) * PER, IMG, IMG))
    masks = np.zeros((len(mine), len(REGIONS) * PER), np.uint8)
    rows, t0, done = [], time.time(), 0
    for k, uid in enumerate(mine):
        if a.deadline and time.time() > a.deadline:
            log(f"deadline: stopped after {k} of {len(mine)}")
            break
        t1 = time.time()
        try:
            v, msk, meta = build_study(a.root / f"{a.split}_series" / uid, by_study.get(uid, []), model, dev)
            vols[k], masks[k] = v, msk
            if k < a.montages:
                montage(v, msk, a.out / f"montage_{k:02d}.png", uid)
        except Exception as e:                              # noqa: BLE001
            meta = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        meta.update(StudyInstanceUID=uid, seconds=round(time.time() - t1, 2))
        rows.append(meta)
        done = k + 1
        if k % 25 == 0 or k == len(mine) - 1:
            log(f"{k + 1}/{len(mine)} {time.time() - t1:.1f} s | mean {(time.time() - t0) / (k + 1):.1f} s/study | "
                f"errors {sum('error' in r for r in rows)}")
    vols.flush()
    np.save(a.out / "masks.npy", masks[:done])
    np.save(a.out / "ids.npy", np.asarray(mine[:done], dtype=object), allow_pickle=True)
    if done < len(mine):                                    # keep the contract: rows = ids
        np.save(a.out / "vols_partial_rows.npy", np.array([done]))
    meta = pd.DataFrame(rows)
    meta.to_csv(a.out / "slots.csv", index=False)
    fill = {n: float(masks[:done, i * PER:(i + 1) * PER].any(1).mean()) for i, (n, *_) in enumerate(REGIONS)}
    how = meta["anchor"].fillna("error").value_counts().to_dict() if "anchor" in meta else {}
    (a.out / "manifest.json").write_text(json.dumps({**manifest(), "split": a.split, "shard": a.shard,
                                                     "n_shards": a.n_shards, "n_studies": done, "region_fill": fill,
                                                     "anchor": how, "wall_min": (time.time() - t0) / 60},
                                                    indent=2, default=str))
    log(f"region fill {json.dumps({k: round(v, 3) for k, v in fill.items()})} | anchor {how}")
    log(f"done {done} studies in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
