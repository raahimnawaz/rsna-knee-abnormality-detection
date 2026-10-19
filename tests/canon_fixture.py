"""Synthetic left and right knees in every storage orientation, to test `build_slots64.py --canon`.

    python tests/canon_fixture.py --out <dir> [--spec c8]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

PROJ = Path(__file__).resolve().parents[1]
H = W = 200
MM = 0.8                      # 160 mm field of view, cropped to 140
N, STEP = 24, 4.0             # slices per series and their spacing, mm

# (row vector r, column vector c) per plane: r = direction of increasing COLUMN index
STD = {"Sagittal": ((0, 1, 0), (0, 0, -1)), "Coronal": ((1, 0, 0), (0, 0, -1)), "Axial": ((1, 0, 0), (0, 1, 0))}
REV = {"Sagittal": ((0, -1, 0), (0, 0, 1)), "Coronal": ((-1, 0, 0), (0, 0, 1)), "Axial": ((-1, 0, 0), (0, -1, 0))}
TRANSPOSED_AXIAL = ((0, 1, 0), (1, 0, 0))
STUDIES = {  # name -> (side, knee centre x, tag, orientation per plane)
    "R_tag": ("R", -150.0, "R", STD),
    "L_tag": ("L", 80.0, "L", STD),
    "R_geo": ("R", -150.0, None, REV),
    "L_geo": ("L", 80.0, None, {**REV, "Axial": TRANSPOSED_AXIAL}),
}


def phantom(P: np.ndarray, centre: np.ndarray, m: np.ndarray) -> np.ndarray:
    u = P - centre
    mu, ap, si = u @ m, u[..., 1], u[..., 2]           # medial, posterior, superior (knee-local)
    v = np.zeros(P.shape[:-1])
    inside = (np.abs(mu) < 55) & (np.abs(ap) < 55) & (np.abs(si) < 55)
    v[inside] = 300 + 2.0 * mu[inside] + 1.0 * ap[inside] + 0.5 * si[inside]
    v[(mu > 20) & (mu < 40) & (ap > -40) & (ap < -20) & (si > 10) & (si < 30)] = 1000   # medial-anterior-superior
    v[(mu > -40) & (mu < -25) & (ap > 10) & (ap < 30) & (si > -30) & (si < -15)] = 700  # lateral-posterior-inferior
    return v


def write(root: Path, studies: dict | None = None, mm: float = MM) -> dict:
    """`studies`: name -> (side, knee centre x, tag, orientation per plane[, mirrored]). A mirrored study is the same
    knee with its ANATOMY reflected medial <-> lateral (what `train_windows.mirror_windows` claims to produce);
    `mm`: pixel size (0.7 makes the 200 px image exactly the 140 mm field, so no crop rounds off-centre)."""
    img_dir = root / "train_series"
    if root.exists():
        shutil.rmtree(root)
    img_dir.mkdir(parents=True)
    random.seed(0)
    rows, truth = [], {}
    for name, (side, cx, tag, orient, *more) in (studies or STUDIES).items():
        mirrored = bool(more and more[0])
        suid = generate_uid()
        truth[suid] = {"name": name, "side": side, "src": "tag" if tag else "geometry"}
        centre = np.array([cx, 0.0, 0.0])
        m = np.array([1.0, 0, 0]) if side == "R" else np.array([-1.0, 0, 0])
        for plane, axis in (("Sagittal", 0), ("Coronal", 1), ("Axial", 2)):
            seuid = generate_uid()
            rows.append({"StudyInstanceUID": suid, "SeriesInstanceUID": seuid,
                         "Anatomical_Plane": plane, "Fluid_Sensitive": 1})
            d = img_dir / suid / seuid
            d.mkdir(parents=True)
            r, c = (np.asarray(v, float) for v in orient[plane])
            n = np.cross(r, c)
            ks = list(range(N))
            random.shuffle(ks)                              # file order and InstanceNumber: shuffled
            for inst, k in enumerate(ks, 1):
                t = (k - (N - 1) / 2) * STEP
                ctr = centre + t * n
                ipp = ctr - (W - 1) / 2 * mm * r - (H - 1) / 2 * mm * c
                jj, ii = np.meshgrid(np.arange(W), np.arange(H))
                P = ipp + jj[..., None] * mm * r + ii[..., None] * mm * c
                a = phantom(P, centre, -m if mirrored else m)
                fm = FileMetaDataset()
                fm.TransferSyntaxUID = ExplicitVRLittleEndian
                fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
                fm.MediaStorageSOPInstanceUID = generate_uid()
                ds = Dataset()
                ds.file_meta = fm
                ds.Rows, ds.Columns = H, W
                ds.PixelSpacing = [mm, mm]
                ds.ImageOrientationPatient = [float(x) for x in (*r, *c)]
                ds.ImagePositionPatient = [float(x) for x in ipp]
                ds.InstanceNumber = random.randint(1, 10_000)
                if tag:
                    ds.Laterality = tag
                ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
                ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
                ds.PixelData = np.round(a).astype(np.uint16).tobytes()
                ds.is_little_endian, ds.is_implicit_VR = True, False
                ds.save_as(d / f"{generate_uid()}.dcm", enforce_file_format=True)
    pd.DataFrame(rows).to_csv(root / "train_series.csv", index=False)
    (root / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth


def build(root: Path, out: Path, canon: bool, spec: str = "c7") -> None:
    cmd = [sys.executable, str(PROJ / "pipeline" / "build_slots64.py"), "--root", str(root), "--out", str(out),
           "--spec", spec, "--workers", "1"] + (["--canon"] if canon else [])
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def check(root: Path, out: Path, truth: dict, spec: str = "c7") -> None:
    sys.path.insert(0, str(PROJ / "pipeline"))
    import build_slots64
    S = build_slots64.SPECS[spec]
    vols, masks = np.load(out / "vols.npy"), np.load(out / "masks.npy")
    ids = np.load(out / "ids.npy", allow_pickle=True).astype(str).tolist()
    meta = pd.read_csv(out / "slots.csv")
    offs = dict(zip([s.name for s in S.SLOTS], S.slot_offsets()))
    by = {truth[u]["name"]: i for i, u in enumerate(ids)}
    n = 0
    # (1) side and its source
    for u, t in truth.items():
        got = meta[meta.StudyInstanceUID == u].iloc[0]
        assert (got.side, got.side_src) == (t["side"], t["src"]), f"{t['name']}: side {got.side}/{got.side_src}"
        n += 1
    # (2) all four identical, in every filled slot
    ref = vols[by["R_tag"]]
    for name, i in by.items():
        assert (masks[i] == masks[by["R_tag"]]).all(), f"{name}: mask differs"
        diff = np.abs(vols[i].astype(int) - ref.astype(int)).max()
        assert diff <= 1, f"{name}: canonical volume differs from R_tag by up to {diff}"
        n += 1
    def trend(v):
        z, y, x = np.nonzero((v > 5) & (v < 250))
        val = v[z, y, x].astype(float)
        return {k: float(np.corrcoef(val, idx)[0, 1]) for k, idx in (("slice", z), ("row", y), ("col", x))}
    want = {
        "sag_1": {"slice": +1, "col": +1, "row": -1},   # lateral->medial, anterior->posterior, superior->inferior
        "cor_1": {"slice": +1, "col": +1, "row": -1},   # anterior->posterior, lateral->medial, superior->inferior
        "ax_1":  {"slice": -1, "col": +1, "row": +1},   # superior->inferior, lateral->medial, anterior->posterior
    }
    for slot, signs in want.items():
        a, b = offs[slot]
        t = trend(ref[a:b])
        for axis, sign in signs.items():
            assert np.sign(t[axis]) == sign and abs(t[axis]) > 0.05, f"{slot} {axis}: corr {t[axis]:+.3f}, want sign {sign:+d}"
            n += 1
    print(f"CANON FIXTURE PASSED: {n} checks -- side from tag and geometry, four storage conventions "
          f"(incl. a transposed axial) -> identical canonical volumes, every axis where documented")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spec", choices=["c7"], default="c7", help="c7 = Stage B")
    a = ap.parse_args()
    root = a.out / "input"
    truth = write(root)
    build(root, a.out / "canon", canon=True, spec=a.spec)
    check(root, a.out / "canon", truth, a.spec)
    # (4) the test can fail: without --canon, the left knee is not the right knee
    build(root, a.out / "raw", canon=False, spec=a.spec)
    v = np.load(a.out / "raw" / "vols.npy")
    ids = np.load(a.out / "raw" / "ids.npy", allow_pickle=True).astype(str).tolist()
    by = {truth[u]["name"]: i for i, u in enumerate(ids)}
    d = np.abs(v[by["L_tag"]].astype(int) - v[by["R_tag"]].astype(int)).max()
    assert d > 50, "without --canon the left and right knees came out the same: the check is blind"
    print(f"control: without --canon, L_tag differs from R_tag by up to {d} -- the check can fail")


if __name__ == "__main__":
    main()
