"""Synthetic DICOM studies WITH PIXELS and known answers, to check `pipeline/build_slots64.py` off Kaggle.

    python notebooks/stage_a_fixture.py
    python pipeline/build_slots64.py --root /tmp/safix/input/competitions/rsna-knee-abnormality-detection \
        --out /tmp/safix/out --check

WHY. Stage 0's fixture checks arithmetic on headers. A cache can fail in ways no header shows and
no crash reports: slices in the wrong order, a crop in pixels rather than millimetres, a small scan
stretched to fill the box, the wrong series in a slot. Each of those trains perfectly and scores
wrongly (§9h). So every slice here draws, in PHYSICAL units about the image centre:

  * an 80 mm bright square -- must come out 192 px on both axes whatever the pixel spacing;
  * a thin stripe whose x position encodes the slice's TRUE position along the normal -- must move
    left to right through every slot. Filenames are shuffled UIDs and InstanceNumber is shuffled
    too, so a reader that trusts either one fails. (A first version ran InstanceNumber backwards
    and positions along +x; for the sagittal IOP the normal is -x, so the two agreed and the trap
    tested nothing -- the geometry-correct reader "failed" the check instead.)
  * dark corners (4% of the area) so the 2-98th percentile window keeps the background above 0,
    which is what lets padding be told apart from a stretched image.
"""
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

INPUT = Path("/tmp/safix/input")
ROOT = INPUT / "competitions" / "rsna-knee-abnormality-detection"
if INPUT.exists():
    shutil.rmtree(INPUT)
IMG = ROOT / "train_series"
IMG.mkdir(parents=True)
random.seed(0)

IOP = {"Sagittal": [0, 1, 0, 0, 0, -1], "Coronal": [1, 0, 0, 0, 0, -1], "Axial": [1, 0, 0, 0, 1, 0]}

# (plane, fluid, n_slices, rows, cols, row_mm, col_mm, photometric)
SPEC = {
    "F1": [("Sagittal", 1, 30, 512, 512, 0.3125, 0.3125, "MONOCHROME2"),
           ("Sagittal", 0, 24, 480, 512, 0.35, 0.30, "MONOCHROME2"),     # anisotropic pixels
           ("Coronal", 1, 20, 512, 512, 0.3125, 0.3125, "MONOCHROME1"),  # stored inverted
           ("Coronal", 0, 28, 448, 448, 0.35, 0.35, "MONOCHROME2"),
           ("Coronal", 0, 12, 448, 448, 0.35, 0.35, "MONOCHROME2"),      # cor_2 must NOT take this
           ("Axial", 1, 18, 448, 448, 0.35, 0.35, "MONOCHROME2")],
    "F2": [("Sagittal", 1, 22, 256, 256, 0.40, 0.40, "MONOCHROME2")],     # 102 mm FOV -> padded
    "F3": [("Sagittal", 1, 10, 512, 512, 0.3125, 0.3125, "MONOCHROME2"),  # shorter than the slot
           ("Axial", 1, 40, 512, 512, 0.3125, 0.3125, "MONOCHROME2")],
}
TRUTH = {  # slot -> None (empty) or the series it must take, by slice count
    "F1": {"sag_1": {"n": 30, "kind": "normal"}, "sag_2": {"n": 24, "kind": "normal"},
           "cor_1": {"n": 20, "kind": "normal"}, "cor_2": {"n": 28, "kind": "normal"},
           "ax": {"n": 18, "kind": "normal"}},
    "F2": {"sag_1": {"n": 22, "kind": "small_fov"}, "sag_2": None, "cor_1": None, "cor_2": None, "ax": None},
    "F3": {"sag_1": {"n": 10, "kind": "normal"}, "sag_2": None, "cor_1": None, "cor_2": None,
           "ax": {"n": 40, "kind": "normal"}},
}


def draw(h, w, row_mm, col_mm, k, n):
    y = (np.arange(h) - (h - 1) / 2)[:, None] * row_mm
    x = (np.arange(w) - (w - 1) / 2)[None, :] * col_mm
    a = np.full((h, w), 300.0)
    a[(np.abs(y) > 0.4 * h * row_mm) & (np.abs(x) > 0.4 * w * col_mm)] = 0          # dark corners
    a[(np.abs(x) <= 40) & (np.abs(y) <= 40)] = 1000                                  # 80 mm square
    xs = -40 + 80 * k / max(n - 1, 1)
    a[(np.abs(y + 45.5) <= 2.5) & (np.abs(x - xs) <= 1.0)] = 1000                    # position stripe
    return a


rows, truth = [], {}
for name, sers in SPEC.items():
    suid = generate_uid()
    truth[suid] = TRUTH[name]
    for plane, fluid, n, h, w, rmm, cmm, photo in sers:
        seuid = generate_uid()
        rows.append({"StudyInstanceUID": suid, "SeriesInstanceUID": seuid,
                     "Anatomical_Plane": plane, "Fluid_Sensitive": fluid})
        d = IMG / suid / seuid
        d.mkdir(parents=True)
        # positions run along the TRUE normal, row x col of IOP -- for this sagittal IOP that is -x
        nrm = np.cross(np.array(IOP[plane][:3], float), np.array(IOP[plane][3:], float))
        order = list(range(n))
        random.shuffle(order)
        inst = random.sample(range(1, n + 1), n)          # InstanceNumber: a shuffle, never the true order
        for k in order:
            a = draw(h, w, rmm, cmm, k, n)
            if photo == "MONOCHROME1":
                a = a.max() - a
            fm = FileMetaDataset()
            fm.TransferSyntaxUID = ExplicitVRLittleEndian
            fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
            fm.MediaStorageSOPInstanceUID = generate_uid()
            ds = Dataset()
            ds.file_meta = fm
            ds.Rows, ds.Columns = h, w
            ds.PixelSpacing = [rmm, cmm]
            ds.ImageOrientationPatient = IOP[plane]
            ds.ImagePositionPatient = list(nrm * (k * 3.5))
            ds.InstanceNumber = inst[k]                     # shuffled: trusting it must fail
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photo
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
            ds.PixelData = a.astype(np.uint16).tobytes()
            ds.is_little_endian, ds.is_implicit_VR = True, False
            ds.save_as(d / f"{generate_uid()}.dcm", enforce_file_format=True)
pd.DataFrame(rows).to_csv(ROOT / "train_series.csv", index=False)
(INPUT / "truth.json").write_text(json.dumps(truth, indent=1))
print(f"fixture: {len(SPEC)} studies, {len(rows)} series -> {ROOT}")
