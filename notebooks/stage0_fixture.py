"""Synthetic DICOM tree with KNOWN geometry, to smoke `build_stage0_kernel.py` off Kaggle.

    python notebooks/stage0_fixture.py
    KAGGLE_INPUT=/tmp/s0fix/input KAGGLE_WORKING=/tmp/s0fix/out python /tmp/s0fix/work.py

WHY THIS EXISTS. Every cache build in this project's history died hours in on something a
20-study run would have caught. Stage 0 is cheap enough that the failure would only cost minutes,
but the thing that matters is not whether it CRASHES -- it is whether the arithmetic is RIGHT.
A geometry summary that runs clean and reports the wrong ratio would send the 3D decision the
wrong way with no symptom, which is 9h's failure class.

So the fixture encodes geometry we know the answer to: r = spacing/in_plane is exactly 10.0 for
Sagittal and Coronal and 8.0 for Axial, and slice filenames are shuffled UIDs like the real
corpus, so a reader that trusts filename order gets the spacing wrong.

⛔ SliceThickness is deliberately 0.85x the true spacing, i.e. there is an inter-slice GAP. A
reader that uses the thickness tag instead of the position delta understates r by 15% here. That
is the whole reason stage0 measures both and reports where they disagree.
"""
import shutil, random
from pathlib import Path
import numpy as np, pandas as pd
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROOT = Path("/tmp/s0fix/input/rsna-knee-abnormality-detection")
if ROOT.exists(): shutil.rmtree(ROOT.parent)
IMG = ROOT / "train_images"; IMG.mkdir(parents=True)
random.seed(0)

# plane, fluid, in_plane_mm, spacing_mm(true), n_slices  -> r = spacing/in_plane
SPEC = {
 "S1": [("Sagittal",1,0.35,3.5,30), ("Sagittal",0,0.35,3.5,25), ("Coronal",1,0.40,4.0,20), ("Axial",1,0.50,4.0,18)],
 "S2": [("Sagittal",1,0.35,3.5,22)],                                  # sparse study
 "S3": [("Sagittal",1,0.35,3.5,28), ("Sagittal",0,0.35,3.5,24),
        ("Coronal",1,0.40,4.0,20), ("Coronal",0,0.40,4.0,16), ("Axial",1,0.50,4.0,18)],
 "S4": [("Sagittal",1,0.35,3.5,2)],                                   # too short for pass 2
}
NORMAL = {"Sagittal": (1,0,0), "Coronal": (0,1,0), "Axial": (0,0,1)}
IOP = {"Sagittal":[0,1,0,0,0,-1], "Coronal":[1,0,0,0,0,-1], "Axial":[1,0,0,0,1,0]}

rows = []
for study, sers in SPEC.items():
    suid = generate_uid()
    for plane, fluid, ip, sp, n in sers:
        seuid = generate_uid()
        rows.append({"StudyInstanceUID": suid, "SeriesInstanceUID": seuid,
                     "Anatomical_Plane": plane, "Fluid_Sensitive": fluid})
        d = IMG / suid / seuid; d.mkdir(parents=True)
        nrm = np.array(NORMAL[plane], dtype=float)
        order = list(range(n)); random.shuffle(order)     # filenames are random UIDs, like the real corpus
        for k in order:
            fm = FileMetaDataset(); fm.TransferSyntaxUID = ExplicitVRLittleEndian
            fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
            fm.MediaStorageSOPInstanceUID = generate_uid()
            ds = Dataset(); ds.file_meta = fm
            ds.PixelSpacing = [ip, ip]
            ds.SliceThickness = sp * 0.85          # thickness < spacing: there IS a gap
            ds.SpacingBetweenSlices = sp
            ds.Rows, ds.Columns = 384, 384
            ds.SeriesDescription = f"{plane[:3]}_{'FS' if fluid else 'T1'}"
            ds.ImageOrientationPatient = IOP[plane]
            ds.ImagePositionPatient = list(nrm * (k * sp))
            ds.is_little_endian, ds.is_implicit_VR = True, False
            ds.save_as(d / f"{generate_uid()}.dcm", enforce_file_format=True)
pd.DataFrame(rows).to_csv(ROOT / "train_series.csv", index=False)
pd.DataFrame({"StudyInstanceUID": [r["StudyInstanceUID"] for r in rows]}).to_csv(ROOT / "train.csv", index=False)
print(f"fixture: {len(SPEC)} studies, {len(rows)} series")
print("expected r: Sagittal 3.5/0.35=10.0 | Coronal 4.0/0.40=10.0 | Axial 4.0/0.50=8.0")
print("expected slots64: 2+ sagittal in S1,S3 = 50% of 4 studies | 2+ coronal in S3 = 25%")
