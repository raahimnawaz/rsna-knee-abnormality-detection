"""B. What does the 336 slot cache cost to build, on real NIfTI?

The fork's layout is six slots per study, GROUP=3 slices each -- so 18 slices per study, against
the ~155 the frozen cache embeds. Cost is dominated by opening the .nii and the in-plane
resample, both of which happen per SERIES regardless of how few slices are kept.

Timed through pipeline.preprocess.load_series_nifti so the number reflects the shared,
parity-checked code path rather than a fast reimplementation that would not be what gets built.
"""
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TARGET_MM", "0.35")
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline import preprocess as pp  # noqa: E402

NII = PROJ / "data" / "nifti" / "nifti_train"
N_STUDY = 20
N_CORPUS = 4407

# The fork's six slots: (plane, fluid_sensitive)
SLOTS = [("Sagittal", 1), ("Coronal", 1), ("Axial", 1),
         ("Sagittal", 0), ("Coronal", 0), ("Axial", 0)]

ser = pd.read_csv(PROJ / "data" / "train_series.csv")
have = {p.name.split("_")[0] for p in NII.glob("*.nii")}
ser = ser[ser.StudyInstanceUID.isin(have)]

path_of = {}
for p in NII.glob("*.nii"):
    stem = p.name[:-4]
    st, _, se = stem.partition("_")
    path_of[(st, se)] = p

lat = pp.study_laterality(PROJ / "data" / "study_meta.csv")

rng = random.Random(0)
studies = sorted(ser.StudyInstanceUID.unique())
pick = rng.sample(studies, min(N_STUDY, len(studies)))

print(__doc__)
print(f"{len(have):,} studies have NIfTI on disk; timing {len(pick)}\n")

per_study, n_series_read, slot_hits = [], 0, defaultdict(int)
for i, st in enumerate(pick):
    rows = ser[ser.StudyInstanceUID == st]
    chosen = {}
    for _, r in rows.iterrows():
        key = (r.Anatomical_Plane, int(r.Fluid_Sensitive))
        if key in SLOTS and key not in chosen:
            chosen[key] = r.SeriesInstanceUID
    t0 = time.time()
    got = 0
    for key, se in chosen.items():
        p = path_of.get((st, se))
        if p is None:
            continue
        try:
            out = pp.load_series_nifti(p, plane=key[0], laterality=lat.get(st))
        except Exception as e:                                    # noqa: BLE001
            print(f"    ! {type(e).__name__}: {str(e)[:60]}")
            continue
        if out is None:
            continue
        got += 1
        slot_hits[key] += 1
    dt = time.time() - t0
    if got:
        per_study.append(dt)
        n_series_read += got
    print(f"  [{i + 1:>2}/{len(pick)}] {got}/6 slots  {dt:>5.2f}s")

a = np.array(per_study)
print(f"\nper study: median {np.median(a):.2f}s  mean {a.mean():.2f}s  "
      f"({n_series_read / len(a):.1f} series/study)")
tot = a.mean() * N_CORPUS / 3600
print(f"full corpus ({N_CORPUS:,} studies), single process: **{tot:.1f} h**")
for w in (4, 8):
    print(f"  {w} workers (perfect scaling, optimistic): {tot / w:.1f} h")
print(f"\nslot coverage over {len(pick)} studies: "
      + ", ".join(f"{p[:3]}{'FS' if f else ''} {slot_hits[(p, f)]}" for p, f in SLOTS))
