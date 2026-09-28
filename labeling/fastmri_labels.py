"""fastMRI+ boxes -> per-knee labels in our 12 columns, plus a seeded patient-level hold-out.

    .venv/bin/python labeling/fastmri_labels.py --plus data/external/fastmri/fastmri_plus \
        --studies data/external/fastmri/cache_stage_a/studies.csv --out data/external/fastmri/cache_stage_a

THE MAPPING, WRITTEN BEFORE ANY TEACHER EXISTS (2026-09-27) and decided by the OFFICIAL criteria (REFERENCE.md §2),
not by what would score well:
- ACL       <- "Ligament - ACL High Grade Sprain". A LOW-grade sprain is 0: the official ACL positive needs a
               high-grade partial or full tear, and "mild signal change without discontinuity" is explicitly negative.
- Effusion  <- "Joint Effusion". Ungraded in fastMRI+, the official positive needs moderate or large: a known
               over-call (small effusions count here), kept because it is still an image read.
- Contusion <- "Bone-Fracture/Contusion/dislocation". fastMRI+ has ONE label for both; the competition splits them,
               so Fracture is left blank rather than guessed.
- Baker's   <- "Periarticular cysts". Includes cysts that are not Baker's (parameniscal, ganglion): noisy positive.
- Everything else is BLANK (NaN): MCL has 4 high-grade knees (too few), menisci need the tear's side (v2), OA /
  synovitis have no fastMRI+ label. `train_windows --allow-missing-labels` masks blanks out of the loss.
Absence = 0: fastMRI+ reviewed every file in `knee_file_list.csv`, so a listed file with no box of a kind is a read
negative. A knee is one patient: its label is the union over that patient's files (PD and PD fat-sat of one exam).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
MAP = {"ACL": ["Ligament - ACL High Grade Sprain"],
       "Effusion": ["Joint Effusion"],
       "Contusion": ["Bone-Fracture/Contusion/dislocation"],
       "Baker's": ["Periarticular cysts"]}
HOLDOUT, SEED = 0.15, 20260927


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plus", type=Path, required=True)
    ap.add_argument("--studies", type=Path, required=True, help="studies.csv from build_fastmri_cache.py")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    k = pd.read_csv(a.plus / "knee.csv")
    k["label"] = k.label.astype(str).str.strip()
    listed = set(pd.read_csv(a.plus / "knee_file_list.csv", header=None)[0].astype(str).str.strip())
    unknown = sorted({x for v in MAP.values() for x in v} - set(k.label))
    if unknown:
        raise SystemExit(f"mapped fastMRI+ labels not found in knee.csv: {unknown}")
    st = pd.read_csv(a.studies)
    rows = []
    for r in st.itertuples():
        files = [f for f in str(r.files).split() if f in listed]
        got = set(k.loc[k.file.isin(files), "label"])
        row = {"StudyInstanceUID": r.StudyInstanceUID, **{c: np.nan for c in LABELS}}
        for col, names in MAP.items():
            row[col] = float(any(n in got for n in names))
        rows.append(row)
    lab = pd.DataFrame(rows)
    lab.to_csv(a.out / "labels_fastmri_v1.csv", index=False)

    rng = np.random.default_rng(SEED)
    uids = lab.StudyInstanceUID.to_numpy()
    hold = set(rng.choice(uids, size=int(round(HOLDOUT * len(uids))), replace=False).tolist())
    pd.Series([u for u in uids if u not in hold], name="StudyInstanceUID").to_csv(a.out / "train_studies.csv", index=False)
    pd.Series([u for u in uids if u in hold], name="StudyInstanceUID").to_csv(a.out / "holdout_studies.csv", index=False)

    print(f"{len(lab)} knees -> {len(uids) - len(hold)} train / {len(hold)} hold-out (seed {SEED})")
    for col in MAP:
        tr = lab.loc[~lab.StudyInstanceUID.isin(hold), col]
        ho = lab.loc[lab.StudyInstanceUID.isin(hold), col]
        print(f"  {col:10s} positives {int(lab[col].sum()):4d} ({lab[col].mean():.1%}) | train {int(tr.sum())} | hold-out {int(ho.sum())}")


if __name__ == "__main__":
    main()
