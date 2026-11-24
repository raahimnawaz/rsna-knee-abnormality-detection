"""Training targets from three published image reads (LLMs that read every series plus the report), averaged.

    python labeling/build_image_read_targets.py         # -> data/student/targets_imgread_v1.csv (+ .json)

Readers (CC BY 4.0): 1st place tascj0/rsna-2026-knee-agent-labels, 21st drizlo/rsna-knee-mri-llm-soft-labels,
22nd matsurihime/rsna-knee-llm-readings (the astra_image__ columns). A cell one reader left blank is the mean of the
other two. The 58 graded studies are left out, as in every earlier target table. 7th place's reads
(songqizhou/rsna-knee-llm-labels, CC BY-NC-SA) only judge the result and never enter it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PROJ = Path(__file__).resolve().parents[1]
D = PROJ / "data"
P = D / "public_llm_labels"
READERS = {
    "1st_tascj0": (P / "tascj0_rsna-2026-knee-agent-labels" / "labels_astra_trained_guide.csv", ""),
    "21st_drizlo": (P / "drizlo_rsna-knee-mri-llm-soft-labels" / "labels.csv", ""),
    "22nd_matsurihime": (P / "matsurihime_rsna-knee-llm-readings" / "llm_readings.csv", "astra_image__"),
}
JUDGES = {
    "7th_image_only": P / "songqizhou_rsna-knee-llm-labels" / "2_read_image_only.csv",
    "7th_verification": P / "songqizhou_rsna-knee-llm-labels" / "3_read_verification.csv",
}
PREV = D / "student" / "targets_teacher_v1.csv"
OUT = D / "student" / "targets_imgread_v1.csv"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(path: Path, prefix: str = "") -> pd.DataFrame:
    d = pd.read_csv(path).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    d = d[[prefix + c for c in LABELS]].apply(pd.to_numeric, errors="coerce")
    d.columns = LABELS
    if ((d < 0) | (d > 1)).any().any():
        raise SystemExit(f"{path.name}: a value outside [0, 1]")
    return d


def macro(y: pd.DataFrame, p: pd.DataFrame) -> float:
    return round(float(np.mean([roc_auc_score(y[c], p[c]) for c in LABELS])), 4)


def main() -> None:
    train = pd.read_csv(D / "train.csv", low_memory=False).drop_duplicates("StudyInstanceUID")
    graded = train[LABELS].notna().all(axis=1)
    gold = train[graded].set_index("StudyInstanceUID")[LABELS].astype(float)
    studies = pd.Index(sorted(train.loc[~graded, "StudyInstanceUID"]))
    if PREV.exists() and set(pd.read_csv(PREV, usecols=["StudyInstanceUID"]).StudyInstanceUID) != set(studies):
        raise SystemExit(f"the study set differs from {PREV.name}")

    reads = {k: load(path, pre) for k, (path, pre) in READERS.items()}
    for k, r in reads.items():
        if not studies.isin(r.index).all() or not gold.index.isin(r.index).all():
            raise SystemExit(f"{k}: a training study is missing")
    stack = np.stack([r.reindex(studies)[LABELS].to_numpy(float) for r in reads.values()])
    n_read = (~np.isnan(stack)).sum(0)
    if (n_read < 2).any():
        raise SystemExit("a cell has fewer than two reads")
    out = pd.DataFrame(np.round(np.nanmean(stack, 0), 4), index=studies, columns=LABELS)
    out.index.name = "StudyInstanceUID"
    out.reset_index().to_csv(OUT, index=False)

    yg = (gold > 0.5).astype(int)
    avg_gold = sum(r.loc[gold.index] for r in reads.values()) / len(reads)
    meta = {"readers": {k: {"file": str(path.relative_to(D)), "sha256": sha(path)} for k, (path, _) in READERS.items()},
            "studies": len(out), "graded_left_out": len(gold),
            "cells_from_two_reads": [{"StudyInstanceUID": studies[i], "label": LABELS[j]}
                                     for i, j in zip(*np.where(n_read == 2))],
            "check_graded_58_macro_auroc": {**{k: macro(yg, r.loc[gold.index]) for k, r in reads.items()},
                                            "average": macro(yg, avg_gold)},
            "judges": {}}
    for name, path in JUDGES.items():
        if path.exists():
            j = load(path)
            ids = studies[studies.isin(j.index)]
            y = (j.loc[ids] > 0.5).astype(int)
            meta["judges"][name] = {"file": str(path.relative_to(D)), "sha256": sha(path), "studies": len(ids),
                                    **{k: macro(y, r.loc[ids].fillna(out.loc[ids])) for k, r in reads.items()},
                                    "average": macro(y, out.loc[ids])}
    meta["out"] = OUT.name
    meta["out_sha256"] = sha(OUT)
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
