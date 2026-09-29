"""Pseudo-labels for unlabelled knees: the mean of the four CNN teacher folds' predictions (Noisy Student, round 1).

    HF_HUB_OFFLINE=1 TW_WORKERS=0 .venv/bin/python fusion/pseudo_label.py \
        --corpus data/external/fastmri/dicom_stage_a --out data/student/nyu_pseudo_r1.csv

The teachers are `fusion/runs_teacher_cnn/fold{1..4}/last.pt` (ResNet34 recipe C on `dread_soft`, the 3,502 training studies in
four site folds -- none saw the 691 or gold). Each scores every study with a usable slot, at every window, exactly as its own
evaluation did (`train_windows.predict`); the pseudo-label is the plain mean of the four probabilities (the teacher on gold was
the mean of the four; no calibration, no thresholding: the student trains on soft targets). Studies whose id carries `__dup_`
(renamed repeats, see the parts' manifests) and studies with no usable slot are left out.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_windows as tw  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--teachers", type=Path, nargs="+",
                    default=[PROJ / f"fusion/runs_teacher_cnn/fold{k}/last.pt" for k in (1, 2, 3, 4)])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    corpus = tw.Corpus(a.corpus, "stage_a")
    ids = [u for u in corpus.ids if "__dup_" not in u and corpus.masks[corpus.row[u]].any()]
    if a.limit:
        ids = ids[: a.limit]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tw.log(f"{len(ids)} studies to label ({len(corpus.ids) - len(ids)} left out: repeats / no usable slot) on {dev}")
    preds = []
    for path in a.teachers:
        t = time.time()
        model = tw.WindowMIL(tw.build_encoder("resnet34", Path("/tmp"), 224, 0.0), 224, n_slots=0).to(dev)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=True)
        p = tw.predict(model, tw.Windows(corpus, ids, np.zeros((len(ids), len(tw.LABELS)), np.float32), k=None, thin=0),
                       dev, False, 2)
        preds.append(p)
        tw.log(f"{path.parent.name}: {len(p)} studies in {(time.time() - t) / 60:.1f} min | mean prob "
               + " ".join(f"{v:.2f}" for v in p.mean(0)))
    mean = np.mean(preds, 0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mean, columns=tw.LABELS).assign(StudyInstanceUID=ids)[["StudyInstanceUID", *tw.LABELS]].to_csv(a.out, index=False)
    r = np.corrcoef(np.stack([p.ravel() for p in preds])) if len(preds) > 1 else None
    tw.log(f"wrote {a.out}: {len(ids)} studies" + ("" if r is None else
           f" | fold-to-fold correlation of predictions {r[np.triu_indices(len(preds), 1)].mean():.3f}"))


if __name__ == "__main__":
    main()
