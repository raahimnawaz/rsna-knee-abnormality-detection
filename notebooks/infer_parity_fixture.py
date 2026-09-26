"""Stage B's inference path against its training cache, on synthetic DICOMs -- window for window.

    python notebooks/canon_fixture.py --out <dir>          # the four knees, and their Stage B cache
    python notebooks/infer_parity_fixture.py --fixture <dir>

WHY AT THE TENSOR LEVEL. `infer_windows.py --check-against` compares PREDICTIONS, which is the right gate
on Kaggle, where the model is the trained one. Off Kaggle there is no trained model, and a random-weight
ViT is almost blind to pixels: on 2026-09-26 a random Stage B model reproduced its OOF to 0.00000 even
with canonical orientation switched OFF -- the check passed a broken path. So here the comparison is
the thing a model would see: every uint8 window, its slot and its depth, from `infer_windows.DicomStudies`
(DICOMs -> `build_study`) against `train_windows.Windows` over the cache `build_slots64 --canon` wrote.
The control rebuilds without --canon and must DIFFER, or the test could not fail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJ / "fusion"), str(PROJ / "pipeline")]
import infer_windows as IW  # noqa: E402
import train_windows as TW  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture", type=Path, required=True, help="canon_fixture.py's --out")
    a = ap.parse_args()
    root = a.fixture / "input"
    corpus = TW.Corpus(a.fixture / "canon", "stage_b")
    uids = list(corpus.ids)
    truth = json.loads((root / "truth.json").read_text())
    tw = TW.Windows(corpus, uids, np.zeros((len(uids), 12), np.float32), k=None)
    by = IW.series_by_study(root, "train")
    fails = []
    for canon in (True, False):
        ds = IW.DicomStudies(root / "train_series", uids, by, 0, spec="c7", canon=canon)
        for i, u in enumerate(uids):
            w2, s2, p2, _ = tw[i]
            _, w1, s1, p1, err, _ = ds[i]
            same = (w1.shape == w2.shape and torch.equal(w1, w2) and torch.equal(s1, s2) and torch.equal(p1, p2))
            d = int((w1.int() - w2.int()).abs().max()) if w1.shape == w2.shape else -1
            print(f"canon={canon!s:5} {truth[u]['name']:6} windows {tuple(w1.shape)} identical={same} "
                  f"max|dpix|={d} {err}")
            if canon and not same:
                fails.append(f"{truth[u]['name']}: the canonical DICOM path differs from the cache")
            if not canon and same:
                fails.append(f"{truth[u]['name']}: identical WITHOUT canon -- the check is blind")
    if fails:
        raise SystemExit("INFER PARITY FAILED: " + "; ".join(fails))
    print(f"INFER PARITY PASSED: {len(uids)} studies, every window, slot and depth identical to the cache; "
          "the no-canon control differs on every study")


if __name__ == "__main__":
    main()
