"""Scoring functions shared by run_extract.py and compare_methods.py.

Separate module for two reasons: run_extract.py does all its work at import time, so importing
anything from it re-runs the full extraction as a side effect; and rule_extractor.py is
deliberately stdlib-only, so numpy/pandas do not belong there.
"""
import warnings

import numpy as np, pandas as pd


def auc(y, s):
    """Rank-based AUC; nan if only one class present.

    Ties get averaged ranks, so a fully binary predictor scores exactly its balanced accuracy.
    That is the correct behaviour and it is why compare_methods.py reports both -- a binary
    label set has one operating point and cannot be rewarded for ranking it does not do.

    Only rows labelled exactly 0 or 1 are ranked. Until 2026-09-23 every row was, so an unlabelled
    (NaN) row still took a rank and shifted the positives' rank sum: auc([1, 0, nan], [.9, .1, .5])
    returned 2.0. No caller passed one -- all 23 filter to complete rows first and the hand labels
    are strictly 0/1 -- so no recorded number moves. It was a door, not a leak (AUDIT.md).

    A non-finite SCORE on a labelled row still returns nan, as it always did, and callers nanmean
    over labels, so that label silently leaves the macro. It now warns when it does.
    """
    y = np.asarray(y, float); s = np.asarray(s, float)
    lab = (y == 0) | (y == 1)
    y, s = y[lab], s[lab]
    if not np.isfinite(s).all():
        warnings.warn("auc: non-finite score on a labelled row -> nan; a nanmean over labels "
                      "will drop this label without saying so", RuntimeWarning, stacklevel=2)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def bal_acc(y, s, thr=0.5):
    """(sensitivity + specificity) / 2 at a fixed threshold; nan if only one class present."""
    y = np.asarray(y, float)
    p = (np.asarray(s, float) >= thr).astype(float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    return 0.5 * (p[pos].mean() + (1 - p[neg]).mean())
