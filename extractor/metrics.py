"""Scoring functions shared by run_extract.py and compare_methods.py.

Separate module for two reasons: run_extract.py does all its work at import time, so importing
anything from it re-runs the full extraction as a side effect; and rule_extractor.py is
deliberately stdlib-only, so numpy/pandas do not belong there.
"""
import numpy as np, pandas as pd


def auc(y, s):
    """Rank-based AUC; nan if only one class present.

    Ties get averaged ranks, so a fully binary predictor scores exactly its balanced accuracy.
    That is the correct behaviour and it is why compare_methods.py reports both -- a binary
    label set has one operating point and cannot be rewarded for ranking it does not do.
    """
    y = np.asarray(y, float); s = np.asarray(s, float)
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
