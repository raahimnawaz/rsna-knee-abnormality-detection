"""Shared scoring functions: rank-based AUROC and balanced accuracy."""
import warnings

import numpy as np, pandas as pd


def auc(y, s):
    """Rank-based AUC; nan if only one class present."""
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
