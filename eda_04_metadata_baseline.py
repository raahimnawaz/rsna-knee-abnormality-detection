"""Does series metadata alone predict the 12 labels? Measured answer: no.

The hypothesis was worth testing because it would have explained the leaderboard. Which
sequences a radiologist orders plausibly depends on what they suspect, and test_series.csv
ships the same Anatomical_Plane / Fluid_Sensitive / Fat_Suppression columns as training --
so a protocol-fingerprint shortcut would score without ever touching a pixel.

It does not. Macro AUC 0.471 on the 58 gold studies, 5-fold CV. Below chance.

Recorded so nobody re-runs it, and because it retires the same concern from the other
direction: any model that fuses across acquisitions is at risk of learning *which series were
acquired* rather than what is in them. That risk is real in principle, but there is no signal
here to learn, so a fused model picking up protocol structure would be fitting noise.

Needs: data/train.csv, data/train_series.csv.  Runs in ~5 s on CPU.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parent
D = PROJ / "data"
SEED = 0
MIN_CLASS = 5   # below this either class, CV AUC is meaningless

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def features() -> pd.DataFrame:
    """Per-study: count of each of the 6 series types, presence flag for each, total count."""
    se = pd.read_csv(D / "train_series.csv")
    se["kind"] = se.Anatomical_Plane + "_" + se.Fluid_Sensitive.map({0: "nonFS", 1: "FS"})
    piv = se.pivot_table(index="StudyInstanceUID", columns="kind",
                         values="SeriesInstanceUID", aggfunc="count").fillna(0)
    kinds = list(piv.columns)
    piv["n_series"] = piv[kinds].sum(axis=1)
    for k in kinds:
        piv["has_" + k] = (piv[k] > 0).astype(int)
    return piv.reset_index()


def main() -> None:
    X_all = features()
    feat = [c for c in X_all.columns if c != "StudyInstanceUID"]
    gold = pd.read_csv(D / "train.csv").dropna(subset=LABELS).merge(X_all,
                                                                   on="StudyInstanceUID")

    print(f"{len(feat)} features over {len(gold)} gold studies: {feat}\n")
    print(f"{'label':<18}{'n_pos':>6}{'CV AUC':>9}")
    print("-" * 33)

    aucs = []
    for lab in LABELS:
        y = gold[lab].astype(int).values
        X = gold[feat].values
        if min(y.sum(), (1 - y).sum()) < MIN_CLASS:
            print(f"{lab:<18}{y.sum():>6}{'--':>9}")
            continue
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
            m = HistGradientBoostingClassifier(max_iter=120, max_depth=3, random_state=SEED)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        a = roc_auc_score(y, oof)
        aucs.append(a)
        print(f"{lab:<18}{y.sum():>6}{a:>9.3f}")

    macro = float(np.mean(aucs))
    print("-" * 33)
    print(f"{'MACRO':<18}{'':>6}{macro:>9.3f}   metadata only, n={len(gold)}")
    print("\n0.5 is chance. Anything near it means the acquisition protocol carries no label")
    print("signal, and the leaderboard is explained by something else -- see PLAN.md 7.1.")
    if macro > 0.60:
        print("\n!! ABOVE 0.60 -- this contradicts the 2026-08-07 measurement of 0.471.")
        print("   Check whether train_series.csv gained columns before believing it.")


if __name__ == "__main__":
    main()
