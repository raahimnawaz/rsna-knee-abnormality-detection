"""The error budget behind docs/REVIEW-2026-10-12.md: where the five-fold model loses, and why.

    python fusion/error_budget.py

Four tables. Gold-58 is REPORTED here, never selected on.
  1. per finding: the five folds' held-out predictions against the reference reader, and their mean on gold with
     each finding's share of the deficit
  2. the report readers against the image model on gold, with a paired bootstrap
  3. held-out macro by scanner, field strength and report language
  4. how densely Stage B samples each plane

The model's predictions are in the repo (`fusion/runs_cnxs_cons2_*`). The rest is competition or third-party data
under the git-ignored `data/`: train.csv, public_llm_labels/, student/targets_consensus_v2.csv,
external/metadata_probe/headers.parquet, stage0/stage0_series.csv.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "labeling"))
import build_consensus_targets as B  # noqa: E402

L = B.LABELS
D = PROJ / "data"
FOLDS = [("f0", 0), ("f12", 1), ("f12", 2), ("f34", 3), ("f34", 4)]      # run folder, fold
REFERENCE = D / "public_llm_labels" / "lixin73_rsna-knee-llm-report-labels-sol56" / "labels_llm_gpt56sol.csv"
WORDS = {"en": "the|of|is|there|with|no|and", "es": "de|la|el|del|rotura|derrame|sin|con|rodilla",
         "nl": "de|het|een|van|geen|met|knie|scheur", "de": "der|die|das|und|kein|keine|mit|des",
         "tr": "ve|ile|izlenmektedir|normal|menisküs|diz|bir"}


def fold_file(run: str, k: int, name: str) -> pd.DataFrame:
    p = PROJ / "fusion" / f"runs_cnxs_cons2_{run}" / "runs_convnext_small" / f"fold{k}" / name
    return pd.read_csv(p).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")


def auc(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = ~np.isnan(y) & ~np.isnan(p)
    return float("nan") if len(set(y[m])) < 2 else roc_auc_score(y[m], p[m])


def rank(d: pd.DataFrame) -> pd.DataFrame:
    return d.rank(pct=True)


def per_finding(oof, y_ref, gold, ens) -> None:
    tgt = pd.read_csv(D / "student" / "targets_consensus_v2.csv").set_index("StudyInstanceUID").reindex(oof.index)
    t = pd.DataFrame(index=L)
    t["held-out vs reference"] = [auc(y_ref[c], oof[c]) for c in L]
    t["its own target vs reference"] = [auc(y_ref[c].where(tgt[c].notna()), tgt[c]) for c in L]
    t["share of the label's signal recovered"] = (t.iloc[:, 0] - 0.5) / (t.iloc[:, 1] - 0.5)
    print(f"\n== 1a. five folds, held out, against the reference reader (n = {len(oof)})")
    print(t.round(3).to_string(), f"\nMACRO {t.iloc[:, 0].mean():.4f}")

    g = pd.DataFrame({"n pos": gold.sum(), "five folds": [auc(gold[c], ens[c]) for c in L]})
    g["share of the deficit"] = (1 - g["five folds"]) / (1 - g["five folds"]).sum()
    print("\n== 1b. the five folds' mean on gold-58")
    print(g.round(3).to_string(), f"\nMACRO {g['five folds'].mean():.4f}")


def report_vs_image(gold, ens, n_boot: int = 3000) -> None:
    val, _ = B.load_readers(D / "public_llm_labels")
    readers = {k: v.reindex(gold.index)[L] for k, v in val.items()
               if v.reindex(gold.index)[L].notna().all(axis=1).sum() >= 50}      # the tables that carry the gold rows
    report = pd.concat([rank(v) for v in readers.values()]).groupby(level=0).mean().loc[gold.index]
    both = rank(report) + rank(ens)
    t = pd.DataFrame({"report": [auc(gold[c], report[c]) for c in L], "image": [auc(gold[c], ens[c]) for c in L],
                      "report + image": [auc(gold[c], both[c]) for c in L]}, index=L)
    rng = np.random.default_rng(0)
    for c in L:
        y, d = gold[c].to_numpy(), []
        for _ in range(n_boot):
            k = rng.integers(0, len(y), len(y))
            if len(set(y[k])) > 1:
                d.append(roc_auc_score(y[k], ens[c].to_numpy()[k]) - roc_auc_score(y[k], report[c].to_numpy()[k]))
        t.loc[c, ["image - report", "sd", "P(image > report)"]] = np.mean(d), np.std(d), np.mean(np.array(d) > 0)
    print(f"\n== 2. report ({', '.join(readers)}; rank-mean) against image on gold-58")
    print(t.round(3).to_string(), f"\nMACRO report {t.report.mean():.4f}  image {t.image.mean():.4f}")


def language(text: str) -> str:
    t = f" {text.lower()} "
    n = {k: len(re.findall(f" ({w}) ", t)) for k, w in WORDS.items()}
    k = max(n, key=n.get)
    return k if n[k] >= 3 else "other"


def subgroups(oof, y_ref, reports) -> None:
    h = pd.read_parquet(D / "external" / "metadata_probe" / "headers.parquet")
    h["tesla"] = pd.to_numeric(h.MagneticFieldStrength, errors="coerce").round(1)
    s = h.groupby("StudyInstanceUID").agg(scanner=("Manufacturer", lambda v: v.mode().iat[0]),
                                          tesla=("tesla", "median")).reindex(oof.index)
    s["language"] = reports.reindex(oof.index).fillna("").map(language)
    print("\n== 3. held-out macro against the reference, by subgroup (n >= 120)")
    for col in s.columns:
        for name, ids in s.groupby(col).groups.items():
            if len(ids) >= 120:
                m = np.nanmean([auc(y_ref.loc[ids, c], oof.loc[ids, c]) for c in L])
                print(f"  {col:9s} {str(name):26s} n {len(ids):5d}  {m:.4f}")


def sampling() -> None:
    d = pd.read_csv(D / "stage0" / "stage0_series.csv")
    d = d[d.error.isna() & (d.n_files <= 80)]                    # a 2-D series, not a 3-D reconstruction
    d["mm"] = d.spacing_between_mm.fillna(d.slice_thickness_mm)
    print("\n== 4. Stage B, the series each first slot takes")
    for plane, slot in (("Sagittal", 26), ("Coronal", 16), ("Axial", 14)):
        g = d[d.plane == plane].assign(want=lambda t: (t.fluid == 1) & (plane != "Axial"))   # sag_1, cor_1: fluid first
        x = g.sort_values(["want", "n_files"], ascending=False).groupby("study").head(1)
        step = ((x.n_files - 1) * 0.96 / (slot - 1)).clip(lower=1) * x.mm
        print(f"  {plane:9s} {slot} of a median {x.n_files.median():.0f} slices: kept "
              f"{np.minimum(slot / x.n_files, 1).median():.0%}, one every {step.median():.1f} mm, "
              f"a window spans {2 * step.median():.1f} mm")


def main() -> None:
    train = pd.read_csv(D / "train.csv").set_index("StudyInstanceUID")
    gold = (train.dropna(subset=L)[L] > 0).astype(int)
    oof = pd.concat([fold_file(r, k, "oof_all.csv") for r, k in FOLDS])[L]
    ens = sum(fold_file(r, k, "oof_gold.csv").loc[gold.index, L] for r, k in FOLDS) / len(FOLDS)
    ref = pd.read_csv(REFERENCE).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    oof = oof.loc[[u for u in oof.index if u in ref.index and u not in gold.index]]
    y_ref = (ref.loc[oof.index, L].apply(pd.to_numeric, errors="coerce") > 0.5).astype(float)
    per_finding(oof, y_ref, gold, ens)
    report_vs_image(gold, ens)
    subgroups(oof, y_ref, train.Report)
    sampling()


if __name__ == "__main__":
    main()
