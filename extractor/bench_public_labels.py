"""Score every available report-label source against the 58 gold studies.

`compare_methods.py` answers "do we beat `nekkon`?" and the answer is yes. That question is
dead: `nekkon`'s CSV is a binary rule set from week one, and the field has since published
four *LLM-read* label tables as free Kaggle Datasets. This module is the same decision gate
re-pointed at what is actually available — README §5 called the comparison stale on
2026-08-09 and this is the measurement that settles it.

Run:
    python extractor/bench_public_labels.py --download    # ~1 MB, needs kaggle auth
    python extractor/bench_public_labels.py

Everything is scored on the same 58 gold studies with the same macro-AUROC the competition
uses, so the rows are directly comparable to the 0.777 in IMPROVEMENTS.md §0.

Two things this script is careful about, because both would flatter us:

  - **Hanley-McNeil SE is printed next to every macro.** At n=58 with 9-35 positives per
    label the noise floor is large, and a table of bare point estimates invites exactly the
    over-reading §0 warns about. A gap has to clear the SEs to mean anything.
  - **Rank-mean, not probability-mean, is the combiner.** The metric reads order only
    (PLAN.md §6), so averaging probabilities lets whichever source is most confident dominate
    for no reason the score can see.

Caveat that belongs next to the result: `llm_labels_v4_blend` is described by its author as a
blend, and a blend may have been selected on these same 58 studies. The unblended reads
(`llm_labels_full`, `report_labels_v2`) are the honest comparison; both still clear ours by
~0.10.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import auc  # noqa: E402  -- shared definition, do not re-implement

PROJ = Path(__file__).resolve().parent.parent
D = PROJ / "data"
PUB = D / "public_llm_labels"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# dataset ref -> the file inside it that carries the 12 columns. Pinned by name because
# several of these datasets ship two or three variants and the differences matter.
SOURCES = {
    "steven_v4":   ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv"),
    # v2 is the author's own documented best: v1 plus filling ONLY the undecided Synovitis
    # cells from the Effusion field. Their forum post publishes 0.8780 -> 0.8873 for that step
    # and reports the blanket twelve-label version (v3) as WORSE at 0.8805. v4_blend has no
    # published derivation, so v2 is the one whose provenance can be defended -- see §2i.
    "steven_v2":   ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v2.csv"),
    "steven_full": ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_full.csv"),
    "pilkwang_v2": ("pilkwang/rsna-knee-llm-labels", "report_labels_v2.csv"),
    "pilkwang_v1": ("pilkwang/rsna-knee-report-labels", "report_labels_v1.csv"),
    "lixin_gpt56": ("lixin73/rsna-knee-llm-report-labels-sol56", "labels_llm_gpt56sol.csv"),
    # Added 2026-09-23 (RUNBOOK §0.3 survey). dread_soft is the table behind the frontier's 0.928
    # single model (MODEL.md §0), CC0. jev_hybrid is, per its README, a per-target pick between
    # steven_v4 and pilkwang_v2 made ON gold-58 plus a blend weight set on gold-74 -- its gold read
    # is selected by construction. jev_text is the same author's text-only half. hosseini_jev ships
    # class probabilities (0 / 1 / 9 = not said); `align` scores it P(1) + 0.5 P(9).
    "dread_soft":   ("dreaddevelopment/rsna-knee-labels", "labels_llm_soft.csv"),
    "jev_hybrid":   ("riadmohamed42/jev-knee-labels", "HYBRID_labels_scores.csv"),
    "jev_text":     ("riadmohamed42/jev-knee-labels", "FINAL2_labels_scores.csv"),
    "hosseini_jev": ("hosseini7798/rsna-knee-label-extraction-jev", "Label_Extraction_Jev_v2.csv"),
}
# What each author discloses about fitting to gold. A read on the rows it was fitted to is a
# CEILING, and choosing between ceilings on 58 rows is RUNBOOK §3 rule 1.
SELECTED_ON_GOLD = {
    "steven_v4": "a blend; may have been selected on gold (author)",
    "jev_hybrid": "per-target winner picked on gold-58, blend weight on gold-74 (README)",
}


# ref -> dataset version to download instead of the latest. Empty means "latest", which is what
# every download before 2026-09-23 took. When `provenance.check` refuses a file because upstream
# moved, find the version whose file matches the record on the dataset's Versions tab and pin it
# here; the Kaggle CLI takes `owner/dataset/versions/N`.
PINNED_VERSIONS: dict[str, int] = {}


def download(accept_new: bool = False) -> None:
    sys.path.insert(0, str(PROJ / "pipeline"))
    from provenance import check
    PUB.mkdir(parents=True, exist_ok=True)
    for ref in sorted({r for r, _ in SOURCES.values()}):
        dest = PUB / ref.replace("/", "_")
        dest.mkdir(exist_ok=True)
        spec = f"{ref}/versions/{PINNED_VERSIONS[ref]}" if ref in PINNED_VERSIONS else ref
        print(f"  {spec}")
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", spec, "-p", str(dest), "--unzip"],
                       check=True, stdout=subprocess.DEVNULL)
        # The targets (steven_v2) and the scoring reference (lixin_gpt56) come from here, so a
        # silently newer file would move both. Hash every file this script reads (AUDIT.md).
        for r, fname in SOURCES.values():
            if r == ref and (dest / fname).exists():
                check(dest / fname, spec, accept_new=accept_new)


def hanley_se(a: float, n_pos: int, n_neg: int) -> float:
    """SE of an AUC estimate (Hanley & McNeil 1982).

    Cheaper than the bootstrap in `diagnose.py` and close enough for a noise floor. It is
    here rather than there because this table needs an SE per *row*, and 9 rows x 12 labels
    of bootstrap is slower than the measurement is worth.
    """
    q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
    v = a * (1 - a) + (n_pos - 1) * (q1 - a ** 2) + (n_neg - 1) * (q2 - a ** 2)
    return float(np.sqrt(max(v, 0.0) / (n_pos * n_neg)))


def to_scores(d: pd.DataFrame) -> pd.DataFrame:
    """Class-probability tables (`ACL_0`, `ACL_1`, `ACL_9` = not said) -> one score per label,
    P(1) + 0.5 P(9): "the report does not say" sits between absent and present, as steven_v2's
    0.5 does. Tables that already carry the twelve columns pass through untouched."""
    if set(LABELS) <= set(d.columns) or not all(f"{lab}_1" in d.columns for lab in LABELS):
        return d
    out = d[["StudyInstanceUID"]].copy()
    for lab in LABELS:
        out[lab] = d[f"{lab}_1"] + 0.5 * d.get(f"{lab}_9", 0.0)
    return out


def load_gold() -> pd.DataFrame:
    g = pd.read_csv(D / "train.csv").dropna(subset=LABELS)
    return g.set_index("StudyInstanceUID")[LABELS].astype(float)


def align(path: Path, idx: pd.Index) -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = to_scores(pd.read_csv(path)).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    if not set(LABELS) <= set(d.columns):
        return None
    # 0.5 for a study the source never scored: neutral under a ranking metric, which is the
    # only thing that keeps a source with partial coverage from being scored as if it guessed.
    return d.reindex(idx)[LABELS].apply(pd.to_numeric, errors="coerce").fillna(0.5)


def score(pred: pd.DataFrame, gold: pd.DataFrame) -> tuple[float, float, dict]:
    per, ses = {}, []
    for lab in LABELS:
        y = gold[lab].values
        a = auc(y, pred[lab].values)
        per[lab] = a
        ses.append(hanley_se(a, int(y.sum()), int(len(y) - y.sum())))
    macro = float(np.mean(list(per.values())))
    return macro, float(np.sqrt(np.sum(np.square(ses))) / len(LABELS)), per


def rank_mean(sources: list[pd.DataFrame], idx: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)
    for lab in LABELS:
        out[lab] = np.mean([rankdata(s[lab].values) for s in sources], axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true", help="pull the public Datasets first")
    ap.add_argument("--accept-new", action="store_true",
                    help="let a download that differs from provenance/inputs.json replace the record")
    args = ap.parse_args()

    if args.download:
        download(accept_new=args.accept_new)

    gold = load_gold()
    idx = gold.index
    print(f"gold studies carrying all twelve labels: {len(idx)}\n")

    P: dict[str, pd.DataFrame] = {}
    ours = align(D / "report_labels_ours.csv", idx)
    if ours is not None:                  # the rule extractor; data/ was deleted 2026-09-07
        P["OURS (rules)"] = ours

    for name, (ref, fname) in SOURCES.items():
        p = align(PUB / ref.replace("/", "_") / fname, idx)
        if p is None:
            print(f"  (missing {name} -- re-run with --download)")
            continue
        d = to_scores(pd.read_csv(PUB / ref.replace("/", "_") / fname)).drop_duplicates("StudyInstanceUID")
        g_cov = float(idx.isin(d.StudyInstanceUID).mean())
        if g_cov < 0.9:
            # align() would fill the missing rows with 0.5 and print a fake 0.5000 macro
            print(f"  ({name}: carries {g_cov:.0%} of the gold rows -- NOT SCORABLE on gold, "
                  f"not a result of 0.5; its author held gold out)")
            continue
        P[name] = p

    rows = []
    for name, p in P.items():
        macro, se, per = score(p, gold)
        rows.append(dict(source=name, macro=macro, se=se, **per))
    t = pd.DataFrame(rows).set_index("source").sort_values("macro", ascending=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("=== single sources, macro AUROC on gold-58 ===")
    print("  every public read is a CEILING: a prompt tuned on these 58 rows leaves no trace")
    print(t[["macro", "se"]].round(4).to_string())

    # The two leak checks bench_external.py runs on the newer tables, applied to every source:
    # D4's "the teacher reads gold 0.8985" is a read of steven_v2 from this list, and nothing had
    # ever checked it. Warn, do not withhold -- withholding is for a proven copy (AUDIT.md).
    from bench_external import gold_rows_differ, leaks_gold
    tr = pd.read_csv(D / "train.csv")
    rest = pd.Index(tr.StudyInstanceUID[~tr.StudyInstanceUID.isin(idx)])
    G = gold.loc[idx, LABELS].to_numpy(float)

    def raw(path: Path, ids) -> np.ndarray | None:
        if not path.exists():
            return None
        d = to_scores(pd.read_csv(path)).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        return d.reindex(ids)[LABELS].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    for name, (ref, fname) in SOURCES.items():
        path = PUB / ref.replace("/", "_") / fname
        vg, vn = raw(path, idx), raw(path, rest)
        if vg is None:
            continue
        if leaks_gold(np.nan_to_num(vg, nan=0.5), G):
            print(f"  ⛔ {name}: GOLD IS IN THIS FILE -- its score above is not a result")
        elif why := gold_rows_differ(vg, vn):
            print(f"  ⚠ {name} SUSPECT: {why}")
        if name in SELECTED_ON_GOLD:
            print(f"  ⚠ {name} FITTED TO GOLD: {SELECTED_ON_GOLD[name]} -- its read is a ceiling")
    print("\n=== per label ===")
    print(t[LABELS].round(3).to_string())

    # PAIRED against the training targets, on the same 58 studies: resample studies once and score
    # both sources on the resample, so the shared difficulty cancels (score_oof.py's reasoning).
    base = "steven_v2"
    if base in P:
        rng = np.random.default_rng(0)
        draws = [rng.integers(0, len(idx), len(idx)) for _ in range(2000)]

        def mac(p: pd.DataFrame, k) -> float:
            v = [auc(gold[lab].values[k], p[lab].values[k]) for lab in LABELS]
            return float(np.nanmean(v))

        print(f"\n=== paired against {base} (the training targets), gold-58, 2,000 study resamples ===")
        for name in t.index:
            if name == base:
                continue
            d0 = mac(P[name], np.arange(len(idx))) - mac(P[base], np.arange(len(idx)))
            ds = np.array([mac(P[name], k) - mac(P[base], k) for k in draws])
            flag = "  (fitted to gold)" if name in SELECTED_ON_GOLD else ""
            print(f"  {name:14s} {d0:+.4f} +-{ds.std():.4f}   P(better) {(ds > 0).mean():.3f}{flag}")

    # JEV's trick -- each finding takes whichever table reads it best -- is selected ON gold, so its
    # own read cannot say whether it transfers. Split-half: pick per label on a random 29, score
    # the composite on the other 29, against the training targets on the same 29. 2,000 splits.
    # This is RUNBOOK §3 rule 1 made into a measurement instead of a warning.
    pool = [n for n in ("steven_v2", "steven_v4", "pilkwang_v2", "jev_text", "hosseini_jev") if n in P]
    if base in P and len(pool) > 1:
        rng = np.random.default_rng(1)
        half = len(idx) // 2
        deltas, picks = [], {lab: {} for lab in LABELS}
        for _ in range(2000):
            perm = rng.permutation(len(idx))
            a_, b_ = perm[:half], perm[half:]
            comp, ok = [], True
            for lab in LABELS:
                y = gold[lab].values
                if y[a_].min() == y[a_].max() or y[b_].min() == y[b_].max():
                    ok = False
                    break
                best = max(pool, key=lambda n: auc(y[a_], P[n][lab].values[a_]))
                picks[lab][best] = picks[lab].get(best, 0) + 1
                comp.append(auc(y[b_], P[best][lab].values[b_]) - auc(y[b_], P[base][lab].values[b_]))
            if ok:
                deltas.append(np.mean(comp))
        deltas = np.asarray(deltas)
        print(f"\n=== per-label pick chosen on half of gold, scored on the other half ({len(deltas)} splits) ===")
        print(f"  pool {pool}")
        print(f"  composite - {base}, held-out half: {deltas.mean():+.4f} +-{deltas.std():.4f}   "
              f"P(better) {(deltas > 0).mean():.3f}")
        for lab in LABELS:
            tot = sum(picks[lab].values())
            top = sorted(picks[lab].items(), key=lambda kv: -kv[1])[:2]
            print(f"    {lab:18s} " + "  ".join(f"{n} {c / tot:.0%}" for n, c in top))

    # What the SCORING reference can and cannot judge: score_oof.py reads lixin_gpt56, so a
    # training table close to it wins there by construction (steven_v4 -> lixin AUC 0.9998).
    ref_name = "lixin_gpt56"
    tables = {n: raw(PUB / r.replace("/", "_") / f, rest) for n, (r, f) in SOURCES.items()}
    tables = {n: v for n, v in tables.items() if v is not None}
    if ref_name in tables:
        print(f"\n=== agreement with {ref_name} (the OOF scoring reference) and coverage, non-gold rows ===")
        from scipy.stats import spearmanr
        R = tables[ref_name]
        for n, V in tables.items():
            rs = [spearmanr(V[:, j], R[:, j], nan_policy="omit").statistic for j in range(len(LABELS))]
            cov = float(np.isfinite(V).all(axis=1).mean())
            print(f"  {n:14s} mean Spearman vs {ref_name} {np.nanmean(rs):.3f}   coverage {cov:.1%}")

    pub = [k for k in P if k != "OURS (rules)"]
    if "OURS (rules)" not in P or len(pub) < 2:
        return

    print("\n=== rank-mean combinations: does ours add anything? ===")
    combos = [("all public", pub), ("all public + ours", pub + ["OURS (rules)"])]
    best2 = list(t.index[t.index != "OURS (rules)"][:2])
    combos += [(" + ".join(best2), best2), (" + ".join(best2 + ["ours"]), best2 + ["OURS (rules)"])]
    for name, keys in combos:
        macro, se, _ = score(rank_mean([P[k] for k in keys], idx), gold)
        print(f"  {name:42s} {macro:.4f}")

    print("\n=== per-label head-to-head: ours vs the best public reader ===")
    wins = 0
    for lab in LABELS:
        o = t.loc["OURS (rules)", lab]
        b = t.loc[pub, lab]
        if o > b.max():
            wins += 1
        print(f"  {lab:18s} ours {o:.3f}   best public {b.max():.3f} ({b.idxmax()})"
              f"{'   <-- OURS WINS' if o > b.max() else ''}")
    print(f"\n  labels where the rule extractor wins: {wins}/12")


if __name__ == "__main__":
    main()
