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
}


def download() -> None:
    PUB.mkdir(parents=True, exist_ok=True)
    for ref in sorted({r for r, _ in SOURCES.values()}):
        dest = PUB / ref.replace("/", "_")
        dest.mkdir(exist_ok=True)
        print(f"  {ref}")
        subprocess.run(["kaggle", "datasets", "download", ref, "-p", str(dest), "--unzip"],
                       check=True, stdout=subprocess.DEVNULL)


def hanley_se(a: float, n_pos: int, n_neg: int) -> float:
    """SE of an AUC estimate (Hanley & McNeil 1982).

    Cheaper than the bootstrap in `diagnose.py` and close enough for a noise floor. It is
    here rather than there because this table needs an SE per *row*, and 9 rows x 12 labels
    of bootstrap is slower than the measurement is worth.
    """
    q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
    v = a * (1 - a) + (n_pos - 1) * (q1 - a ** 2) + (n_neg - 1) * (q2 - a ** 2)
    return float(np.sqrt(max(v, 0.0) / (n_pos * n_neg)))


def load_gold() -> pd.DataFrame:
    g = pd.read_csv(D / "train.csv").dropna(subset=LABELS)
    return g.set_index("StudyInstanceUID")[LABELS].astype(float)


def align(path: Path, idx: pd.Index) -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_csv(path).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
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
    args = ap.parse_args()

    if args.download:
        download()

    gold = load_gold()
    idx = gold.index
    print(f"gold studies carrying all twelve labels: {len(idx)}\n")

    P: dict[str, pd.DataFrame] = {}
    ours = align(D / "report_labels_ours.csv", idx)
    if ours is None:
        sys.exit("data/report_labels_ours.csv not found -- run extractor/run_extract.py first")
    P["OURS (rules)"] = ours

    for name, (ref, fname) in SOURCES.items():
        p = align(PUB / ref.replace("/", "_") / fname, idx)
        if p is None:
            print(f"  (missing {name} -- re-run with --download)")
            continue
        P[name] = p

    rows = []
    for name, p in P.items():
        macro, se, per = score(p, gold)
        rows.append(dict(source=name, macro=macro, se=se, **per))
    t = pd.DataFrame(rows).set_index("source").sort_values("macro", ascending=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("=== single sources, macro AUROC on gold-58 ===")
    print(t[["macro", "se"]].round(4).to_string())
    print("\n=== per label ===")
    print(t[LABELS].round(3).to_string())

    pub = [k for k in P if k != "OURS (rules)"]
    if len(pub) < 2:
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
