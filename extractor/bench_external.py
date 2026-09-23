"""Score the label sources we did NOT have on 2026-09-06, plus the frontier's own extractor.

`bench_public_labels.py` ranks the six tables surveyed on 2026-08-10. That survey was 27 days
stale when this was written, and re-running the search found **four more label tables published
since**, one of them from a different model family. It also found that the thing `DATA.md`
proposed to build — a severity-graded, multilingual, directional-negation report extractor —
**is already implemented inside the public frontier notebook sitting in
`data/external/kernel_sources/`**, and has been since 2026-08-13.

Two results worth knowing before reading any of it:

  - **The frontier's rule extractor scores 0.8617 on gold-58, BELOW our incumbent's 0.8873**
    (paired −0.0255, P(better)=0.009). Its own `read_labels()` refuses to fall back to it,
    calling it "the weaker labels". It wins only on the acuity labels — Fracture +0.040,
    Contusion +0.026, Lateral OA +0.010 — which is exactly where §6 of the audit says the
    reports carry acuity language. Complementary, not dominant.
  - **Two of the four new tables have the gold labels baked in.** They match all 58 gold rows
    exactly and are 0/1-valued there while carrying 506 distinct values elsewhere, so they
    score a perfect 1.0000 on gold-58. That is a leak, not a result, and anyone benchmarking
    them naively gets a miracle. `--check-leak` is on by default and refuses to report a score
    for a source that fails it.

Run:
    python extractor/bench_external.py --download   # pulls the four new tables
    python extractor/bench_external.py

Scores everything three ways: gold-58 (leak-checked), the 53 non-gold hand-read studies (which
no public table can have leaked into), and paired bootstrap deltas against the incumbent.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import auc  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
D = PROJ / "data"
PUB = D / "public_llm_labels"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Published AFTER the 2026-08-10 survey. `qwen3` is the interesting one: every table we held
# was a GPT-family read, and §2g measured those as near-duplicates (|r| 0.87-0.95), so a
# different model family is worth more to an ensemble than a seventh near-copy.
NEW_SOURCES = {
    "rayanbabur_soft": ("rayanbabur/rsna-knee-2026-calibrated-soft-targets",
                        "train_targets_soft.csv"),
    "yunus_v5_merged": ("yunusgmsoy/rsna-knee-llm-labels-4-source-merged",
                        "report_labels_v5.csv"),
    "yunus_llm": ("yunusgmsoy/rsna-knee-llm-report-labels", "yunus_llm_labels.csv"),
    "qwen3_8b": ("laymond/rsna-knee-abnormality-qwen3-8b-weak-labels",
                 "qwen_knee_weak_labels.csv"),
}
INCUMBENT = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v2.csv")
STEVEN_V4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")

# The frontier's self-contained extractor is cells 2-4 of its notebook: pure `re` +
# `unicodedata`, no other imports. It ends on the `return out` of `extract()`.
FRONTIER_SRC = D / "external" / "kernel_sources" / \
    "mattiaangeli_bend-the-knee-to-dinov3-ensembled.py"
FRONTIER_END = "    return out\n# ===== CELL 5 ====="

N_BOOT = 4000
SEED = 20260906


def download() -> None:
    for ref, _ in NEW_SOURCES.values():
        dest = PUB / ref.replace("/", "_")
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  {ref}")
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", ref,
                        "-p", str(dest), "--unzip", "-q"], check=False)


def load_frontier():
    """Import the frontier's `extract()` without executing the rest of its notebook."""
    if not FRONTIER_SRC.exists():
        return None
    text = FRONTIER_SRC.read_text()
    cut = text.find(FRONTIER_END)
    if cut < 0:
        return None
    ns: dict = {}
    exec(compile(text[:cut + len("    return out")], str(FRONTIER_SRC), "exec"), ns)
    return ns.get("extract")


def macro(Y: np.ndarray, V: np.ndarray) -> float:
    return float(np.nanmean([auc(Y[:, k], V[:, k]) if len(set(Y[:, k])) > 1 else np.nan
                             for k in range(len(LABELS))]))


def paired(Y, A, B, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(Y), len(Y))
        try:
            out.append(macro(Y[i], B[i]) - macro(Y[i], A[i]))
        except Exception:
            pass
    b = np.array(out)
    b = b[~np.isnan(b)]
    return b.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), (b > 0).mean()


def align(path: Path, idx, fill: bool = True) -> np.ndarray | None:
    if not path.exists():
        return None
    d = pd.read_csv(path).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    if not set(LABELS) <= set(d.columns):
        return None
    v = d.reindex(idx)[LABELS].apply(pd.to_numeric, errors="coerce")
    return v.fillna(0.5).values if fill else v.values


def leaks_gold(V: np.ndarray, G: np.ndarray) -> bool:
    """A table that reproduces gold exactly, in 0/1, has gold written into it."""
    return bool((V == G).mean() > 0.999 and np.isin(V, [0.0, 1.0]).mean() > 0.999)


# How far gold rows may sit from the rest of a table before it is called SUSPECT. The two tables
# that copied gold read ~1.0 binary on gold rows against a soft table elsewhere (506 distinct
# values); an honest reader of 58 reports has no reason to write them any differently from 4,349.
SUSPECT_BINARY_GAP = 0.30        # share of cells exactly 0 or 1, gold rows minus the rest
SUSPECT_CONFIDENCE_GAP = 0.15    # mean |p - 0.5|, gold rows minus the rest


def gold_rows_differ(Vg: np.ndarray, Vn: np.ndarray) -> str:
    """'' if gold rows look like the rest of the table, else a one-line reason.

    `leaks_gold` only catches a table that COPIED gold. A table that blended it in, softened it to
    0.05/0.95, or was otherwise written differently on exactly these 58 rows passes that check and
    reads optimistic -- and gold-58 is the only truth any competitor has, so fitting to it is the
    likely leak, not the exotic one. This catches the ones that leave a trace in the values.
    What it cannot catch is a prompt or threshold TUNED on gold-58: that leaves no trace, so every
    third-party gold-58 read is a ceiling, never an estimate (AUDIT.md).
    """
    g, n = Vg[np.isfinite(Vg)], Vn[np.isfinite(Vn)]
    if not len(g) or not len(n):
        return ""
    bin_g, bin_n = np.isin(g, [0.0, 1.0]).mean(), np.isin(n, [0.0, 1.0]).mean()
    conf_g, conf_n = np.abs(g - 0.5).mean(), np.abs(n - 0.5).mean()
    why = []
    if bin_g - bin_n > SUSPECT_BINARY_GAP:
        why.append(f"0/1 cells {bin_g:.0%} on gold rows vs {bin_n:.0%} elsewhere")
    if conf_g - conf_n > SUSPECT_CONFIDENCE_GAP:
        why.append(f"mean |p-0.5| {conf_g:.2f} on gold rows vs {conf_n:.2f} elsewhere")
    return "; ".join(why)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()

    tr = pd.read_csv(D / "train.csv")
    gold = tr.dropna(subset=LABELS).set_index("StudyInstanceUID")[LABELS].astype(float)
    G = gold.values

    sources = {"INCUMBENT steven_v2": INCUMBENT, "steven_v4_blend": STEVEN_V4, **NEW_SOURCES}
    V: dict[str, np.ndarray] = {}
    leaked: set[str] = set()

    print(f"=== gold-58, leak-checked  (n={len(gold)}) ===")
    print("  every read below is a CEILING: a prompt tuned on these 58 rows leaves no trace")
    nongold = pd.Index(tr.StudyInstanceUID[~tr.StudyInstanceUID.isin(gold.index)])
    for name, (ref, fname) in sources.items():
        path = PUB / ref.replace("/", "_") / fname
        a = align(path, gold.index)
        if a is None:
            print(f"  {name:22s} -- not present or wrong schema (try --download)")
            continue
        V[name] = a
        if leaks_gold(a, G):
            leaked.add(name)
            print(f"  {name:22s}   ⛔ GOLD IS IN THIS FILE — score withheld")
            continue
        why = gold_rows_differ(align(path, gold.index, fill=False),
                               align(path, nongold, fill=False))
        print(f"  {name:22s}   {macro(G, a):.4f}" + (f"   ⚠ SUSPECT: {why}" if why else ""))

    ext = load_frontier()
    if ext is not None:
        F = pd.DataFrame([ext(r) for r in tr.Report])
        F.insert(0, "StudyInstanceUID", tr.StudyInstanceUID.values)
        V["FRONTIER rule extractor"] = (F.set_index("StudyInstanceUID")
                                        .reindex(gold.index)[LABELS].astype(float).values)
        print(f"  {'FRONTIER rule extractor':22s}   {macro(G, V['FRONTIER rule extractor']):.4f}"
              f"   (zero API cost, severity-graded, 9 languages)")

    base = V.get("INCUMBENT steven_v2")
    if base is None:
        return

    print("\n=== paired vs the incumbent, gold-58 (leaked sources omitted) ===")
    for name, a in V.items():
        if name.startswith("INCUMBENT") or name in leaked:
            continue
        m, lo, hi, p = paired(G, base, a)
        print(f"  {name:24s} {m:+.4f}  95% [{lo:+.4f},{hi:+.4f}]  P(>0)={p:.3f}")

    hand_p = PROJ / "labeling" / "model_labels.csv"
    if not hand_p.exists():
        return
    hand = pd.read_csv(hand_p).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    ids = [i for i in hand.index if i not in set(gold.index)]
    H = hand.loc[ids][LABELS].astype(float).values
    print(f"\n=== the {len(ids)} non-gold hand-read studies — private, nothing can leak in ===")
    HV = {}
    for name, (ref, fname) in sources.items():
        a = align(PUB / ref.replace("/", "_") / fname, pd.Index(ids))
        if a is not None:
            HV[name] = a
            print(f"  {name:22s}   {macro(H, a):.4f}")
    print("\n  ⚠ every source reads 0.98-0.99 here. The hand set measures agreement with a")
    print("    careful REPORT read, and that is nearly solved — it discriminates weakly.")
    print("    To make it bite, label where the sources DISAGREE, not stratified by language.")
    hb = HV.get("INCUMBENT steven_v2")
    if hb is not None:
        print()
        for name, a in HV.items():
            if name.startswith("INCUMBENT"):
                continue
            m, lo, hi, p = paired(H, hb, a)
            print(f"  paired {name:22s} {m:+.4f}  95% [{lo:+.4f},{hi:+.4f}]  P(>0)={p:.3f}")


if __name__ == "__main__":
    main()
