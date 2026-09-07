"""Everything `DATA.md` claims about the label stage, re-measured from the CSVs.

`bench_public_labels.py` ranks label *sources*. This module asks the prior question — **what
is actually wrong with the labels we have, and which instrument can see it** — because the
2026-09-06 version of `DATA.md` answered it with two numbers that do not survive contact with
their own definition:

  - the "6.68 : 1 over-call" is a **binarisation threshold**, not a property of the labels
    (§1 below sweeps it from 29.2:1 to 0.09:1 on one unchanged file), and
  - the "79.0% -> 84.7% cell-agreement headroom" is that same artifact read from the other
    side: at the prevalence-matched cut the public labels already sit at 84.8%.

AUROC — the only thing the competition reads — is invariant across every row of that sweep.
So both numbers are measurements aimed at a question they cannot answer, which `DECISIONS.md`
names as this project's repeating failure. What replaces them is §3: on the 30 studies that
are both gold-labelled and hand-read, a careful human beats the public reader by **+0.153**
like-for-like, and graded output is worth **+0.184** on top of that. Those two are the route.

Sections, each independently runnable via --only:

  1  overcall     the D1 threshold sweep
  2  headroom     per-label AUROC of the shipped targets, and what macro 0.93 costs
  3  ceiling      human vs public on the gold ∩ hand studies, with the binarisation control
  4  power        marginal vs paired bootstrap on gold-58; the McNemar discordant-cell table
  5  proxy        is the image-model OOF a usable large-n stand-in for gold? (it is not)
  6  addressable  per label: does the report text even carry the graded quantity?
  7  language     public vs hand read, by language — where the reader is weakest

Run:
    python extractor/audit_label_stage.py                # all seven
    python extractor/audit_label_stage.py --only 3 4     # the two that set the gate

Needs `data/train.csv`, `data/targets.csv`, `data/lang_detected.csv`,
`labeling/model_labels.csv`, and for §5 `data/external/pilkwang_weights/oof.npz`.
Sections whose inputs are missing say so and are skipped; none of them writes anything.
"""
import argparse
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import auc  # noqa: E402  -- shared definition, do not re-implement

PROJ = Path(__file__).resolve().parent.parent
D = PROJ / "data"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

N_BOOT = 4000
SEED = 20260906


def gold() -> pd.DataFrame:
    g = pd.read_csv(D / "train.csv").dropna(subset=LABELS)
    return g.set_index("StudyInstanceUID")[LABELS].astype(float)


def table(path: Path, idx: pd.Index) -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_csv(path).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    if not set(LABELS) <= set(d.columns):
        return None
    return d.reindex(idx)[LABELS].apply(pd.to_numeric, errors="coerce")


def macro(Y: np.ndarray, P: np.ndarray) -> float:
    """Macro AUROC, skipping any label that is degenerate in this (re)sample."""
    return float(np.nanmean([auc(Y[:, k], P[:, k]) if len(set(Y[:, k])) > 1 else np.nan
                             for k in range(len(LABELS))]))


def prevalence_match(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Binarise each label at its own gold prevalence — the cut that flatters a source most.

    Used as a control, never as a proposal: §1 is the whole point that this choice is free.
    """
    B = np.zeros_like(P)
    for k in range(P.shape[1]):
        B[np.argsort(-P[:, k])[:int(Y[:, k].sum())], k] = 1
    return B


# ----------------------------------------------------------------- 1
def s1_overcall(G, P):
    print("The same file, the same AUROC, ten different 'over-call ratios'.\n")
    Y, V = G.values, np.nan_to_num(P.values, nan=0.5)
    print(f"  macro AUROC, unchanged by every row below: {macro(Y, V):.4f}\n")
    print(f"  {'cut':>6}  {'FP':>5} {'FN':>4}  {'over-call':>10}  {'cell agree':>10}")
    for th in (0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        B = V >= th
        fp, fn = int((B & (Y == 0)).sum()), int((~B & (Y == 1)).sum())
        print(f"  {th:>6}  {fp:5d} {fn:4d}  {fp / max(fn, 1):>8.2f}:1  "
              f"{1 - (fp + fn) / Y.size:>10.3f}")
    B = prevalence_match(V, Y).astype(bool)
    fp, fn = int((B & (Y == 0)).sum()), int((~B & (Y == 1)).sum())
    print(f"  {'prev':>6}  {fp:5d} {fn:4d}  {fp / max(fn, 1):>8.2f}:1  "
          f"{1 - (fp + fn) / Y.size:>10.3f}")
    print("\n  DATA.md's 6.68:1 is the 0.60 row. Its '79.0% cell agreement' is the same row.")
    print("  The prevalence row reads 84.8% — at or above the 84.7% called the human ceiling.")
    print("  => the over-call gate is satisfiable by editing a constant. It cannot move AUROC.")


# ----------------------------------------------------------------- 2
def s2_headroom(G, P):
    Y, V = G.values, np.nan_to_num(P.values, nan=0.5)
    per = pd.Series({l: auc(Y[:, k], V[:, k]) for k, l in enumerate(LABELS)}).sort_values()
    npos = G.sum().astype(int)
    print(f"  {'label':18s} {'n_pos':>5} {'AUROC':>7}")
    for l, a in per.items():
        print(f"  {l:18s} {npos[l]:5d} {a:7.3f}")
    m = per.mean()
    weak, strong = per.index[:6], per.index[6:]
    need = (0.93 * 12 - per[strong].sum()) / 6
    print(f"\n  macro {m:.4f}.  To reach 0.93 the twelve must gain {12 * (0.93 - m):.3f} in total.")
    print(f"  The top six are already {per[strong].mean():.3f} and have nowhere to go, so the")
    print(f"  bottom six must move {per[weak].mean():.3f} -> {need:.3f}."
          f"  That is the whole job, and it is six labels wide.")


# ----------------------------------------------------------------- 3
def s3_ceiling(G, P):
    hand = pd.read_csv(PROJ / "labeling" / "model_labels.csv")
    hand = hand.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    ids = [i for i in G.index if i in hand.index]
    if len(ids) < 10:
        print(f"  only {len(ids)} studies are both gold and hand-read — skipping")
        return
    Y = G.loc[ids].values
    H = hand.loc[ids][LABELS].astype(float).values
    V = np.nan_to_num(P.loc[ids].values, nan=0.5)
    Vb = prevalence_match(V, Y)

    print(f"  n = {len(ids)} studies that are both gold-labelled and hand-read.\n")
    print(f"    human, binary                    {macro(Y, H):.4f}")
    print(f"    public, binarised to match       {macro(Y, Vb):.4f}")
    print(f"    public, as shipped (soft)        {macro(Y, V):.4f}")

    rng = np.random.default_rng(SEED)
    read, soft = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(ids), len(ids))
        read.append(macro(Y[i], H[i]) - macro(Y[i], Vb[i]))     # reading skill
        soft.append(macro(Y[i], V[i]) - macro(Y[i], Vb[i]))     # graded output
    for name, b in (("reading skill  (human - public, both binary)", np.array(read)),
                    ("graded output  (public soft - public binary)", np.array(soft))):
        b = b[~np.isnan(b)]
        print(f"\n  {name}")
        print(f"    {b.mean():+.4f}   95% [{np.percentile(b, 2.5):+.4f}, "
              f"{np.percentile(b, 97.5):+.4f}]   P(>0) = {(b > 0).mean():.3f}")
    print("\n  Both are real and they are separable. The public reader wins today only because")
    print("  it is graded and our hand export was binarised. Build a reader that is both.")


# ----------------------------------------------------------------- 4
def s4_power(G, P):
    alt = D / "public_llm_labels" / "stevenleehans_rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv"
    Q = table(alt, G.index)
    if Q is None:
        print("  need llm_labels_v4_blend.csv as a stand-in challenger — skipping")
        return
    Y = G.values
    A, B = np.nan_to_num(P.values, nan=0.5), np.nan_to_num(Q.values, nan=0.5)
    rng = np.random.default_rng(SEED)
    marg, pair = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(Y), len(Y))
        a, b = macro(Y[i], A[i]), macro(Y[i], B[i])
        marg.append(a)
        pair.append(b - a)
    marg, pair = np.array(marg), np.array(pair)
    print(f"  marginal SD of one macro   {marg.std():.4f}   (95% CI width {4 * marg.std():.4f})")
    print(f"  paired   SD of a delta     {pair.std():.4f}   (95% CI width {4 * pair.std():.4f})")
    print(f"  the pairing is worth {marg.std() / pair.std():.2f}x, and makes"
          f" {1.96 * pair.std():+.4f} the smallest honest gate.\n")

    Ab, Bb = prevalence_match(A, Y), prevalence_match(B, Y)
    d = Ab != Bb
    nb, na = int(((Bb == Y) & d).sum()), int(((Ab == Y) & d).sum())
    print(f"  cells where two label sets disagree: {int(d.sum())} of {Y.size}"
          f"  ({nb} new-right / {na} old-right)")
    try:
        from scipy.stats import binomtest
        print("\n  discordant cells needed for a 2:1 win to clear p<0.05:")
        for m in (20, 30, 40, 60, 80):
            k = round(m * 2 / 3)
            print(f"    {m:3d} discordant, {k}:{m - k} -> p = {binomtest(k, m, 0.5).pvalue:.4f}")
    except ImportError:
        pass


# ----------------------------------------------------------------- 5
def s5_proxy(G, _P):
    z = D / "external" / "pilkwang_weights" / "oof.npz"
    if not z.exists():
        print("  no pilkwang oof.npz — skipping")
        return
    npz = np.load(z, allow_pickle=True)
    ids = pd.Index(npz["ids"])
    oof = pd.DataFrame(npz["pred"], index=ids, columns=LABELS)
    nongold = ids[~pd.Series(npz["gold_mask"], index=ids).values]

    pub = D / "public_llm_labels"
    srcs = {
        "steven_v4": pub / "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        "steven_v2": pub / "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv",
        "steven_full": pub / "stevenleehans_rsna-knee-llm-report-labels/llm_labels_full.csv",
        "pilkwang_v2": pub / "pilkwang_rsna-knee-llm-labels/report_labels_v2.csv",
        "pilkwang_v1": pub / "pilkwang_rsna-knee-report-labels/report_labels_v1.csv",
        "lixin_gpt56": pub / "lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv",
        "ours_rules": D / "report_labels_ours.csv",
    }
    rows = []
    for name, p in srcs.items():
        g = table(p, G.index)
        s = table(p, nongold)
        if g is None or s is None:
            continue
        prox = []
        for l in LABELS:
            v, o = s[l].values, oof.loc[nongold, l].values
            m = ~np.isnan(v)
            y = (v[m] >= 0.5).astype(int)
            prox.append(auc(y, o[m]) if 0 < y.sum() < len(y) else np.nan)
        rows.append(dict(source=name,
                         gold58=macro(G.values, np.nan_to_num(g.values, nan=0.5)),
                         proxy=float(np.nanmean(prox))))
    t = pd.DataFrame(rows).sort_values("gold58", ascending=False)
    print(t.round(4).to_string(index=False))
    try:
        from scipy.stats import spearmanr
        r = spearmanr(t.gold58, t.proxy)
        print(f"\n  Spearman over {len(t)} label sources: rho = {r.statistic:.3f}, p = {r.pvalue:.3f}")
    except ImportError:
        pass
    print("  The image model was trained on pilkwang's labels and ranks them first, so this")
    print("  instrument scores agreement-with-its-own-teacher. NOT usable as the Stage L gate.")


# ----------------------------------------------------------------- 6
ABOVE_CUT = {  # words that put a finding ABOVE an official cut-point
    "english": ["moderate", "large", "severe", "marked", "extensive", "complete",
                "full-thickness", "full thickness", "high-grade", "high grade"],
    "spanish": ["moderad", "sever", "import", "marcad", "extens", "complet", "grande"],
    "turkish": ["orta derece", "ileri", "belirgin", "yaygin", "tam kat", "buyuk", "komplet"],
    "croatian": ["umjeren", "izrazit", "tesk", "opsezn", "potpun", "velik", "kompletn"],
    "greek": ["metri", "sovar", "ektetam", "plhr", "megal"],
    "german": ["massig", "maessig", "ausgepragt", "schwer", "deutlich", "komplett", "hochgradig"],
    "bulgarian": ["umeren", "izrazen", "teshk", "goljam", "pulen"],
    "dutch": ["matig", "ernstig", "uitgebreid", "fors", "volledig", "hooggradig"],
    "french": ["moder", "sever", "import", "marque", "etendu", "complet"],
}
ACUITY = {
    "english": ["acute", "chronic", "old", "remote", "subacute"],
    "spanish": ["agud", "cronic", "antigu", "recient"],
    "turkish": ["akut", "kronik", "eski", "yeni"],
    "croatian": ["akutn", "kronicn", "star", "svjez"],
    "greek": ["oxei", "xroni", "palai", "prosfat"],
    "german": ["akut", "chronisch", "alt", "frisch"],
    "bulgarian": ["akut", "hronich", "star"],
    "dutch": ["acuut", "chronisch", "oud", "recent"],
    "french": ["aigu", "chronique", "ancien", "recent"],
}
WINDOW = 90  # chars either side of a finding mention


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).lower()
    return "".join(c for c in s if not unicodedata.combining(c))


def s6_addressable(*_):
    import json
    import re
    gl = json.load(open(PROJ / "labeling" / "glossary.json"))
    tr = pd.read_csv(D / "train.csv")
    lg = pd.read_csv(D / "lang_detected.csv")[["StudyInstanceUID", "lang"]]
    tr = tr.merge(lg, on="StudyInstanceUID", how="left")
    tr["t"] = tr.Report.map(_fold)

    fold = lambda d: {k: [_fold(x) for x in v] for k, v in d.items()}  # noqa: E731
    find = {k: fold(v) for k, v in gl["findings"].items()}
    below, above, acu = fold(gl["cues"]["minimal"]), fold(ABOVE_CUT), fold(ACUITY)
    num = re.compile(r"\d+[.,]?\d*\s*(mm|cm)\b")

    print(f"  {'label':18s} {'mention':>8} {'graded':>7} {'below':>7} {'above':>7}"
          f" {'acuity':>7} {'measured':>9}")
    for lab in LABELS:
        terms = dict(find.get(lab, {}))
        if lab.endswith("OA"):  # the glossary keeps one generic OA vocabulary
            for k, v in find.get("_OA_generic", {}).items():
                terms[k] = terms.get(k, []) + v
        n = m = b = a = ac = nm = 0
        for txt, lang in zip(tr.t, tr.lang):
            lang = lang if isinstance(lang, str) else "english"
            n += 1
            tt = terms.get(lang, []) + terms.get("english", [])
            hits = [h.start() for x in tt for h in re.finditer(re.escape(x), txt)]
            if not hits:
                continue
            m += 1
            ctx = " ".join(txt[max(0, h - WINDOW):h + WINDOW] for h in hits[:8])
            b += any(x in ctx for x in below.get(lang, []) + below["english"])
            a += any(x in ctx for x in above.get(lang, []) + above["english"])
            ac += any(x in ctx for x in acu.get(lang, []) + acu["english"])
            nm += bool(num.search(ctx))
        d = max(m, 1)
        print(f"  {lab:18s} {m / n:8.3f} {(b + a) / d:7.3f} {b / d:7.3f} {a / d:7.3f}"
              f" {ac / d:7.3f} {nm / d:9.3f}")
    print("\n  'mention' bounds every label from above: an extractor cannot rank what the report")
    print("  never discusses. Synovitis at ~0.12 is not a reading problem, it is an absence one.")

    # NFKD leaves 'ß' intact, so the German 'mäßig' folds to 'maßig' and a plain
    # "massig" test misses it. Match on the stem that survives both spellings.
    moderate = ("massig", "maßig", "maessig", "moder", "matig", "umeren", "metri")
    bad = {lg_: [w for w in ws if any(w.startswith(x[:4]) for x in moderate)]
           for lg_, ws in below.items()}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        print(f"\n  ⚠ glossary bug — 'moderate' sits in the MINIMAL cue list for {bad}.")
        print("    Moderate is ABOVE the cut for Effusion and Baker's, so the cue set inverts")
        print("    the criterion in those languages. Fix before any cue-driven extraction.")


# ----------------------------------------------------------------- 7
def s7_language(G, P):
    hand = pd.read_csv(PROJ / "labeling" / "model_labels.csv")
    hand = hand.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    lg = pd.read_csv(D / "lang_detected.csv").set_index("StudyInstanceUID")["lang"]
    full = table(D / "targets.csv", hand.index)
    if full is None:
        print("  no data/targets.csv — skipping")
        return
    keep = full.notna().all(axis=1)
    H = hand.loc[keep[keep].index][LABELS].astype(float).values
    V = (full.loc[keep[keep].index].values >= 0.6).astype(float)
    df = pd.DataFrame({
        "lang": lg.reindex(keep[keep].index).values,
        "agree": (H == V).mean(1),
        "over": ((V == 1) & (H == 0)).mean(1),
        "under": ((V == 0) & (H == 1)).mean(1),
    })
    out = df.groupby("lang").agg(n=("agree", "size"), cell_agree=("agree", "mean"),
                                 over=("over", "mean"), under=("under", "mean"))
    print(out.round(3).sort_values("n", ascending=False).to_string())
    print("\n  Small n per language — directional. The pattern to act on is that the reader's")
    print("  errors are one-directional (over) everywhere, and worst outside English.")


SECTIONS = {
    1: ("the D1 over-call is a threshold, not a property", s1_overcall),
    2: ("per-label AUROC, and what macro 0.93 actually costs", s2_headroom),
    3: ("the ceiling: reading skill vs graded output", s3_ceiling),
    4: ("gate power: pair the comparison, or measure nothing", s4_power),
    5: ("is the image-model OOF a usable large-n proxy?", s5_proxy),
    6: ("does the report carry the graded quantity at all?", s6_addressable),
    7: ("where the public reader is weakest, by language", s7_language),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", type=int, nargs="+", choices=sorted(SECTIONS),
                    help="run just these sections")
    args = ap.parse_args()

    G = gold()
    P = table(D / "targets.csv", G.index)
    if P is None:
        sys.exit("data/targets.csv not found — see README 'Regenerating derived data'")
    print(f"gold studies: {len(G)}   incumbent targets: data/targets.csv\n")

    for n in (args.only or sorted(SECTIONS)):
        title, fn = SECTIONS[n]
        print(f"\n{'=' * 78}\n§{n}  {title}\n{'=' * 78}")
        try:
            fn(G, P)
        except FileNotFoundError as e:
            print(f"  missing input, skipped: {e}")


if __name__ == "__main__":
    main()
