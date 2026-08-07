"""Head-to-head between every available label source. The week-2 decision gate (PLAN.md 7).

`nekkon/weak-labels-for-all-12-knee-mri-findings` is public, so "the extractor is the solution"
(PLAN.md 5) is now a claim that has to be earned rather than assumed. This scores each source
against the same references and answers one question: does ours beat the free one?

Sources (whichever exist):
  rules   data/pseudo_labels.csv        5-state soft targets, 0.95/0.65/0.45/0.03/0.08
  llm     data/llm_pseudo_labels.csv    same shape, from llm_extract.py
  public  data/public_weak_labels.csv   binary 0/1  -- fetch with --fetch
  hand    labeling/model_labels.csv     binary, 86 studies, our own reading

References:
  gold    the 58 studies with expert image-derived labels
  hand    our 86 hand labels (as reference for anything that is not a source)

TWO METRICS, and the gap between them is the point:

  AUC       rewards ranking. Soft targets can separate a confident call from a hedged one;
            binary labels cannot, so their AUC is pinned to balanced accuracy. This is the
            metric that matters downstream -- the competition scores macro AUC, and a vision
            model trained on graded targets inherits the gradation.
  bal-acc   (sensitivity + specificity) / 2 at a 0.5 threshold. Puts every source on one
            operating point, so it is the fair head-to-head irrespective of gradation.

Reporting only AUC would flatter us for free. Read both.
"""
import argparse, subprocess, sys
import numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import auc, bal_acc  # same rank-based AUC that produced the recorded 0.7746

PROJ = Path(__file__).resolve().parents[1]
D, LAB = PROJ / "data", PROJ / "labeling"
PUBLIC_KERNEL = "nekkon/weak-labels-for-all-12-knee-mri-findings"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def fetch_public() -> None:
    out = D / "_public_dl"
    print(f"fetching {PUBLIC_KERNEL} ...")
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output",
                        PUBLIC_KERNEL, "-p", str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"kaggle kernels output failed:\n{r.stderr.strip()}")
    csvs = list(out.glob("*.csv"))
    if not csvs:
        sys.exit(f"no CSV in {out} -- the notebook's output may have changed")
    df = pd.read_csv(csvs[0])
    missing = [c for c in ["StudyInstanceUID"] + LABELS if c not in df.columns]
    if missing:
        sys.exit(f"{csvs[0].name} is missing {missing} -- schema changed, inspect it by hand")
    df[["StudyInstanceUID"] + LABELS].to_csv(D / "public_weak_labels.csv", index=False)
    print(f"wrote data/public_weak_labels.csv  ({len(df):,} studies)")


def load_sources() -> dict[str, pd.DataFrame]:
    src = {}
    for name, path in [("rules", D / "pseudo_labels.csv"),
                       ("llm", D / "llm_pseudo_labels.csv"),
                       ("public", D / "public_weak_labels.csv"),
                       ("hand", LAB / "model_labels.csv")]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "StudyInstanceUID" not in df.columns:
            print(f"  skip {name}: no StudyInstanceUID "
                  f"(run labeling/rekey_labels.py --rekey?)")
            continue
        # The labelling sample plants N_DUP repeat studies under different item_ids for the
        # intra-rater check (sample_for_labeling.py:65), so `hand` genuinely has duplicate
        # UIDs. Left un-deduped they fan out every merge and corrupt the row alignment.
        n = len(df)
        df = df.drop_duplicates(subset="StudyInstanceUID", keep="first")
        if len(df) < n:
            print(f"  {name}: dropped {n - len(df)} duplicate StudyInstanceUID rows "
                  f"(kept first; intra-rater agreement is eval_model_labels.py's job)")
        src[name] = df[["StudyInstanceUID"] + LABELS]
    return src


def score_table(sources, ref, ref_name):
    """Per-label AUC and balanced accuracy for every source against one reference.

    Each source is scored only on the reference rows it actually covers. A source that covers
    a subset (the hand labels overlap gold on ~31 of 58) must not have its missing rows scored
    as negatives -- that silently rewards it for not answering. Coverage is printed so the
    columns are read as what they are: different denominators, not directly comparable.
    """
    names = list(sources)
    merged = {n: ref[["StudyInstanceUID"]].merge(sources[n], on="StudyInstanceUID", how="left")
              for n in names}
    print(f"\n{'='*78}\nvs {ref_name.upper()}  (n={len(ref)})\n{'='*78}")
    print("coverage: " + ",  ".join(
        f"{n} {int(merged[n][LABELS].notna().all(axis=1).sum())}/{len(ref)}" for n in names))

    head = f"{'label':<18}{'n_pos':>6}" + "".join(f"{n:>17}" for n in names)
    print(head); print(f"{'':<24}" + "".join(f"{'AUC / bal-acc':>17}" for _ in names))
    print("-" * len(head))

    acc = {n: {"auc": [], "ba": []} for n in names}
    for lab in LABELS:
        row = f"{lab:<18}{int(ref[lab].astype(float).sum()):>6}"
        for n in names:
            s_all = merged[n][lab].astype(float).values
            keep = ~np.isnan(s_all)
            y, s = ref[lab].astype(float).values[keep], s_all[keep]
            if keep.sum() < 10 or min(y.sum(), (1 - y).sum()) < 2:
                row += f"{'--':>17}"; continue
            a, b = auc(y, s), bal_acc(y, s)
            if not np.isnan(a): acc[n]["auc"].append(a)
            if not np.isnan(b): acc[n]["ba"].append(b)
            row += f"{a:>9.3f} /{b:>6.3f}"
        print(row)

    print("-" * len(head))
    row, out = f"{'MACRO':<18}{'':>6}", {}
    for n in names:
        if not acc[n]["auc"]:
            row += f"{'--':>17}"; continue
        a, b = float(np.mean(acc[n]["auc"])), float(np.mean(acc[n]["ba"]))
        out[n] = (a, b)
        row += f"{a:>9.3f} /{b:>6.3f}"
    print(row)
    return out


def agreement(sources):
    """How redundant are the sources? Two that agree everywhere cannot ensemble."""
    names = list(sources)
    if len(names) < 2:
        return
    print(f"\n{'='*78}\nSOURCE-vs-SOURCE label agreement, full corpus\n{'='*78}")
    print(f"{'':<10}" + "".join(f"{n:>10}" for n in names))
    for a in names:
        row = f"{a:<10}"
        for b in names:
            if a == b:
                row += f"{'-':>10}"; continue
            m = sources[a].merge(sources[b], on="StudyInstanceUID", suffixes=("_a", "_b"))
            if not len(m):
                row += f"{'n/a':>10}"; continue
            va = (m[[f"{c}_a" for c in LABELS]].values >= 0.5)
            vb = (m[[f"{c}_b" for c in LABELS]].values >= 0.5)
            row += f"{(va == vb).mean():>10.3f}"
        print(row)
    print("\nHigh agreement means the two sources are near-substitutes and stacking them buys")
    print("little. Low agreement with similar accuracy is the case worth ensembling.")


def main(args):
    if args.fetch:
        fetch_public()
        return

    sources = load_sources()
    if not sources:
        sys.exit("no label sources found -- run extractor/run_extract.py first")
    print(f"sources: {', '.join(sources)}")
    if "public" not in sources:
        print("  (no public set yet: python extractor/compare_methods.py --fetch)")

    tr = pd.read_csv(D / "train.csv")
    gold = tr.dropna(subset=LABELS)[["StudyInstanceUID"] + LABELS]
    res = score_table({k: v for k, v in sources.items() if k != "gold"}, gold, "gold")

    if "hand" in sources and len(sources) > 1:
        hand = sources["hand"]
        others = {k: v for k, v in sources.items() if k != "hand"}
        score_table(others, hand, "hand labels")

    agreement(sources)

    print(f"\n{'='*78}\nVERDICT\n{'='*78}")
    if "public" in res and "rules" in res:
        ra, rb = res["rules"]; pa, pb = res["public"]
        print(f"  ours (rules)   AUC {ra:.3f}  bal-acc {rb:.3f}")
        print(f"  public         AUC {pa:.3f}  bal-acc {pb:.3f}")
        d = rb - pb
        verb = "beats" if d > 0.02 else ("loses to" if d < -0.02 else "ties")
        print(f"\n  On bal-acc -- the metric that does not favour graded targets -- ours "
              f"{verb} public by {abs(d):.3f}.")
        print("  n=58 with 9-35 positives per label, so this has wide error bars. Treat a gap")
        print("  under ~0.05 as a tie and decide on the per-label rows, not the macro.")
    else:
        print("  Need both 'rules' and 'public' for the week-2 decision. Run --fetch.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true",
                    help="download the public weak labels into data/")
    main(ap.parse_args())
