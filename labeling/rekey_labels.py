"""Bind the hand labels to StudyInstanceUID instead of positional item_id.

model_labels.csv joins on item_id alone. item_id is assigned positionally after a shuffle
(sample_for_labeling.py:70-71) and carries no reference to the study it names, so any upstream
drift -- a different lingua release reclassifying reports, a refreshed train.csv -- renumbers
every item and silently repoints all 86 labels at the wrong reports. Nothing raises.

  --emit-map   labeling_sample.csv        -> item_id_map.csv   (safe to commit: no report text)
  --rekey      model_labels.csv + map     -> model_labels.csv with a StudyInstanceUID column
  --check      verify the current sample still matches what the labels were made against

--check has two modes. If item_id_map.csv exists it compares the mapping directly, which is
exact. If it does not -- the case on any clone made before the map was first emitted -- it
falls back to re-measuring agreement with gold. Correct alignment reproduces the recorded
84.7%; a drifted mapping collapses toward chance, because the labels are then being scored
against unrelated studies.
"""
import argparse, sys
import pandas as pd
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
LAB, D = PROJ / "labeling", PROJ / "data"
SAMPLE, LABELS_CSV, MAP = LAB/"labeling_sample.csv", LAB/"model_labels.csv", LAB/"item_id_map.csv"

L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
     "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

# From README / FINDINGS: full-text reading vs gold, measured on the original sample.
EXPECTED_GOLD_AGREEMENT = 0.847
EXPECTED_GOLD_N = 30      # all 30 blind gold studies were labelled deliberately
TOL = 0.05


HINT = {
    SAMPLE:     "python labeling/sample_for_labeling.py",
    MAP:        "python labeling/rekey_labels.py --emit-map",
    D/"train.csv": "python -m kaggle competitions download "
                   "-c rsna-knee-abnormality-detection -f train.csv -p data",
}


def _need(p: Path):
    if not p.exists():
        sys.exit(f"missing {p.relative_to(PROJ)}"
                 + (f"\n  regenerate with: {HINT[p]}" if p in HINT else ""))
    return pd.read_csv(p)


def emit_map():
    samp = _need(SAMPLE)
    cols = ["item_id", "StudyInstanceUID"] + [c for c in ("is_gold", "dup_of") if c in samp]
    samp[cols].to_csv(MAP, index=False)
    print(f"wrote {MAP.relative_to(PROJ)}  ({len(samp)} items)")
    print("no report text in this file -- commit it, it is the durable fingerprint.")


def rekey():
    mine, mp = _need(LABELS_CSV), _need(MAP)
    if "StudyInstanceUID" in mine.columns:
        sys.exit("model_labels.csv already carries StudyInstanceUID; nothing to do.")
    out = mine.merge(mp[["item_id", "StudyInstanceUID"]], on="item_id", how="left")
    lost = out.StudyInstanceUID.isna().sum()
    if lost:
        sys.exit(f"{lost} of {len(mine)} item_ids are absent from the map -- refusing to write. "
                 f"The map and the labels come from different samples.")
    out = out[["item_id", "StudyInstanceUID"] + L]
    out.to_csv(LABELS_CSV, index=False)
    print(f"re-keyed {len(out)} labels onto StudyInstanceUID -> {LABELS_CSV.relative_to(PROJ)}")


def check():
    samp, mine = _need(SAMPLE), _need(LABELS_CSV)

    if MAP.exists():
        mp = pd.read_csv(MAP)
        cur = samp.set_index("item_id").StudyInstanceUID
        ref = mp.set_index("item_id").StudyInstanceUID
        shared = cur.index.intersection(ref.index)
        bad = (cur[shared] != ref[shared]).sum()
        missing = len(ref.index.difference(cur.index))
        print(f"exact check against {MAP.name}: {len(shared)} shared item_ids, "
              f"{bad} mismatched, {missing} absent from the current sample")
        if bad or missing:
            sys.exit("DRIFT. The current sample does not reproduce the mapping the labels were "
                     "made against. Do not evaluate until this is resolved.")
        print("OK -- mapping reproduces exactly.")
        return

    # No fingerprint was ever captured. Fall back to the statistical signature.
    #
    # Gold MUST come from train.csv, joined through the StudyInstanceUID the *current* sample
    # assigns to each item_id. The sample carries its own label columns, but those travel with
    # item_id and stay self-consistent under drift, so comparing against them proves nothing.
    print(f"{MAP.name} not found -- falling back to gold-agreement check.\n")
    tr = _need(D/"train.csv")
    gold = tr.dropna(subset=L)

    m = samp[["item_id", "StudyInstanceUID"]].merge(mine, on="item_id")
    if not len(m):
        sys.exit("no item_ids in common between the sample and the labels -- certain drift.")
    g = m.merge(gold[["StudyInstanceUID"] + L], on="StudyInstanceUID",
                suffixes=("_mine", "_gold"))
    print(f"labelled items: {len(m)};  of those, {len(g)} map to gold-labelled studies "
          f"(expected {EXPECTED_GOLD_N})")
    if len(g) < EXPECTED_GOLD_N - 5:
        sys.exit(f"expected ~{EXPECTED_GOLD_N} labelled items to land on gold studies -- all 30 "
                 f"blind gold studies were labelled deliberately. Finding {len(g)} means "
                 f"item_id no longer points where it did. DRIFT.")

    agree = (g[[f"{c}_mine" for c in L]].values == g[[f"{c}_gold" for c in L]].values).mean()
    print(f"label-level agreement with gold: {agree:.3f}  (recorded: {EXPECTED_GOLD_AGREEMENT})")
    if abs(agree - EXPECTED_GOLD_AGREEMENT) > TOL:
        sys.exit(f"outside +/-{TOL} of the recorded value -- treat the mapping as drifted.")
    print("OK -- consistent with the recorded figure. Run --emit-map now to pin it exactly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for flag, fn in (("--emit-map", emit_map), ("--rekey", rekey), ("--check", check)):
        g.add_argument(flag, dest="fn", action="store_const", const=fn)
    ap.parse_args().fn()
