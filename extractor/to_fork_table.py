"""Our labels in the format pilkwang's notebook reads. README "Where this goes next" Phase 0.

The fork loads `report_labels*.csv` from a mounted Dataset and requires, for each of the 12
targets, both a score column and a `<target>__conf` column. It turns the second into a sample
weight, `W = 0.25 + 0.75 * conf`, so confidence is not decoration -- it decides how much each
study pulls on the loss.

We have the scores (`pseudo_labels.csv`) and the states that produced them
(`extract_states.csv`), but never needed a confidence channel, because `fusion/train.py` weights
every study equally. So the mapping below is NEW, and it is a guess.

    CONF is a guess and is a confound on this arm.

Confidence here means "how much evidence did the extractor have", which is NOT P(gold=1|state)
from IMPROVEMENTS 1.3. A `neg` sits at P(gold=1)=0.073 -- that is a *confident* negative, not a
weak one. An `absent` at 0.167 is the extractor having seen nothing at all, which is the least
evidence available even though its rate is higher. Ordering by evidence rather than by rate is
the whole reason these two columns differ.

If this arm loses on the leaderboard, the loss is ambiguous between "our labels are worse" and
"this mapping is wrong", and the follow-up is a constant-confidence rerun that isolates the
targets. Per README's rules that would be a second submission, not a second variable in this one.

    python extractor/to_fork_table.py --out data/report_labels_ours.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "data"

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Evidence, not probability. See the module docstring for why these are not IMPROVEMENTS 1.3's
# P(gold=1|state) -- an explicit negation is strong evidence and a low rate at the same time.
CONF = {
    "pos":     1.00,   # explicit positive statement
    "neg":     0.90,   # explicit negation -- confident, and confidently NEGATIVE
    "hedged":  0.60,   # stated, with uncertainty language
    "weak":    0.50,   # indirect or non-specific evidence
    "absent":  0.25,   # never mentioned; the extractor saw nothing
}


def build(scores: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in TARGETS if c not in scores.columns]
    if missing:
        sys.exit(f"pseudo_labels.csv is missing {missing}")
    if list(scores.StudyInstanceUID) != list(states.StudyInstanceUID):
        # Both are written by the same run of run_extract.py, so a mismatch means one is stale.
        sys.exit("pseudo_labels.csv and extract_states.csv disagree on row order or coverage -- "
                 "re-run extractor/run_extract.py so both come from one pass.")

    out = pd.DataFrame({"StudyInstanceUID": scores.StudyInstanceUID})
    for t in TARGETS:
        out[t] = scores[t].astype(float)
        s = states[t].astype(str)
        unknown = sorted(set(s) - set(CONF))
        if unknown:
            sys.exit(f"{t}: unmapped states {unknown}. Add them to CONF rather than defaulting, "
                     f"which would silently weight an unknown state like a known one.")
        out[t + "__conf"] = s.map(CONF).astype(float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", default=str(D / "pseudo_labels.csv"))
    ap.add_argument("--states", default=str(D / "extract_states.csv"))
    ap.add_argument("--out", default=str(D / "report_labels_ours.csv"))
    args = ap.parse_args()

    out = build(pd.read_csv(args.scores), pd.read_csv(args.states))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"{len(out):,} studies -> {args.out}")
    print(f"{'label':<18}{'mean score':>11}{'mean conf':>11}")
    for t in TARGETS:
        print(f"{t:<18}{out[t].mean():>11.3f}{out[t + '__conf'].mean():>11.3f}")
    print(f"\ncolumns: {len(out.columns)} (1 uid + {len(TARGETS)} scores + {len(TARGETS)} conf)")


if __name__ == "__main__":
    main()
