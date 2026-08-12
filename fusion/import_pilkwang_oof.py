"""Write `pilkwang/rsna-knee-weights::oof.npz` out as a run dir the project scorer can read.

WHY THIS EXISTS. The 0.891 fork is a 20-member ensemble whose weights ship as a Kaggle Dataset,
and that dataset also ships **`oof.npz` -- honest out-of-fold predictions for all 4,407 training
studies**, 368 KB. So the fork's own predictions can be scored on our instrument with no GPU, no
Kaggle run, and no submission. Until 2026-08-12 every comparison against the fork was a
leaderboard number read off a web page; this makes it a local, paired, per-label measurement.

    python fusion/import_pilkwang_oof.py            # -> fusion/runs_pilkwang/oof_all.csv
    python fusion/score_oof.py fusion/runs_pilkwang fusion/runs_port

THE REFERENCE IS NOT NEUTRAL FOR THIS PARTICULAR A/B, AND IT LEANS OUR WAY. `score_oof.py`
scores through `lixin_gpt56`, which correlates 0.947 with `steven_v2` (what the port trained on)
and 0.866 with `pilkwang_v2` (what these members trained on) -- the same asymmetry §2s flagged
for the gate arm. **So this comparison hands the port a handicap in its own favour, and only a
result where the port still LOSES is clean.** That is the read this file is for: it can prove
the port does not deserve an ensemble slot, and it cannot prove that it does.

The 20 members are 5 folds x 4 seeds (2026, 7717, 20260808, 31337) of ONE config -- dinov2-small
@336, 12 slices, unfreeze_last 6, cls_mean, the six SAG/COR/AX slots. `manifest.json` carries no
second architecture. So `oof.npz` is the OOF of a seed-and-fold average, not of a diverse
ensemble, which is worth knowing before treating 0.891 as a hard architectural ceiling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", type=Path, required=True, help="oof.npz from the weights dataset")
    ap.add_argument("--manifest", type=Path, default=None, help="manifest.json, for the summary")
    ap.add_argument("--out", type=Path, default=PROJ / "fusion" / "runs_pilkwang")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    targets = [str(t) for t in z["targets"]]
    if targets != L:
        raise SystemExit(
            "target order differs from the project's -- refusing to write a silently "
            f"mis-columned oof_all.csv\n  theirs: {targets}\n  ours:   {L}")

    ids = [str(u) for u in z["ids"]]
    pred = np.asarray(z["pred"], dtype=float)
    if pred.shape != (len(ids), 12):
        raise SystemExit(f"unexpected pred shape {pred.shape}")

    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(pred, columns=L)
    df.insert(0, "StudyInstanceUID", ids)
    df.to_csv(a.out / "oof_all.csv", index=False)

    summary = {
        "source": "pilkwang/rsna-knee-weights::oof.npz",
        "n_studies": len(ids),
        "n_gold_in_oof": int(np.asarray(z["gold_mask"]).sum()),
        "note": "OOF of a 20-member 5-fold x 4-seed average of ONE dinov2-small@336 config.",
    }
    if a.manifest and a.manifest.exists():
        man = json.load(open(a.manifest))
        mem = man["members"]
        cfgs = {json.dumps(m["config"], sort_keys=True) for m in mem}
        summary |= {
            "n_members": len(mem),
            "distinct_configs": len(cfgs),
            "folds": sorted({m["fold"] for m in mem}),
            "seeds": sorted({m["seed"] for m in mem}),
            "mean_holdout_auc": round(float(np.mean([m["holdout"] for m in mem])), 4),
            "mean_gold58_auc": round(float(np.mean([m["annot"] for m in mem])), 4),
        }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote {a.out/'oof_all.csv'}  ({len(ids)} studies x 12)")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
