"""Train the CNN teachers locally: four folds, each predicting the studies it never saw.

    HF_HUB_OFFLINE=1 python fusion/local_teacher_folds.py --corpus data/stage_a --out fusion/runs_teacher_cnn
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TARGETS = PROJ / "data/public_llm_labels/dreaddevelopment_rsna-knee-labels/labels_llm_soft.csv"
RECIPE_C = ["--arch", "resnet34", "--bb-lr", "1e-3", "--aug-affine", "1", "--epochs", "17", "--mps", "--save-weights"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    a = ap.parse_args()

    F = pd.read_csv(PROJ / "provenance/folds_site.csv")
    T = pd.read_csv(TARGETS).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    known = set(T.index[T[LABELS].notna().all(axis=1)])
    pool = sorted(u for u in F.loc[(F.fold != 0) & (~F.is_gold.astype(bool)), "StudyInstanceUID"] if u in known)
    ref = set(pd.read_csv(PROJ / "fusion/runs_dread_ortho/train_studies.csv").StudyInstanceUID)
    if set(pool) != ref:
        raise SystemExit(f"pool {len(pool)} != the Kaggle runs' 3,502 training studies -- refusing")
    val691 = pd.read_csv(PROJ / "fusion/runs_dread_ortho/val_studies.csv").StudyInstanceUID.tolist()

    for k in a.folds:
        run = a.out / f"fold{k}"
        if (run / "summary.json").exists():
            print(f"fold {k}: done already, skipped", flush=True)
            continue
        run.mkdir(parents=True, exist_ok=True)
        fk = set(F.loc[F.fold == k, "StudyInstanceUID"])
        train = [u for u in pool if u not in fk]
        val = [u for u in pool if u in fk] + val691            # its own unseen fold, then the held-out set
        pd.Series(train, name="StudyInstanceUID").to_csv(run / "train_list.csv", index=False)
        pd.Series(val, name="StudyInstanceUID").to_csv(run / "val_list.csv", index=False)
        print(f"fold {k}: train {len(train)} | predict {len(val) - len(val691)} unseen + {len(val691)} held-out", flush=True)
        cmd = [sys.executable, "-u", str(PROJ / "fusion/train_windows.py"), *RECIPE_C,
               "--corpus", str(a.corpus), "--layout", "stage_a", "--labels", str(TARGETS),
               "--train", str(run / "train_list.csv"), "--val", str(run / "val_list.csv"),
               "--gold", str(PROJ / "data/train.csv"), "--out", str(run), "--weights-dir", "/tmp"]
        env = {**os.environ, "TW_WORKERS": os.environ.get("TW_WORKERS", "0")}
        with open(run / "train.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        print(f"fold {k}: exit {rc}", flush=True)
        if rc:
            raise SystemExit(f"fold {k} failed -- see {run / 'train.log'}")


if __name__ == "__main__":
    main()
