"""Decoupled targets for KDC: Lateral OA and PF OA made independent of the other OA compartments.

    python labeling/build_decoupled_targets.py            # -> data/student/targets_teacher_v3_dc.csv (+ .json)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
SRC = PROJ / "data" / "student" / "targets_teacher_v3.csv"
REP = PROJ / "data" / "student" / "targets_consensus_v2.csv"
OUT = PROJ / "data" / "student" / "targets_teacher_v3_dc.csv"
SEED = 20261018
AMBIG = (0.35, 0.65)
PLAN = {"Lateral OA": ("Medial OA", "PF OA"), "PF OA": ("Medial OA", "Lateral OA")}


def decouple(t: pd.DataFrame, col: str, others: tuple[str, str], rng: np.random.Generator) -> tuple[pd.Series, dict]:
    """t: the report-only table."""
    y = (t[col] >= 0.5).astype(float)
    amb = t[col].between(*AMBIG)
    s = pd.DataFrame({"y": y, "a": (t[others[0]] >= 0.5).astype(int), "b": (t[others[1]] >= 0.5).astype(int)})[~amb]
    r = round(float(s.y.mean()), 4)
    keep = pd.Series(False, index=t.index)
    per = {}
    for key, g in s.groupby(["a", "b"]):
        p_idx, n_idx = g.index[g.y == 1], g.index[g.y == 0]
        if len(p_idx) == 0 or len(n_idx) == 0:
            per[str(key)] = {"n": len(g), "pos": len(p_idx), "kept": 0}
            continue
        if len(p_idx) / len(g) <= r:
            kp, kn = len(p_idx), min(len(n_idx), int(round(len(p_idx) * (1 - r) / r)))
        else:
            kp, kn = min(len(p_idx), int(round(len(n_idx) * r / (1 - r)))), len(n_idx)
        chosen = list(rng.choice(p_idx, kp, replace=False)) + list(rng.choice(n_idx, kn, replace=False))
        keep[chosen] = True
        per[str(key)] = {"n": len(g), "pos": len(p_idx), "kept_pos": kp, "kept_neg": kn}
    out = y.where(keep)
    k = out.notna()
    phi = {o: round(float(np.corrcoef(out[k], (t.loc[k, o] >= 0.5).astype(float))[0, 1]), 4) for o in others}
    before = {o: round(float(np.corrcoef(y[~amb], (t.loc[~amb, o] >= 0.5).astype(float))[0, 1]), 4) for o in others}
    return out, {"rate": r, "ambiguous_masked": int(amb.sum()), "kept": int(k.sum()), "kept_pos": int(out[k].sum()),
                 "strata (other1, other2)": per, "phi_with_others_before": before, "phi_with_others_after": phi}


def main() -> None:
    t = pd.read_csv(SRC)
    t = t.set_index("StudyInstanceUID")
    rep = pd.read_csv(REP).set_index("StudyInstanceUID").reindex(t.index)
    if rep[[c for cs in PLAN.items() for c in (cs[0], *cs[1])]].isna().any().any():
        raise SystemExit(f"{REP.name}: a study of {SRC.name} has no report-only OA value")
    rng = np.random.default_rng(SEED)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    new, meta = t.copy(), {"source": SRC.name, "source_sha256": sha(SRC), "report_source": REP.name,
                           "report_source_sha256": sha(REP), "seed": SEED, "ambiguous_band": AMBIG, "columns": {}}
    for col, others in PLAN.items():
        new[col], meta["columns"][col] = decouple(rep, col, others, rng)
    new.reset_index().to_csv(OUT, index=False)
    meta["out_sha256"] = hashlib.sha256(OUT.read_bytes()).hexdigest()
    untouched = [c for c in t.columns if c not in PLAN]
    assert new[untouched].equals(t[untouched]), "a column outside the plan changed"
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
