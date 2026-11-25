"""Option B training targets: the three image reads (targets_imgread_v1), 2nd place's PF OA rule (half the report
read), then our October teacher's held-out predictions mixed in at 2nd place's per-finding shares, rank-mapped onto
the base so each finding keeps its mean. Rules: docs/experiment-log.md, 11-25 08:35 PST (first written as
~/rsna-knee-work/2026-11-25/option-b/RULES.md, before the build).

    python build_imgread_mix_targets.py --out DIR      # -> DIR/targets_imgread_mix_v1.csv (+ .json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path.home() / "rsna-knee-abnormality-detection"
sys.path.insert(0, str(REPO / "labeling"))
from build_consensus_targets import LABELS, on_base_ties  # noqa: E402  (the rank map teacher_v1 was built with)

D = REPO / "data"
P = D / "public_llm_labels"
READS = D / "student" / "targets_imgread_v1.csv"
REPORT = D / "student" / "targets_consensus_v2.csv"
OLD = D / "student" / "targets_teacher_v1.csv"
TEACHER_RUNS = {0: "runs_cnxs_teach1_f0", 1: "runs_cnxs_teach1_f12", 2: "runs_cnxs_teach1_f12",
                3: "runs_cnxs_teach1_f34", 4: "runs_cnxs_teach1_f34"}
PF = "PF OA"
W_REPORT_PF = 0.5
READ_SHARE = {c: 0.7 if c in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "PF OA", "Lateral OA") else 0.4
              for c in LABELS}
IMAGE_READS = [(P / "tascj0_rsna-2026-knee-agent-labels" / "labels_astra_trained_guide.csv", ""),
               (P / "drizlo_rsna-knee-mri-llm-soft-labels" / "labels.csv", ""),
               (P / "matsurihime_rsna-knee-llm-readings" / "llm_readings.csv", "astra_image__")]
REPORT_READS_58 = [P / "stevenleehans_rsna-knee-llm-report-labels" / "llm_labels_v2.csv",
                   P / "yunusgmsoy_rsna-knee-llm-report-labels" / "yunus_llm_labels.csv",
                   P / "riadmohamed42_jev-knee-labels" / "FINAL2_labels_scores.csv",
                   P / "pilkwang_rsna-knee-llm-labels" / "report_labels_v2.csv"]
NOT_BROKEN_58 = 0.926
WEAK = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis"]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def table(p: Path, prefix: str = "") -> pd.DataFrame:
    d = pd.read_csv(p, low_memory=False).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    d = d[[prefix + c for c in LABELS]].apply(pd.to_numeric, errors="coerce")
    d.columns = LABELS
    return d


def fold_dir(f: int) -> Path:
    return REPO / "fusion" / TEACHER_RUNS[f] / "runs_convnext_small" / f"fold{f}"


def fix_pf(reads: pd.DataFrame, report: pd.Series) -> pd.DataFrame:
    base = reads.copy()
    rep = report.reindex(base.index)
    base[PF] = (W_REPORT_PF * rep + (1 - W_REPORT_PF) * reads[PF]).where(rep.notna(), reads[PF])
    return base


def mix(base: pd.DataFrame, teacher: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    for c in LABELS:
        mapped = on_base_ties(teacher[c].reindex(base.index), base[c]).reindex(base.index)
        out[c] = READ_SHARE[c] * base[c] + (1 - READ_SHARE[c]) * mapped
    return out


def aucs(y: pd.DataFrame, p: pd.DataFrame) -> pd.Series:
    return pd.Series({c: roc_auc_score(y[c], p[c]) for c in LABELS})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out_csv = a.out / "targets_imgread_mix_v1.csv"

    reads, report, old = table(READS), table(REPORT), table(OLD)
    studies = reads.index.sort_values()
    if set(report.index) != set(studies) or set(old.index) != set(studies):
        raise SystemExit("the report table or teacher_v1 covers different studies")
    if reads.isna().any().any() or report[PF].isna().any():
        raise SystemExit("a missing read or report PF OA value")
    teacher = pd.concat([pd.read_csv(fold_dir(f) / "oof_all.csv") for f in TEACHER_RUNS]).set_index("StudyInstanceUID")
    if not teacher.index.is_unique or set(teacher.index) != set(studies) or teacher[LABELS].isna().any().any():
        raise SystemExit("the teacher's held-out predictions do not cover each study exactly once")

    base = fix_pf(reads.loc[studies], report.loc[studies, PF])
    out = mix(base, teacher.loc[studies, LABELS]).round(4)
    if out.isna().any().any() or ((out < 0) | (out > 1)).any().any():
        raise SystemExit("a value missing or outside [0, 1]")
    out.index.name = "StudyInstanceUID"
    a.out.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(out_csv, index=False)

    per = {c: {"read_share": READ_SHARE[c], "teacher_share": round(1 - READ_SHARE[c], 2),
               "positive_share": {"reads": round(float((reads[c] > 0.5).mean()), 3),
                                  "after_pf_fix": round(float((base[c] > 0.5).mean()), 3),
                                  "out": round(float((out[c] > 0.5).mean()), 3),
                                  "teacher_v1": round(float((old.loc[studies, c].dropna() > 0.5).mean()), 3)},
               "mean_out_vs_base": [round(float(out[c].mean()), 4), round(float(base[c].mean()), 4)],
               "spearman_out_vs_reads": round(float(spearmanr(out[c], reads.loc[studies, c])[0]), 3),
               "spearman_out_vs_teacher_v1": round(float(spearmanr(out[c], old.loc[studies, c],
                                                                   nan_policy="omit")[0]), 3)}
           for c in LABELS}

    # not-broken check on the 58: the recipe rebuilt there approximately (report only, never a selector)
    tr = pd.read_csv(D / "train.csv", low_memory=False).drop_duplicates("StudyInstanceUID")
    gold = tr[tr[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS]
    g, y = gold.index, (gold > 0.5).astype(int)
    reads58 = pd.concat([table(p, pre).reindex(g) for p, pre in IMAGE_READS]).groupby(level=0).mean().loc[g]
    rep58 = pd.concat([table(p).reindex(g) for p in REPORT_READS_58]).groupby(level=0).mean().loc[g]
    t58 = pd.concat([pd.read_csv(fold_dir(f) / "oof_gold.csv") for f in TEACHER_RUNS]).groupby(
        "StudyInstanceUID")[LABELS].mean().loc[g]
    base58 = fix_pf(reads58, rep58[PF])
    out58 = mix(base58, t58)
    parts = {"reads": reads58, "reads_pf_fixed": base58, "old_teacher_5fold_mean": t58, "option_b": out58}
    chk = {}
    for k, p in parts.items():
        s = aucs(y, p)
        chk[k] = {"macro": round(float(s.mean()), 4), "weak_four": round(float(s[WEAK].mean()), 4),
                  "per_finding": {c: round(float(v), 3) for c, v in s.items()}}
    passed = chk["option_b"]["macro"] >= NOT_BROKEN_58

    meta = {"rules": "docs/experiment-log.md 11-25 (written before the build)",
            "inputs": {"reads": {"file": str(READS.relative_to(REPO)), "sha256": sha(READS)},
                       "report_pf": {"file": str(REPORT.relative_to(REPO)), "sha256": sha(REPORT)},
                       "teacher_oof": {f"fold{f}": {"file": str((fold_dir(f) / "oof_all.csv").relative_to(REPO)),
                                                    "sha256": sha(fold_dir(f) / "oof_all.csv")} for f in TEACHER_RUNS}},
            "pf_rule": f"PF OA = {W_REPORT_PF} x report + {1 - W_REPORT_PF} x reads (2nd place's expert-pf)",
            "mix_rule": "y = w x base + (1 - w) x teacher rank-mapped onto base (on_base_ties); w per finding below",
            "studies": len(out), "per_finding": per,
            "check_58_not_broken": {"threshold": NOT_BROKEN_58, "passed": bool(passed), **chk,
                                    "report_pf_missing_on_58": int(rep58[PF].isna().sum())},
            "out": out_csv.name, "out_sha256": sha(out_csv)}
    out_csv.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    print(f"wrote {out_csv} ({len(out)} studies, sha256 {meta['out_sha256'][:16]}…)")
    print("\nshare of studies > 0.5:   reads   +PF fix   option B   teacher_v1")
    for c in LABELS:
        s = per[c]["positive_share"]
        print(f"  {c:17s} {s['reads']:6.3f} {s['after_pf_fix']:9.3f} {s['out']:10.3f} {s['teacher_v1']:12.3f}")
    print("\nthe 58 (report only):  macro   weak four")
    for k, v in chk.items():
        print(f"  {k:24s} {v['macro']:.4f}  {v['weak_four']:.4f}")
    print(f"\nnot broken (>= {NOT_BROKEN_58}): {'PASS' if passed else 'FAIL'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
