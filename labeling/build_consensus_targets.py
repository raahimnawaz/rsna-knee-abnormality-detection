r"""Build training targets from several independent readers of the same reports, with silence masked.

    python labeling/build_consensus_targets.py --labels-dir data/public_llm_labels --out data/student/targets_consensus_v1.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SYN = "Synovitis"
BASE_SYN_FILLS = (0.35, 0.76, 0.82)
ABSENT, LO, HI = 0.05, 0.05, 0.95


def _csv(p: Path, **kw) -> pd.DataFrame:
    d = pd.read_csv(p, low_memory=False, **kw)
    return d.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")


def load_readers(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """-> (values, addressed)."""
    val, adr = {}, {}

    val["dread"] = _csv(root / "dreaddevelopment_rsna-knee-labels" / "labels_llm_soft.csv")[LABELS]
    val["steven_v2"] = _csv(root / "stevenleehans_rsna-knee-llm-report-labels" / "llm_labels_v2.csv")[LABELS]
    val["yunus"] = _csv(root / "yunusgmsoy_rsna-knee-llm-report-labels" / "yunus_llm_labels.csv")[LABELS]
    val["jev"] = _csv(root / "riadmohamed42_jev-knee-labels" / "FINAL2_labels_scores.csv")[LABELS]
    val["nartaa_raw"] = pd.read_parquet(root / "nartaa_rsna-knee-hpo-assets" / "labels_v0raw.parquet") \
        .drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[LABELS]

    p = _csv(root / "pilkwang_rsna-knee-llm-labels" / "report_labels_v2.csv")
    unk = pd.DataFrame({c: p[f"{c}__verdict"].astype(str).str.upper().eq("UNK") for c in LABELS})
    val["pilkwang"] = p[LABELS].where(~unk)
    adr["pilkwang"] = ~unk

    v = root / "vmohitrao_rsna-knee-report-labels"
    st = _csv(v / "label_states.csv")[LABELS]
    val["vmohitrao"] = st.replace({"P": 1.0, "N": 0.0, "B": 0.0, "U": np.nan, "M": np.nan}).astype(float)
    adr["vmohitrao"] = st.ne("M") & st.notna()

    q = _csv(root / "laymond_rsna-knee-abnormality-qwen3-8b-weak-labels" / "weak_labels_v2_raw.csv",
             usecols=lambda c: c == "StudyInstanceUID" or c.endswith("__mentioned"))
    adr["laymond"] = pd.DataFrame({c: q[f"{c}__mentioned"].eq(1) for c in LABELS})

    h = _csv(root / "hosseini7798_rsna-knee-label-extraction-jev" / "Label_Extraction_Jev_v2.csv",
             usecols=lambda c: c == "StudyInstanceUID" or c.endswith("_9"))
    adr["hosseini"] = pd.DataFrame({c: h[f"{c}_9"].lt(0.5) for c in LABELS})

    for k in val:
        val[k] = val[k].apply(pd.to_numeric, errors="coerce")
        if ((val[k] < 0) | (val[k] > 1)).any().any():
            raise SystemExit(f"reader {k}: values outside [0, 1]")
    return val, adr


def on_base_ties(t: pd.Series, b: pd.Series) -> pd.Series:
    """The reader's order, the base's values on the same studies."""
    ids = t.index.intersection(b.index)
    t, sb = t.loc[ids], np.sort(b.loc[ids].to_numpy())
    lo = (rankdata(t.to_numpy(), method="min") - 1).astype(int)
    hi = rankdata(t.to_numpy(), method="max").astype(int)
    cs = np.concatenate([[0.0], np.cumsum(sb)])
    return pd.Series((cs[hi] - cs[lo]) / (hi - lo), index=ids)


def consensus(base: pd.Series, readers: dict[str, pd.Series], w_base: float) -> tuple[pd.Series, pd.Series]:
    """-> (target, number of readers behind each cell)."""
    b = base.dropna()
    est = pd.concat([on_base_ties(r.dropna(), b) for r in readers.values()], axis=1).reindex(b.index)
    n = est.notna().sum(axis=1)
    cons = est.mean(axis=1, skipna=True)
    out = (w_base * b + (1 - w_base) * cons).where(n > 0, b)
    return out, n


def synovitis(val: dict[str, pd.DataFrame], adr: dict[str, pd.DataFrame], effusion: pd.Series,
              ids: pd.Index) -> tuple[pd.Series, dict]:
    """Supervise Synovitis only where it was read, or where Effusion is read as absent."""
    base = val["dread"][SYN].reindex(ids)
    votes = pd.concat({k: a[SYN].reindex(ids) for k, a in adr.items()}, axis=1).astype(float)
    votes["dread"] = (~base.round(2).isin(BASE_SYN_FILLS)).astype(float)
    addressed = votes.mean(axis=1) >= 0.5
    reads = pd.concat({
        "dread": base.where(~base.round(2).isin(BASE_SYN_FILLS)),
        "pilkwang": val["pilkwang"][SYN].reindex(ids),
        "vmohitrao": val["vmohitrao"][SYN].reindex(ids),
        "jev": val["jev"][SYN].reindex(ids),
        "nartaa_raw": val["nartaa_raw"][SYN].reindex(ids),
    }, axis=1)
    read = reads.mean(axis=1, skipna=True).clip(LO, HI)
    eff_absent = effusion.reindex(ids) <= effusion.min() + 1e-9
    y = pd.Series(np.nan, index=ids)
    y[addressed] = read[addressed]
    quiet_dry = ~addressed & eff_absent
    y[quiet_dry] = ABSENT
    info = {"addressed": int(addressed.sum()), "absent_by_effusion": int(quiet_dry.sum()),
            "masked": int(y.isna().sum()), "supervised_share": round(float(y.notna().mean()), 4),
            "positive_share_of_supervised": round(float((y.dropna() > 0.5).mean()), 4),
            "mention_votes": {k: round(float(votes[k].mean()), 4) for k in votes.columns}}
    return y, info


def add_teacher(out: pd.DataFrame, teacher: pd.DataFrame, w: float) -> tuple[pd.DataFrame, dict]:
    """Blend an outside image teacher into the targets, and let it fill the cells left unsupervised."""
    new, info = out.copy(), {}
    for lab in LABELS:
        t = teacher[lab].reindex(out.index)
        sup = out[lab].notna() & t.notna()
        mapped = on_base_ties(t[sup], out.loc[sup, lab])
        new.loc[sup, lab] = (1 - w) * out.loc[sup, lab] + w * mapped
        fill = out[lab].isna() & t.notna()
        if fill.any():
            xs, ys = np.sort(t[sup].to_numpy()), np.sort(out.loc[sup, lab].to_numpy())
            new.loc[fill, lab] = np.interp(t[fill].to_numpy(), xs, ys)
        info[lab] = {"supervised_with_teacher": int(sup.sum()), "filled_by_teacher": int(fill.sum()),
                     "still_masked": int(new[lab].isna().sum()),
                     "spearman_teacher_vs_target": round(float(spearmanr(t[sup], out.loc[sup, lab])[0]), 3)}
    return new, info


def macro(y: pd.DataFrame, p: pd.DataFrame) -> float:
    return float(np.mean([roc_auc_score((y[c] > 0.5).astype(int), p[c]) for c in LABELS]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels-dir", type=Path, required=True, help="one folder per public table, <owner>_<dataset>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--w-base", type=float, default=0.5, help="the base table's share of each target")
    ap.add_argument("--synovitis", choices=["masked", "base"], default="masked",
                    help="base: keep the base table's filled Synovitis column (the control arm)")
    ap.add_argument("--teacher", type=Path, help="an outside image teacher's predictions for the training studies "
                    "(StudyInstanceUID + 12 columns, e.g. a --predict-all run's oof_all.csv)")
    ap.add_argument("--w-teacher", type=float, default=0.5, help="the teacher's share of each supervised cell")
    ap.add_argument("--own-oof", type=Path, nargs="+", default=[], help="our fold models' held-out predictions "
                    "(oof_all.csv of each fold kernel; disjoint studies), blended in after --teacher")
    ap.add_argument("--w-own", type=float, default=0.5, help="our own models' share of each cell")
    ap.add_argument("--gold", type=Path, help="train.csv: score the readers that carry its graded rows, report only")
    a = ap.parse_args()

    val, adr = load_readers(a.labels_dir)
    base = val["dread"]
    ids = base.index
    readers = {k: v for k, v in val.items() if k != "dread"}
    out = pd.DataFrame(index=ids, columns=LABELS, dtype=float)
    report = {"base": "dread", "w_base": a.w_base, "readers": sorted(readers), "mention_sources": sorted(adr),
              "synovitis": a.synovitis, "per_finding": {}}
    for lab in LABELS:
        if lab == SYN:
            continue
        r = {k: v[lab] for k, v in readers.items()}
        out[lab], n = consensus(base[lab], r, a.w_base)
        rho = [spearmanr(v.reindex(ids), base[lab], nan_policy="omit")[0] for v in r.values()]
        report["per_finding"][lab] = {
            "readers_per_cell_mean": round(float(n.mean()), 2),
            "spearman_reader_vs_base_mean": round(float(np.mean(rho)), 3),
            "spearman_target_vs_base": round(float(spearmanr(out[lab], base[lab])[0]), 3),
            "cells_moved_over_0.1": int(((out[lab] - base[lab]).abs() > 0.1).sum())}
    if a.synovitis == "masked":
        out[SYN], report["per_finding"][SYN] = synovitis(val, adr, base["Effusion"], ids)
    else:
        out[SYN] = base[SYN]
        report["per_finding"][SYN] = {"kept": "the base column"}
    if a.teacher:
        t = pd.read_csv(a.teacher).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[LABELS]
        miss = len(out.index.difference(t.index))
        out, report["teacher"] = add_teacher(out, t.apply(pd.to_numeric, errors="coerce"), a.w_teacher)
        report["teacher"].update(file=str(a.teacher), sha256=hashlib.sha256(a.teacher.read_bytes()).hexdigest(),
                                 w_teacher=a.w_teacher, studies_without_teacher=miss)
    if a.own_oof:
        parts = [pd.read_csv(f) for f in a.own_oof]
        o = pd.concat(parts)
        if o.StudyInstanceUID.duplicated().any():
            raise SystemExit("--own-oof: a study is in two files; held-out predictions must not overlap")
        o = o.set_index("StudyInstanceUID")[LABELS].apply(pd.to_numeric, errors="coerce")
        miss = len(out.index.difference(o.index))
        out, report["own_oof"] = add_teacher(out, o, a.w_own)
        report["own_oof"].update(files=[str(f) for f in a.own_oof], w_own=a.w_own, studies_without_oof=miss,
                                 sha256=[hashlib.sha256(f.read_bytes()).hexdigest() for f in a.own_oof])
    v = out.to_numpy(float)
    if out.drop(columns=SYN).isna().any().any() or ((v < 0) | (v > 1)).any():
        raise SystemExit("a target is missing or outside [0, 1]")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().rename(columns={"index": "StudyInstanceUID"}).to_csv(a.out, index=False)
    report["rows"] = int(len(out))
    report["sha256"] = hashlib.sha256(a.out.read_bytes()).hexdigest()
    if a.gold:
        g = pd.read_csv(a.gold).dropna(subset=LABELS).set_index("StudyInstanceUID")[LABELS].astype(float)
        if len(out.index.intersection(g.index)):
            raise SystemExit("graded studies are in the targets")
        report["gold_readers_report_only"] = {
            k: round(macro(g.loc[i], v.loc[i].fillna(0.5)), 4) for k, v in val.items()
            for i in [g.index.intersection(v.index)] if len(i) >= 50}
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    for lab, v in report["per_finding"].items():
        print(f"  {lab:17s} {v}")
    for lab in LABELS if a.teacher else []:
        print(f"  teacher {lab:17s} {report['teacher'][lab]}")
    for lab in LABELS if a.own_oof else []:
        print(f"  own oof {lab:17s} {report['own_oof'][lab]}")
    print(f"wrote {a.out} ({len(out)} studies, sha256 {report['sha256'][:16]}) and {a.out.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
