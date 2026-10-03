"""Teacher -> student targets: the report label, re-ordered by image teachers that never saw the study.

    python labeling/build_student_targets.py \
        --base data/public_llm_labels/dreaddevelopment_rsna-knee-labels/labels_llm_soft.csv \
        --teacher cnn='runs_teacher_t*/oof_all.csv' \
        --teacher nyu='runs_nyu_teacher/oof_all.csv@ACL,Effusion,Contusion' \
        --out data/student/targets_round1.csv
    # the gate before any GPU is spent on a student (RUNBOOK, TARGET SYSTEM):
    python labeling/build_student_targets.py --base <report table WITH gold rows, e.g. steven_v2> \
        --teacher nyu='runs_nyu_teacher/oof_gold.csv@ACL,Effusion,Contusion' --gold data/train.csv --out /tmp/g.csv

HOW, per finding. A teacher's scores are first put on the base's scale: within the studies that teacher covers, its
ranks are mapped onto the base's own sorted values there (same levels, same prevalence -- AUC only sees order, and the
sources are on different scales). The student target is then the MEAN of the base value and each covering teacher's
value: agreement keeps the level, a clear conflict (report 0.95, teacher at the bottom) lands near 0.5 -- uncertain,
not flipped. A study no teacher covers keeps its base value EXACTLY. Nothing is fitted; every source counts once.
(09-27: the first version re-ranked everything onto the base's marginal; it moved uncovered studies at tied level
boundaries and flipped conflicts outright -- caught by this file's own invariant test.)

WHY a teacher must not have seen the study: a model scoring studies it trained on mostly repeats their labels
(DATA.md §12 item 7). Pass OOF files -- one per teacher fold, globbed together -- never a model's training-set output.
Each file is mapped onto the base on ITS OWN studies before the folds are joined (AUDIT 09-28): the folds are
different models on site-grouped studies, and one ranking over their pooled raw scores moves whole sites.

WHERE THE BASE SAYS NOTHING (`--base-silent FINDING=level,level`, 10-03). A report table fills a finding the report
never discusses. `dread_soft` fills Synovitis from its own Effusion level: 3,751 of its 4,349 Synovitis cells are
0.35 / 0.76 / 0.82, a fixed function of Effusion 0.05-0.20 / 0.85 / 0.95 (the reports couple the two at 0.92 / 0.01,
the graders at 0.63 / 0.22). Such a cell carries no reading of the finding, so averaging it in counts the Effusion
REPORT twice. For the levels named, a study a teacher covers takes the teachers' value alone (still on the base's
scale); a study no teacher covers keeps its base value, as everywhere else. Without the flag nothing changes.

Also writes <out>.audit.csv: per finding, the studies whose level moved furthest (the confident-learning idea:
where independent image models disagree most with the report). With --gold, scores base vs combined on the
fully-labelled studies of train.csv (gold-58): report label alone vs report + teachers.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def read_table(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"{pattern}: no files")
    d = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    return d[[c for c in LABELS if c in d.columns]].apply(pd.to_numeric, errors="coerce")


def read_parts(pattern: str) -> list[pd.DataFrame]:
    """A teacher's files kept APART: each is one fold model, with its own calibration. Pooling their raw scores
    before ranking lifts whole site-grouped folds (AUDIT 09-28, MEDIUM; measured on the CNN folds: up to 0.11
    of a percentile per fold), so each file is mapped onto the base on its own studies in `combine`."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"{pattern}: no files")
    return [pd.read_csv(f).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
            .pipe(lambda d: d[[c for c in LABELS if c in d.columns]].apply(pd.to_numeric, errors="coerce"))
            for f in files]


def parse_teacher(spec: str) -> tuple[str, str, list[str]]:
    name, rest = spec.split("=", 1)
    pattern, _, labs = rest.partition("@")
    labs = [x.strip() for x in labs.split(",")] if labs else list(LABELS)
    bad = [x for x in labs if x not in LABELS]
    if bad:
        raise SystemExit(f"teacher {name}: unknown findings {bad}")
    return name, pattern, labs


def unit_rank(s: pd.Series) -> pd.Series:
    """Ranks in (0, 1] over the non-missing entries; ties averaged."""
    s = s.dropna()
    return pd.Series(rankdata(s.to_numpy()) / len(s), index=s.index)


def on_base(tt: pd.Series, b: pd.Series) -> pd.Series:
    """The teacher's order, the base's values on these studies."""
    order = np.argsort(rankdata(tt.to_numpy(), method="ordinal"))
    v = np.empty(len(tt))
    v[order] = np.sort(b.loc[tt.index].to_numpy())
    return pd.Series(v, index=tt.index)


def parse_silent(specs: list[str], base: pd.DataFrame) -> dict[str, list[float]]:
    out = {}
    for spec in specs:
        lab, _, levels = spec.partition("=")
        if lab not in LABELS or not levels:
            raise SystemExit(f"--base-silent {spec!r}: want FINDING=level,level")
        vals = [float(x) for x in levels.split(",")]
        have = set(base[lab].dropna().round(4))
        missing = [v for v in vals if round(v, 4) not in have]
        if missing:
            raise SystemExit(f"--base-silent {lab}: the base has no level {missing} (it has {sorted(have)})")
        out[lab] = vals
    return out


def combine(base: pd.DataFrame, teachers: list[tuple[str, list[pd.DataFrame], list[str]]],
            silent: dict[str, list[float]] | None = None) -> tuple[pd.DataFrame, dict]:
    out, info = base.copy(), {}
    for lab in LABELS:
        b = base[lab].dropna()
        ests = []
        for name, parts, labs in teachers:
            if lab not in labs:
                continue
            # one fold file at a time onto the base, then the teacher's value per study = the mean over the files that
            # scored it (one for a fold's own studies; all of them for the 691 / gold). The teacher still counts once.
            per = [on_base(tt, b) for t in parts if lab in t.columns
                   for tt in [t.loc[t.index.intersection(b.index), lab].dropna()] if len(tt) >= 2]
            if per:
                ests.append(pd.concat(per, axis=1).mean(axis=1))
        if not ests:
            info[lab] = {"studies": int(len(b)), "with_a_teacher": 0, "level_changed": 0}
            continue
        stack = pd.concat([b] + ests, axis=1)
        covered = stack.iloc[:, 1:].notna().any(axis=1)
        dropped = 0
        if silent and lab in silent:
            # the base value here is a fill, not a reading: the teachers' value stands alone (on the base's scale)
            fill = b.round(4).isin([round(v, 4) for v in silent[lab]]) & covered
            stack.loc[fill[fill].index, stack.columns[0]] = np.nan
            dropped = int(fill.sum())
        student = stack.mean(axis=1, skipna=True)
        out.loc[covered[covered].index, lab] = student[covered]
        info[lab] = {"studies": int(len(b)), "with_a_teacher": int(covered.sum()),
                     "level_changed": int(((out.loc[b.index, lab] - b).abs() > 1e-9).sum()),
                     **({"base_silent_levels": silent[lab], "teacher_alone": dropped} if silent and lab in silent else {})}
    return out, info


def macro(y: pd.DataFrame, p: pd.DataFrame) -> float:
    return float(np.nanmean([roc_auc_score(y[c], p[c]) for c in LABELS if y[c].nunique() > 1 and p[c].notna().all()]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="the report-derived table (StudyInstanceUID + 12 findings)")
    ap.add_argument("--teacher", action="append", default=[],
                    help="NAME=GLOB[@finding,finding]: OOF predictions; the glob joins a teacher's folds")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gold", type=Path, help="train.csv: score base vs combined on its fully labelled rows")
    ap.add_argument("--audit-top", type=int, default=30, help="per finding, how many biggest moves to list")
    ap.add_argument("--base-silent", action="append", default=[],
                    help="FINDING=level,level: base levels that are a FILL, not a reading; a covered study takes the "
                         "teachers' value alone there")
    a = ap.parse_args()

    base = read_table(a.base)
    silent = parse_silent(a.base_silent, base)
    teachers = []
    for spec in a.teacher:
        name, pattern, labs = parse_teacher(spec)
        parts = read_parts(pattern)
        teachers.append((name, parts, labs))
        ids = pd.Index(sorted(set().union(*[set(p.index) for p in parts])))
        print(f"teacher {name}: {len(parts)} file(s), {len(ids)} studies ({len(ids.intersection(base.index))} in the base), "
              f"findings {labs if len(labs) < 12 else 'all 12'}")
    comb, info = combine(base, teachers, silent)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    comb.reset_index().rename(columns={"index": "StudyInstanceUID"}).to_csv(a.out, index=False)

    moved = []
    for lab in LABELS:
        d = (comb[lab] - base[lab]).dropna()
        top = d.abs().sort_values(ascending=False).head(a.audit_top)
        moved += [{"finding": lab, "StudyInstanceUID": u, "base": float(base.loc[u, lab]),
                   "student": float(comb.loc[u, lab])} for u in top.index if top[u] > 1e-9]
    pd.DataFrame(moved).to_csv(a.out.with_suffix(".audit.csv"), index=False)

    report = {"base": str(a.base), "teachers": [{"name": n, "findings": l} for n, _, l in teachers], "per_finding": info}
    if a.gold:
        g = pd.read_csv(a.gold).dropna(subset=LABELS).set_index("StudyInstanceUID")[LABELS].astype(float)
        idx = g.index.intersection(base.index)
        report["gold"] = {"n": int(len(idx)), "base_macro": round(macro(g.loc[idx], base.loc[idx]), 4),
                          "student_macro": round(macro(g.loc[idx], comb.loc[idx]), 4)}
        print(f"gold ({len(idx)}): base {report['gold']['base_macro']} -> student targets {report['gold']['student_macro']}")
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    for lab, v in info.items():
        print(f"  {lab:17s} {v['with_a_teacher']:5d} of {v['studies']} studies have a teacher; "
              f"{v['level_changed']} change level"
              + (f"; {v['teacher_alone']} take the teacher alone (base levels {v['base_silent_levels']} are a fill)"
                 if "teacher_alone" in v else ""))
    print(f"wrote {a.out}, {a.out.with_suffix('.audit.csv').name}, {a.out.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
