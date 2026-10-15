r"""OAI baseline MOAKS reads -> the competition's twelve findings, as one soft target per knee.

    python labeling/oai_moaks_targets.py \
        --moaks ~/oai_data/mri_assessments/"Semi-Quant Scoring_ASCII"/kMRI_SQ_MOAKS_BICL00.txt \
        --out ~/oai_data/derived/oai_moaks_targets_v1.csv

The OAI tables are under the owner's NDA Data Use Certification (DAR 26933, to 2027-10-15): the
input and the output are subject-level data. They stay in ~/oai_data and off GitHub and Kaggle: this
script refuses any work tree with a remote, and a remote-less one unless git ignores the path. Only
the code is in the repo.

Each cell is P(the competition's criterion is met | the MOAKS grades), with constants fixed here from
the two definitions (REFERENCE.md §2 at 9c23daf; MOAKS: Hunter et al. 2011, OAI's
kMRI_SQ_MOAKS_BICL_Descrip.pdf). Nothing is fitted to any label of ours, gold included.

| finding | MOAKS input | rule |
|---|---|---|
| Medial / Lateral / PF OA | cartilage per subregion: size of any loss (0-3) . full-thickness extent (0-3) | max over the compartment's subregions; full thickness on 10%+ of a subregion is the criterion's ">50% depth over ~1 cm"; partial-depth loss is not graded by MOAKS, so it stays uncertain |
| Medial / Lateral Meniscus | tear type per horn (0-8), root tear | a MOAKS tear reaches the surface and maceration is a morphologic abnormality: positive; "signal abnormality, not a tear" is the criterion's intrasubstance negative |
| ACL | 0 none, 1 partial, 2 complete; repair | complete positive; partial straddles the high-grade cut; a reconstructed knee is masked |
| Effusion | whole-knee effusion-synovitis 0-3 | the criterion's own cut: moderate or large |
| Synovitis | Hoffa-synovitis 0-3 and effusion-synovitis 0-3 | non-enhanced MRI cannot separate synovitis from fluid (OAI's own note); the least certain row here |
| Baker's | popliteal cyst yes / no (one project only) | presence without size: the criterion needs moderate or large |
| Contusion | bone marrow lesion size in 15 subregions | no lesion anywhere: negative; any lesion: masked (in an OA cohort a lesion is mostly degenerative, not impact) |
| MCL, Fracture | not scored by MOAKS | masked |

Knees read by two projects get the mean of their reads, cell by cell. Projects score different
blocks: cartilage, lesions and menisci on all 2,400 baseline knees; ACL, effusion and synovitis on
projects 22 and 65; the popliteal cyst on project 22 alone. A cell no read scored is NaN.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

COMPARTMENTS = {  # MOAKS Descrip §1.7.1; the trochlea (FMA, FLA) belongs to the patello-femoral joint
    "Medial OA": ["FMC", "FMP", "TMA", "TMC", "TMP"],
    "Lateral OA": ["FLC", "FLP", "TLA", "TLC", "TLP"],
    "PF OA": ["PM", "PL", "FMA", "FLA"],
}
BML_REGIONS = ["FMA", "FLA", "FMC", "FLC", "FMP", "FLP", "SS", "TMA", "TLA", "TMC", "TLC", "TMP", "TLP",
               "PM", "PL"]

# P(>50% depth over ~1 cm | one subregion's grades). Full-thickness extent first, then any-depth size.
P_CART_FT = {1: 0.60, 2: 0.92, 3: 0.97}        # <10%, 10-75%, >75% of the subregion full thickness
P_CART_SIZE = {0: 0.02, 1: 0.08, 2: 0.30, 3: 0.45}  # no full thickness: depth of the loss unknown
P_MENISCUS = {0: 0.02, 1: 0.08}                  # 2-8 (tears, maceration): P_MEN_TEAR
P_MEN_TEAR = 0.93
P_ACL = {0: 0.02, 1: 0.45, 2: 0.97}
P_EFFUSION = {0: 0.02, 1: 0.10, 2: 0.90, 3: 0.98}
P_SYN_HOFFA = {0: 0.03, 1: 0.20, 2: 0.60, 3: 0.85}
P_SYN_EFF = {0: 0.02, 1: 0.10, 2: 0.45, 3: 0.70}
P_BAKERS = {0: 0.03, 1: 0.55}
P_NO_BML = 0.03


def grade(s: pd.Series) -> pd.Series:
    """'2.1: 10-75% area, 1-10% full thickness' -> 2.1; '.: Missing Form' -> NaN."""
    head = s.astype(str).str.split(":", n=1).str[0].str.strip()
    return pd.to_numeric(head, errors="coerce")


def cartilage(row_size: np.ndarray, row_ft: np.ndarray) -> np.ndarray:
    p = np.where(row_ft >= 1, np.vectorize(lambda f: P_CART_FT.get(int(f), np.nan))(np.nan_to_num(row_ft)),
                 np.vectorize(lambda s: P_CART_SIZE.get(int(s), np.nan))(np.nan_to_num(row_size)))
    p[np.isnan(row_size)] = np.nan
    return p


def per_read(d: pd.DataFrame) -> pd.DataFrame:
    v = lambda c: grade(d["V00" + c])
    out = pd.DataFrame(index=d.index, columns=LABELS, dtype=float)

    for lab, regs in COMPARTMENTS.items():
        g = np.stack([v("MCM" + r).to_numpy() for r in regs], 1)            # e.g. 2.1
        size, ft = np.floor(g), np.round((g - np.floor(g)) * 10)
        p = cartilage(size, ft)
        out[lab] = np.where(np.isnan(p).all(1), np.nan, np.nanmax(np.where(np.isnan(p), -1, p), 1))

    for lab, side in [("Medial Meniscus", "M"), ("Lateral Meniscus", "L")]:
        t = np.stack([v(f"MMT{side}{h}").to_numpy() for h in "ABP"], 1)
        root = v("MMRT" + side).to_numpy()
        worst = np.where(np.isnan(t).all(1), np.nan, np.nanmax(np.where(np.isnan(t), -1, t), 1))
        p = np.where(worst >= 2, P_MEN_TEAR, np.vectorize(lambda x: P_MENISCUS.get(int(x), np.nan))(np.nan_to_num(worst)))
        p = np.where(root == 1, P_MEN_TEAR, p)
        out[lab] = np.where(np.isnan(worst) & (root != 1), np.nan, p)

    acl, repair = v("MACLTR"), v("MACLRP")
    out["ACL"] = acl.map(P_ACL).where(repair != 1)

    eff, hoffa = v("MEFFWK"), v("MSYIC")
    out["Effusion"] = eff.map(P_EFFUSION)
    syn = pd.concat([hoffa.map(P_SYN_HOFFA), eff.map(P_SYN_EFF)], axis=1)
    out["Synovitis"] = syn.max(axis=1, skipna=False).where(hoffa.notna() & eff.notna())

    out["Baker's"] = v("MPOPCYS").map(P_BAKERS)

    bml = np.stack([v("MBMS" + r).to_numpy() for r in BML_REGIONS], 1)
    all_read = ~np.isnan(bml).any(1)
    out["Contusion"] = np.where(all_read & (np.nan_to_num(bml).max(1) == 0), P_NO_BML, np.nan)

    out["MCL"] = np.nan
    out["Fracture"] = np.nan
    return out


def unsafe_for_oai(p: Path) -> str | None:
    """Why `p` may not hold OAI-derived data, or None. Never inside a work tree that has a remote (a
    push moves it to a third-party system); inside a remote-less one only where git ignores it."""
    git = lambda *a: subprocess.run(["git", "-C", str(p), *a], capture_output=True, text=True)
    if git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        return None
    if git("remote").stdout.strip():
        return f"{p} is inside a git work tree with a remote"
    if git("check-ignore", "-q", str(p / "_probe.csv")).returncode != 0:
        return f"{p} is inside a git work tree and not ignored"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--moaks", type=Path, required=True, help="kMRI_SQ_MOAKS_BICL00.txt (baseline reads)")
    ap.add_argument("--out", type=Path, required=True, help="a CSV outside any git work tree")
    a = ap.parse_args()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    why = unsafe_for_oai(a.out.parent.resolve())
    if why:
        raise SystemExit(f"{why}: OAI-derived data stay out of every repo (DUC §3)")

    d = pd.read_csv(a.moaks, sep="|", low_memory=False, encoding="latin-1", dtype=str)
    d["side"] = d["SIDE"].str.extract(r"(Right|Left)")[0].str[0]
    reads = pd.concat([d[["ID", "side", "READPRJ"]], per_read(d)], axis=1)

    knees = reads.groupby(["ID", "side"])
    out = knees[LABELS].mean()
    out.insert(0, "n_reads", knees.size())
    out.insert(1, "projects", knees["READPRJ"].agg(lambda s: "+".join(sorted(s))))
    out = out.reset_index().rename(columns={"ID": "oai_id"})
    out.to_csv(a.out, index=False)

    print(f"{len(d)} reads -> {len(out)} knees ({(out.n_reads > 1).sum()} read twice); wrote {a.out}")
    print(f"{'finding':18s} {'cells':>6s} {'mean':>6s} {'>=0.5':>6s}")
    for lab in LABELS:
        c = out[lab].dropna()
        print(f"{lab:18s} {len(c):6d} {c.mean() if len(c) else float('nan'):6.3f} "
              f"{(c >= 0.5).mean() if len(c) else float('nan'):6.3f}")


if __name__ == "__main__":
    main()
