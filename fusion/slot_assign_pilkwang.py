"""Reproduce the fork's SIX-SLOT assignment offline, from the published header parquet.

WHY THIS EXISTS. `IMPROVEMENTS.md` and the project memory both record that the fork's slots are
**not reconstructable**:

    `Fluid_Sensitive` and `Fat_Suppression` are byte-identical over all 24,371 series, so the
    fork's six slots (SAG_FLUID_FS ... SAG_T1) are not reconstructable from the competition
    metadata -- those columns give exactly 3 planes x 2.

That is true of the **competition** metadata and false of what is on disk. `annotate()` in
`pilkwang/rsna-knee-baseline-v1` does not read `Fluid_Sensitive` at all. It recovers fat
suppression and pulse-sequence weighting from seven raw DICOM header fields --
`SeriesDescription`, `SequenceName`, `ScanOptions`, `ScanningSequence`, `RepetitionTime`,
`EchoTime`, `PixelSpacing` -- and **every one of them is a column of
`data/external/dicom_headers_zhukovoleksiy.parquet`**, which has been on disk since 2026-08-10.
The old note anticipated this as a *fallback* ("if the reproduction gate misses, recover the finer
split from SeriesDescription/EchoTime"); it is in fact the primary path and it is sufficient.

This matters because feeding their frozen members requires their slot assignment. A T1 series
landing in a fluid slot is not a small error: `pick_slots` documents that relaxing the predicate
would put one series in two slots for 2,383 of 4,407 studies and leave 56% of the T1 slot holding
PD or T2, and the presence mask would then assert an acquisition that was never made.

    python fusion/slot_assign_pilkwang.py            # write data/slots_pilkwang.csv + fill report

VERIFICATION, and it is the point of running this before any GPU. Their own comment says the
fat-suppressed fluid-sensitive series "exist for nearly every study; the T1 and the non-suppressed
fluid-sensitive series are scarcer, which is what the presence mask is for". Fill rates that
contradict that shape mean the recovery is wrong, and it is far cheaper to learn that here than
from a member scoring 0.6.

Transcribed verbatim from the fork rather than re-derived: the regexes, the token match on
`ScanOptions` (GE writes `SAT_GEMS` for spatial saturation, so a substring test on "SAT" fires on
non-fat-sat series), the `np.where` cascade for `weight`, and the tie-break toward the stack with
the most slices.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
D = PROJ / "data"

# --- verbatim from pilkwang/rsna-knee-baseline-v1, lines 870-920 ------------------------------ #

SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
                        r"water excit|\btirm\b|\bsting\b|\bfatsup\b")
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Recover fat suppression and pulse-sequence weighting from the header.

    Transcribed from the fork. The one deviation is mechanical: the parquet stores
    `PixelSpacing` already split, so the `str.split("|").str[0]` there is guarded rather than
    assumed, and a non-finite spacing is left as NaN for the caller to refuse.
    """
    df = df.copy()
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)

    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs

    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = (desc.str.contains(_T1_RX), desc.str.contains(_T2_RX),
                   desc.str.contains(_PD_RX))

    df["weight"] = np.where(t1 & ~t2 & ~pdw, "T1",
                     np.where(t2 & ~pdw, "T2",
                       np.where(pdw, "PD",
                         np.where(gre, "GRE",
                           np.where(tr < 800, "T1",
                             np.where(te > 60, "T2",
                               np.where(tr >= 800, "PD", "UNK")))))))
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])

    ps = df["PixelSpacing"].astype(str)
    df["px"] = pd.to_numeric(ps.str.split("|").str[0].replace({"": None, "None": None,
                                                               "nan": None}),
                             errors="coerce")
    return df


def pick_slots(series_df: pd.DataFrame, slots=SLOTS_RECOVERED) -> pd.DataFrame:
    """One series per slot per study; ties broken toward the thickest stack.

    Returns long form -- one row per (study, slot) that is FILLED. An absent row is an absent
    slot, which is what the presence mask expresses. No fallback is applied: `RULES_NATIVE` sets
    `slot_fallback: False`, and the 20 members all carry the native rules.
    """
    rows = []
    for study, g in series_df.groupby("StudyInstanceUID", sort=False):
        for name, plane, fluid, fs in slots:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            if fluid is not None:
                sel &= (g["fluid"] == fluid)
            cand = g[sel]
            if not len(cand):
                continue
            best = cand.sort_values("n_slices", ascending=False).iloc[0]
            rows.append({"StudyInstanceUID": study, "slot": name,
                         "SeriesInstanceUID": best["SeriesInstanceUID"],
                         "plane": plane, "n_slices": int(best["n_slices"]),
                         "px": float(best["px"]) if np.isfinite(best["px"]) else np.nan,
                         "weight": best["weight"], "fatsat": bool(best["fatsat"])})
    return pd.DataFrame(rows)


def main() -> int:
    hdr = pd.read_parquet(D / "external" / "dicom_headers_zhukovoleksiy.parquet")
    series = pd.read_csv(D / "train_series.csv")
    plane_map = dict(zip(series["SeriesInstanceUID"], series["Anatomical_Plane"]))

    hdr = annotate(hdr)
    hdr["plane"] = hdr["SeriesInstanceUID"].map(plane_map)
    n_plane = int(hdr["plane"].notna().sum())
    print(f"headers {len(hdr):,} series, plane known for {n_plane:,} "
          f"({n_plane / len(hdr):.1%})")

    print("\nrecovered weighting x fat suppression, over all series with a plane:")
    tab = pd.crosstab(hdr.loc[hdr["plane"].notna(), "weight"],
                      hdr.loc[hdr["plane"].notna(), "fatsat"])
    print(tab.to_string())

    slots = pick_slots(hdr[hdr["plane"].notna()])
    n_study = slots["StudyInstanceUID"].nunique()
    print(f"\nassigned {len(slots):,} slots over {n_study:,} studies")

    print("\nFILL RATE PER SLOT -- the check. Their comment: the fat-suppressed "
          "fluid-sensitive\nseries exist for nearly every study; T1 and non-suppressed "
          "fluid-sensitive are scarcer.")
    for name, _, _, _ in SLOTS_RECOVERED:
        k = int((slots["slot"] == name).sum())
        print(f"  {name:<16} {k:>6,} / {n_study:,}  ({k / n_study:6.1%})")

    per = slots.groupby("StudyInstanceUID").size()
    print(f"\nslots per study: mean {per.mean():.2f}, "
          f"min {per.min()}, max {per.max()}")
    print("  " + "  ".join(f"{k}:{v}" for k, v in
                           sorted(per.value_counts().items())))

    out = D / "slots_pilkwang.csv"
    slots.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
