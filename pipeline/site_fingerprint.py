"""Per-study scanner fingerprint, for site-grouped folds.

`IMPROVEMENTS.md` §2i-a: a public probe measured **0.053** of macro AUC as pure site
memorisation — DICOM headers with no pixels score 0.6516 under random folds and 0.5981 grouped
on a scanner fingerprint. `data/folds.csv` is ungrouped, so every OOF number this project has
produced carries an unknown share of that. The reproduction gate is the casualty: it compares
our number against someone else's, and the inflation is the same size as the effects we intend
to measure.

The headers come from `data/external/dicom_headers_zhukovoleksiy.parquet` — the published
output of `zhukovoleksiy/rsna-metadata-probe`, 24,371 series x 43 fields. **We did not have to
run a Kaggle notebook for this**, which is README rule 4 paying for itself a third time.

Two details that are the whole of the correctness here, and both were found by reading that
notebook's code rather than its post:

  - **`ImagingFrequency` must be rounded to 3 decimals.** It is recorded to six, and the same
    scanner drifts in the last three: `63.685259` and `63.685256` are one Siemens Avanto. Left
    raw it shatters the corpus into 3,249 fingerprints with 2,667 singletons, and a "grouped"
    fold built on it would be barely grouped at all -- the failure would be silent and would
    look like a working guard.
  - **Aggregate per study by column mode, not by picking one series' fingerprint.** 90% of
    studies carry series whose raw fingerprints disagree, again mostly on that float.

Writes `data/site_fingerprint.csv`: StudyInstanceUID, site_id, fingerprint.
"""
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
D = PROJ / "data"
HEADERS = D / "external" / "dicom_headers_zhukovoleksiy.parquet"
OUT = D / "site_fingerprint.csv"

CATS = ["Manufacturer", "ManufacturerModelName", "SoftwareVersions", "ReceiveCoilName"]
FREQ = "ImagingFrequency"
FREQ_DP = 3          # see module docstring -- this constant is load-bearing


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None


def build(headers: Path = HEADERS) -> pd.DataFrame:
    if not headers.exists():
        sys.exit(f"{headers} not found.\n"
                 f"  kaggle kernels output zhukovoleksiy/rsna-metadata-probe -p {headers.parent}\n"
                 f"  then rename headers.parquet to {headers.name}")
    d = pd.read_parquet(headers)
    g = d.groupby("StudyInstanceUID")

    agg = pd.DataFrame({c: g[c].agg(_mode) for c in CATS})
    # stored as text in the parquet; coerce before any arithmetic touches it
    freq = pd.to_numeric(d[FREQ], errors="coerce")
    agg["freq"] = freq.groupby(d["StudyInstanceUID"]).median().round(FREQ_DP)

    # Every field must be forced to a real str BEFORE concatenation. The parquet's string
    # columns are pyarrow-backed, so `.astype(str)` leaves pd.NA intact and `pd.NA + "|"` is
    # pd.NA -- one missing field silently nulls the whole fingerprint. That is not cosmetic:
    # pd.factorize maps NaN to -1, so on the first version of this file **1,077 studies (24% of
    # the corpus) collapsed into a single group** and `--group-by site` would have forced a
    # quarter of the data into one fold while reporting a healthy-looking 215 groups. It is the
    # exact failure this module's docstring warns about, committed in the module itself.
    def s(col) -> pd.Series:
        return agg[col].map(lambda v: "NA" if pd.isna(v) else str(v))

    fp = (s("Manufacturer") + "|" + s("ManufacturerModelName") + "|"
          + s("SoftwareVersions") + "|" + s("freq") + "|" + s("ReceiveCoilName"))
    assert fp.notna().all(), "fingerprint still nullable -- see the comment above"

    out = pd.DataFrame({"StudyInstanceUID": fp.index, "fingerprint": fp.values})
    out["site_id"] = pd.factorize(out.fingerprint)[0]
    assert (out.site_id >= 0).all(), "factorize produced -1: a null fingerprint survived"
    return out.reset_index(drop=True)


def main() -> None:
    out = build()
    vc = out.fingerprint.value_counts()
    print(f"{len(out):,} studies  ->  {len(vc)} distinct scanner fingerprints")
    print(f"  top 20 cover {vc.head(20).sum() / len(out):.1%} of studies")
    print(f"  singletons {int((vc == 1).sum())}  ·  median group {int(vc.median())}  "
          f"·  largest {int(vc.max())}")
    # The published probe reports 265 fingerprints and 45.5% in the top 20. Landing far from
    # that means the rounding or the aggregation has drifted -- see the module docstring.
    if not (200 <= len(vc) <= 340):
        print(f"  WARNING: expected ~265 fingerprints, got {len(vc)}. Check {FREQ} rounding.")
    print(f"\n{vc.head(8).to_string()}")
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(PROJ)}")


if __name__ == "__main__":
    main()
