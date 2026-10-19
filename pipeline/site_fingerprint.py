"""Per-study scanner fingerprint, used for site-grouped folds."""
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
D = PROJ / "data"
HEADERS = D / "external" / "dicom_headers_zhukovoleksiy.parquet"
OUT = D / "site_fingerprint.csv"

CATS = ["Manufacturer", "ManufacturerModelName", "SoftwareVersions", "ReceiveCoilName"]
FREQ = "ImagingFrequency"
FREQ_DP = 3


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
    freq = pd.to_numeric(d[FREQ], errors="coerce")
    agg["freq"] = freq.groupby(d["StudyInstanceUID"]).median().round(FREQ_DP)

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
    if HEADERS.exists():
        sys.path.insert(0, str(PROJ / "pipeline"))
        from provenance import check
        check(HEADERS, "kaggle kernels output zhukovoleksiy/rsna-metadata-probe",
              accept_new="--accept-new" in sys.argv)
    out = build()
    vc = out.fingerprint.value_counts()
    print(f"{len(out):,} studies  ->  {len(vc)} distinct scanner fingerprints")
    print(f"  top 20 cover {vc.head(20).sum() / len(out):.1%} of studies")
    print(f"  singletons {int((vc == 1).sum())}  ·  median group {int(vc.median())}  "
          f"·  largest {int(vc.max())}")
    if not (200 <= len(vc) <= 340):
        print(f"  WARNING: expected ~265 fingerprints, got {len(vc)}. Check {FREQ} rounding.")
    print(f"\n{vc.head(8).to_string()}")
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(PROJ)}")


if __name__ == "__main__":
    main()
