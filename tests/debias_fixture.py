"""Offline test of build_public_blend_kernel's de-bias cell on synthetic DICOM headers: no Kaggle, no model.

    python tests/debias_fixture.py
"""
from __future__ import annotations

import base64
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "kaggle"))
sys.path.insert(0, str(PROJ / "pipeline"))
import build_public_blend_kernel as P  # noqa: E402
import header_prior as HP  # noqa: E402

L = HP.LABELS
SITES = [("SIEMENS", "Skyra", "VE11", "3.0"), ("GE MEDICAL SYSTEMS", "Signa", "27.0", "1.5"),
         ("Philips", "Ingenia", "5.6", "3.0")]


def write_split(root: Path, split: str, n: int, rng: np.random.RandomState) -> pd.Series:
    rows = []
    site_of = {}
    for i in range(n):
        s = i % len(SITES)
        study = f"1.2.3.{split}.{i}"
        site_of[study] = s
        for j, (plane, desc) in enumerate([("Sagittal", f"PD FS SAG {s}"), ("Coronal", "T1 COR")]):
            series = f"{study}.{j}"
            rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series, "Fluid_Sensitive": int(j == 0),
                         "Fat_Suppression": int(j == 0), "Anatomical_Plane": plane})
            d = root / f"{split}_series" / study / series
            d.mkdir(parents=True)
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = FileDataset(str(d / "1.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
            ds.Manufacturer, ds.ManufacturerModelName, ds.SoftwareVersions, ds.MagneticFieldStrength = SITES[s]
            ds.ReceiveCoilName = f"Knee{s}"
            ds.SeriesDescription = desc
            ds.RepetitionTime, ds.EchoTime = 3000 + 100 * s + rng.randint(50), 30 + j
            ds.SliceThickness, ds.PixelBandwidth, ds.EchoTrainLength, ds.FlipAngle, ds.Rows = 3.0, 200, 8, 90, 512
            ds.save_as(d / "1.dcm", enforce_file_format=True)
    pd.DataFrame(rows).to_csv(root / f"{split}_series.csv", index=False)
    return pd.Series(site_of)


def run(cell: str, root: Path, work: Path, inp: Path, commit: bool) -> dict:
    src = cell.replace("/kaggle/working", str(work)).replace('"/kaggle/input"', repr(str(inp)))
    g = {"NB_T0": __import__("time").time(), "DATA": root, "_P_COMMIT": commit, "_P_READ": False}
    exec(compile(src, "debias", "exec"), g)
    return g


def main() -> None:
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        root, work, inp = t / "comp", t / "working", t / "input" / "table"
        for p in (root, work, inp):
            p.mkdir(parents=True)
        site_tr = write_split(root, "train", 240, rng)
        site_te = write_split(root, "test", 150, rng)
        tab = HP.read_split(root, "train")
        tab.to_csv(inp / "train_series_headers.csv", index=False)
        base = {0: 0.15, 1: 0.45, 2: 0.75}
        Y = pd.DataFrame({c: (rng.rand(len(site_tr)) < site_tr.map(base).to_numpy()).astype(float) for c in L},
                         index=pd.Index(site_tr.index, name="StudyInstanceUID"))
        Y.loc[Y.index[::2], "Synovitis"] = np.nan
        Y.reset_index().to_csv(inp / "report_labels.csv", index=False, float_format="%.0f")
        shas = {f: hashlib.sha256((inp / f).read_bytes()).hexdigest() for f in ("train_series_headers.csv", "report_labels.csv")}
        src = (PROJ / "pipeline" / "header_prior.py").read_bytes()

        def cell(sh=shas):
            return (P.DEBIAS.replace("__MARGIN_MIN__", "25").replace("__H_W__", repr(HP.DEBIAS_W))
                    .replace("__H_SHAS__", repr(sh)).replace("__H_KERNEL__", "fixture")
                    .replace("__H_SRC__", repr(base64.b64encode(src).decode()))
                    .replace("__H_SRC_SHA__", hashlib.sha256(src).hexdigest()).replace("__BUILT_FROM__", "fixture"))

        ids = list(site_te.index)
        sub = pd.DataFrame({"StudyInstanceUID": ids} | {c: rng.rand(len(ids)) + 0.3 * site_te.to_numpy() for c in L})
        sub.to_csv(work / "submission.csv", index=False)
        before = (work / "submission.csv").read_bytes()
        n = 0

        g = run(cell(), root, work, inp.parent, commit=True)
        assert g["H_OK"], "commit run did not finish"
        out = pd.read_csv(work / "submission.csv")
        prior = g["_h_prior"]
        for c in L:
            r_in = pd.Series(prior[c].to_numpy()).corr(sub[c], method="spearman")
            r_out = pd.Series(prior[c].to_numpy()).corr(out[c], method="spearman")
            assert r_out < r_in, (c, r_in, r_out)
            assert 0.0 <= out[c].min() and out[c].max() <= 1.0
        n += 1
        moved = {c: 1 - out[c].corr(sub[c], method="spearman") for c in ("ACL", "Synovitis")}
        assert moved["Synovitis"] > moved["ACL"], moved
        n += 1
        print(f"ok  de-bias lowers Spearman(prior, submission) on all 12; Synovitis moves most {moved}")

        (work / "submission.csv").write_bytes(before)
        g = run(cell({k: "0" * 64 for k in shas}), root, work, inp.parent, commit=False)
        assert not g["H_OK"] and (work / "submission.csv").read_bytes() == before
        n += 1
        print("ok  a wrong table sha256: H_FAILED, the submission byte for byte")

        bad = HP.load_table(inp / "train_series_headers.csv")
        bad.loc[bad.StudyInstanceUID == sorted(bad.StudyInstanceUID.unique())[0], "SoftwareVersions"] = "XX"
        alt = t / "input2" / "table"
        alt.mkdir(parents=True)
        bad.to_csv(alt / "train_series_headers.csv", index=False)
        (alt / "report_labels.csv").write_bytes((inp / "report_labels.csv").read_bytes())
        sh2 = {f: hashlib.sha256((alt / f).read_bytes()).hexdigest() for f in shas}
        try:
            run(cell(sh2), root, work, alt.parent, commit=True)
            raise AssertionError("a table that differs from the DICOMs passed the parity check")
        except RuntimeError as e:
            assert "HEADER_PARITY" in str(e), e
        n += 1
        print("ok  a table that disagrees with the DICOMs: HEADER_PARITY fails the commit run")

        for d in list((root / "test_series").iterdir())[:30]:
            for f in d.rglob("*.dcm"):
                f.unlink()
        (work / "submission.csv").write_bytes(before)
        g = run(cell(), root, work, inp.parent, commit=False)
        assert not g["H_OK"] and (work / "submission.csv").read_bytes() == before
        n += 1
        print("ok  20% of test studies unreadable: H_FAILED, the submission byte for byte")
        print(f"DEBIAS FIXTURE: {n} checks passed")


if __name__ == "__main__":
    main()
