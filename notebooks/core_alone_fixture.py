"""(B) the core alone, off Kaggle: both builds, run AS BUILT on CPU over synthetic DICOM studies.

    .venv/bin/python notebooks/core_alone_fixture.py [--work <dir>]

Needs the pinned public notebook (`data/public_notebooks/bend-the-knee-2fc5a420/`) and the public core
weights (`data/external/raptor/raptor_ft_coatnet_v5_full_swa.pt`, CC0). No GPU, no network, no competition
DICOMs (the studies are synthetic: this tests the plumbing, not the anatomy). A few minutes on the M5.

WHAT IT CHECKS, IN ORDER, AND HOW EACH PART COULD FAIL.
  1. Both builds (`--core-alone` with a student, and without: the public weights). The cells are the intro,
     public cells 1-3 verbatim, the core cell (student only), the core-alone cell and its check. Public
     cell -2's Raptor code sits in the core-alone cell verbatim. No DINOv2 / A5 / Rad / publish cell rides
     along. The only sources are the public core's dataset and, for the student, its kernel.
  2. A synthetic competition root: four studies of small DICOM series written in shuffled file order, one
     study without a second coronal series. test.csv, test_series.csv and sample_submission.csv carry the
     competition's columns.
  3. The student build runs as built. Only the /kaggle paths, the competition root and the two T4s (two CPU
     devices here, in two threads, as on Kaggle) are swapped. It must give a submission.csv with every study
     in test.csv order and ranks in [0, 1], and its check must pass.
  4. Index parity: every study is re-scored one at a time on one device by the same public functions,
     forward and reversed, blended 6:1 and ranked. The result must equal the submission. A mis-split, where
     study i's scores land on study j, fails here.
  5. The public build runs the same way and loads the public file at its pinned sha256. The CONTROL: it must
     score differently from the student, or step 4 could pass on the wrong weights.
  6. The check refuses a load of another file, a missing load, another sha256, and a failed study in a
     commit-sized run.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

PROJ = Path(__file__).resolve().parents[1]
PUB_DIR = PROJ / "data" / "public_notebooks" / "bend-the-knee-2fc5a420"
PUB_V5 = PROJ / "data" / "external" / "raptor" / "raptor_ft_coatnet_v5_full_swa.pt"
META = PROJ / "fusion" / "runs_coatnet_raptor_check" / "last_meta.json"   # a real Stage A CoAtNet run's
BUILDER = PROJ / "notebooks" / "build_submit_kernel.py"
KERNEL = "raahimnawaz/fixture-coatnet-student"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# plane -> (row direction, column direction): the series normal is their cross product
IOP = {"Sagittal": ([0, 1, 0], [0, 0, -1]), "Coronal": ([1, 0, 0], [0, 0, -1]), "Axial": ([1, 0, 0], [0, 1, 0])}
SERIES = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", 1)]


def sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(BUILDER), "--public-dir", str(PUB_DIR), "--core-alone",
                           "--out", str(out), *extra], capture_output=True, text=True)


def refuses(fn, what: str) -> None:
    try:
        fn()
    except RuntimeError as e:
        print(f"  refused, as it must: {what} -- {str(e)[:110]}")
        return
    raise SystemExit(f"FIXTURE FAILED: did not refuse {what}")


def write_series(d: Path, plane: str, n: int, rng: np.random.Generator, size: int = 96) -> None:
    """n MR slices with real geometry tags, file names shuffled so only the geometry gives the order."""
    d.mkdir(parents=True)
    row, col = (np.asarray(v, float) for v in IOP[plane])
    normal = np.cross(row, col)
    yy, xx = np.mgrid[:size, :size]
    names = [f"{k:03d}.dcm" for k in rng.permutation(n)]
    for k in range(n):
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = MRImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = fm
        ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = MRImageStorage, fm.MediaStorageSOPInstanceUID, "MR"
        ds.Rows = ds.Columns = size
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.PixelSpacing = [0.5, 0.5]
        ds.ImageOrientationPatient = [*row.tolist(), *col.tolist()]
        ds.ImagePositionPatient = (normal * 3.0 * k).tolist()
        ds.InstanceNumber = k + 1
        cy, cx = rng.uniform(25, 70, 2)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.uniform(6, 18) ** 2))
        img = 300 + 2500 * blob * (1 + 0.3 * np.sin(k)) + rng.normal(0, 120, (size, size))
        ds.PixelData = np.clip(img, 0, 4095).astype(np.uint16).tobytes()
        pydicom.dcmwrite(d / names[k], ds, enforce_file_format=True)


def make_root(root: Path) -> list[str]:
    rng = np.random.default_rng(7)
    ids, rows = [], []
    for s in range(4):
        study = generate_uid()
        ids.append(study)
        for plane, fluid in SERIES[:3] + SERIES[4:] if s == 3 else SERIES:      # study 4: no 2nd coronal
            series = generate_uid()
            write_series(root / "test_series" / study / series, plane, 5, rng)
            rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series, "Fluid_Sensitive": fluid,
                         "Fat_Suppression": fluid, "Anatomical_Plane": plane})
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(root / "test.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "test_series.csv", index=False)
    pd.DataFrame([[i] + [0.5] * 12 for i in ids], columns=["StudyInstanceUID"] + LABELS).to_csv(
        root / "sample_submission.csv", index=False)
    return ids


def patched(src: str, swaps: list[tuple[str, str]], what: str) -> str:
    for old, new in swaps:
        if old not in src:
            raise SystemExit(f"FIXTURE FAILED: {what} no longer contains {old[:60]!r}")
        src = src.replace(old, new)
    return src


def run_build(built: Path, work: Path, root: Path, tag: str) -> tuple[dict, list[str]]:
    """Exec the built notebook's cells in one namespace, as Kaggle would, with the paths and devices swapped."""
    kmeta = json.loads((built / "kernel-metadata.json").read_text())
    srcs = ["".join(c["source"]) for c in json.loads((built / kmeta["code_file"]).read_text())["cells"]]
    kw = work / f"kaggle_working_{tag}"
    kw.mkdir()
    paths = [("/kaggle/working", str(kw))]
    gpu = re.search(r"^DEVS = \[\]\n.*?^print\(f'devices: .*?\n", srcs[-3], re.S | re.M)
    if gpu is None:
        raise SystemExit("FIXTURE FAILED: public cell 3's GPU preflight moved")
    cells = {"assets": srcs[1], "dense": srcs[-4], "rsna": srcs[-3].replace(gpu.group(0), "DEVS = []\n"),
             "alone": srcs[-2], "check": srcs[-1]}
    cells["rsna"] = patched(cells["rsna"], paths, "public cell 3")
    cells["alone"] = patched(cells["alone"], paths + [(
        '_AL_DEVICES = [_al_torch.device(f"cuda:{_i}") for _i in range(2)]',
        '_AL_DEVICES = [_al_torch.device("cpu"), _al_torch.device("cpu")]')], "the core-alone cell")
    cells["check"] = patched(cells["check"], paths, "the check cell")
    order = ["assets", "dense", "rsna", "alone", "check"]
    if len(srcs) == 7:
        cells["core"] = patched(srcs[2], [('_CPath("/kaggle/input")', f'_CPath({str(work / "input")!r})'),
                                          ('_CPath("/tmp/student_core")', f'_CPath({str(work / "student_core")!r})')],
                                "the core cell")
        order.insert(1, "core")
    ns: dict = {"__name__": "__main__"}
    os.environ["RSNA_COMP_ROOT"] = str(root)
    cwd = os.getcwd()
    try:
        for name in order:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(cells[name], f"<{tag}:{name}>", "exec"), ns)
            if name == "assets":                  # the attachments, before any lookup (the core cell appends its own)
                ns["ASSET_ROOTS"][:] = [str(work / "public_assets")]
            if name in ("core", "alone", "check"):
                print("   " + "\n   ".join(l for l in buf.getvalue().splitlines() if l.startswith(("[core]", "[alone]"))))
    finally:
        os.chdir(cwd)
    ns["_fixture_cells"], ns["_fixture_kw"] = cells, kw
    return ns, srcs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=None, help="scratch dir (default: a new temp dir)")
    a = ap.parse_args()
    work = a.work or Path(tempfile.mkdtemp(prefix="core_alone_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    torch.manual_seed(0)

    # ---- a student last.pt as train_windows saves one (our key names, fp16), moved off the public weights
    pub = torch.load(PUB_V5, map_location="cpu", weights_only=False)
    ren = {"clsW": "w", "clsb": "b"}
    sd = {("encoder." + k[len("backbone."):]) if k.startswith("backbone.") else ren.get(k, k): v.clone()
          for k, v in pub["model"].items()}
    sd["w"] += 0.05 * torch.randn_like(sd["w"])
    sd["att.3.weight"] += 0.05 * torch.randn_like(sd["att.3.weight"])
    run = work / "input" / "fixture-coatnet-student" / "runs_coatnet"
    run.mkdir(parents=True)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in sd.items()}, run / "last.pt")
    (run / "last_meta.json").write_text(json.dumps(dict(json.loads(META.read_text()), zoom=0.0), indent=1))
    s_sha = sha(run / "last.pt")
    (work / "public_assets" / "raptor-knee-maxspan").mkdir(parents=True)
    (work / "public_assets" / "raptor-knee-maxspan" / PUB_V5.name).symlink_to(PUB_V5)

    # ---- 1. both builds
    rs = build(work / "built_student", "--core-kernel", KERNEL, "--core-sha", s_sha, "--tag", "fixture")
    rp = build(work / "built_public")
    for r in (rs, rp):
        if r.returncode:
            raise SystemExit(f"FIXTURE FAILED: a build: {r.stderr[-800:]}")
    pub_nb = json.loads((PUB_DIR / json.loads((PUB_DIR / "kernel-metadata.json").read_text())["code_file"]).read_text())
    pub_srcs = ["".join(c["source"]) for c in pub_nb["cells"]]
    for built, n, core in ((work / "built_student", 7, True), (work / "built_public", 6, False)):
        kmeta = json.loads((built / "kernel-metadata.json").read_text())
        srcs = ["".join(c["source"]) for c in json.loads((built / kmeta["code_file"]).read_text())["cells"]]
        assert len(srcs) == n, (built, len(srcs))
        assert [srcs[1], srcs[-4], srcs[-3]] == pub_srcs[1:4], "public cells 1-3 are not verbatim"
        assert ("Upgraded stack, part 1" in srcs[2]) == core and "The core alone, part 2" in srcs[-2]
        assert "The core alone, part 3" in srcs[-1]
        before = pub_srcs[-2]
        for seg in (before[before.index("_KE_SRC = r'''"):before.index("\n\ndef _ke_run_raptor_arms():")].strip("\n"),
                    before[before.index("_ke_input_lock = "):before.index("rsna_phase('public_raptor', 'START')")].strip("\n")):
            assert seg in srcs[-2], "public cell -2's Raptor code is not verbatim in the core-alone cell"
        joined = "\n".join(srcs)
        for foreign in ("_DINOV2_MATCHED_MEMBERS", "rsna_phase('dinov2'", "_coat_substitute", "Publish submission.csv"):
            assert foreign not in joined, f"{foreign} rode along"
        assert kmeta["dataset_sources"] == ["dreaddevelopment/raptor-knee-maxspan"], kmeta["dataset_sources"]
        assert kmeta["kernel_sources"] == ([KERNEL] if core else []), kmeta["kernel_sources"]
        assert kmeta["id"] == ("raahimnawaz/rsna-knee-sub-alone-fixture" if core
                               else "raahimnawaz/rsna-knee-sub-alone-public"), kmeta["id"]
    print("1. both builds: intro, public cells 1-3 verbatim, (core,) core alone, check; cell -2's Raptor code "
          "verbatim; no DINOv2 / A5 / Rad / publish; sources the core's dataset (+ the student kernel)")

    # ---- 2. the synthetic competition root
    root = work / "comp"
    ids = make_root(root)
    print(f"2. synthetic root: {len(ids)} studies, 19 series of 5 shuffled DICOM slices (study 4 without a 2nd coronal)")

    # ---- 3. the student build, as built
    print("3. the student build, run as built on two CPU devices:")
    ns, _ = run_build(work / "built_student", work, root, "student")
    sub = pd.read_csv(ns["_fixture_kw"] / "submission.csv", dtype={"StudyInstanceUID": str})
    assert sub.columns.tolist() == ["StudyInstanceUID"] + LABELS and sub.StudyInstanceUID.tolist() == ids
    assert np.isfinite(sub[LABELS].to_numpy()).all() and sub[LABELS].to_numpy().min() >= 0 and sub[LABELS].to_numpy().max() <= 1
    loads = [e for e in ns["_RSNA_AUDIT"]["events"] if e["kind"] == "raptor_checkpoint"]
    assert [Path(e["file"]).name for e in loads] == [f"student_core_fixture_{s_sha[:12]}.pt"] * 2, loads
    assert not [e for e in ns["_RSNA_AUDIT"]["events"] if e["kind"] == "raptor_study_failed"]
    events = list(ns["_RSNA_AUDIT"]["events"])    # the run's own log, before step 4 loads the model again

    # ---- 4. index parity: one study at a time, one device, the same public functions
    ke, arm = ns["_KE_NS"], ns["_al_arms"][0]
    ns["_ke_input_ids"].clear()                   # the public preparer refuses a second preparation of a study
    model, _ = ke["load_model"](ke["find_weight_file"](arm["file"]), arm["arch"], arm["res"], torch.device("cpu"))
    reader = ke["_make_reader"]()
    raw = np.zeros((2, len(ids), 12), np.float32)
    for i, uid in enumerate(ids):
        w = ns["_ke_prepare_windows"](arm, uid, ns["_al_series"], ns["_al_series_root"], reader)
        raw[0, i] = ke["infer_probs"](model, w, torch.device("cpu"))
        raw[1, i] = ke["infer_probs"](model, w.flip(1).contiguous(), torch.device("cpu"))
    got = np.load(ns["_fixture_kw"] / "core_alone_raw.npz")["raw_probabilities"]
    ref = ke["rankpct"](np.tensordot(np.asarray([6 / 7, 1 / 7]), raw, axes=(0, 0)))
    d_raw, d_rank = float(np.abs(got - raw).max()), float(np.abs(sub[LABELS].to_numpy() - ref).max())
    print(f"4. index parity, study by study on one device: |raw| {d_raw:.1e}, |ranked 6:1 blend| {d_rank:.1e}")
    if not (d_raw < 1e-5 and d_rank < 1e-6):
        raise SystemExit(f"FIXTURE FAILED: parity raw {d_raw:.2e}, ranks {d_rank:.2e}")

    # ---- 5. the public build, and the control
    print("5. the public build, run as built:")
    pns, _ = run_build(work / "built_public", work, root, "public")
    ploads = [e for e in pns["_RSNA_AUDIT"]["events"] if e["kind"] == "raptor_checkpoint"]
    assert [Path(e["file"]).name for e in ploads] == [PUB_V5.name] * 2 and all(
        e["sha256"] == sha(PUB_V5) for e in ploads), ploads
    pgot = np.load(pns["_fixture_kw"] / "core_alone_raw.npz")["raw_probabilities"]
    d_ctrl = float(np.abs(pgot - got).max())
    print(f"   control |public weights - student| on the raw probabilities: {d_ctrl:.2e}")
    if not d_ctrl > 1e-3:
        raise SystemExit(f"FIXTURE FAILED: the public weights score like the student ({d_ctrl:.2e})")

    # ---- 6. the check's refusals
    check_src = ns["_fixture_cells"]["check"]
    other = {"kind": "raptor_checkpoint", "file": "/kaggle/input/raptor-knee-native384/raptor_ft_coatnet_v8_full_swa.pt",
             "sha256": "0" * 64}
    ours = [e for e in events if e["kind"] == "raptor_checkpoint"]

    def check(evs):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(check_src, dict(ns, _RSNA_AUDIT={"events": evs}))

    check(events)
    refuses(lambda: check(events + [other]), "a load of another Raptor file")
    refuses(lambda: check(ours[:1]), "one load where there are two devices")
    refuses(lambda: check([dict(e, sha256="f" * 64) for e in ours]), "our file name with another sha256")
    refuses(lambda: check(events + [{"kind": "raptor_study_failed", "study": ids[0], "error": "x"}]),
            "a failed study in a commit-sized run")
    print(f"CORE ALONE FIXTURE PASSED ({work})")


if __name__ == "__main__":
    main()
