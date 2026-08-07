"""Kaggle-side DICOM audit. Answers the two questions that need pixels, in one pass.

Run as a Kaggle Script notebook with the competition dataset attached. Nothing here runs
locally -- the images are 570 GB and never leave Kaggle.

  Q1  LATERALITY (PLAN.md 3.2, "the sharpest trap")
      Four of the twelve labels are side-specific, and "medial" is on opposite sides of the
      image for a left vs a right knee. Feed a model mixed handedness and those four labels
      are noise no backbone can recover. The DICOMs are stripped to an 86-tag allowlist, so
      the question is whether (0020,0060) Laterality survived. If it did not, the fallback is
      the sign of the x-coordinate in ImagePositionPatient -- but de-identification often
      zeroes or shifts that, so this script checks whether the geometry is trustworthy before
      recommending it.

  Q2  TRANSFER SYNTAX (PLAN.md 3.1 step 2, 9.4)
      The corpus mixes uncompressed, JPEG Lossless, JPEG 2000, and Implicit VR. JPEG 2000
      decode is slow and drives the whole preprocessing budget. Measures decode cost per
      syntax so the cache build can be planned rather than guessed.

Writes /kaggle/working/dicom_audit.json -- download it and record the findings in FINDINGS.md.

Header reads use stop_before_pixels=True and are fast; only the decode benchmark touches
pixel data, and only for a handful of files per syntax.
"""
import json, time, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.uid import UID

N_STUDIES = 200          # studies to sample for the header audit
N_DECODE_PER_SYNTAX = 6  # files per transfer syntax for the decode benchmark
SEED = 20260807

# Tags that decide whether the vision pipeline is buildable as planned.
CRITICAL = {
    "Laterality":              (0x0020, 0x0060),
    "ImageLaterality":         (0x0020, 0x0062),
    "BodyPartExamined":        (0x0018, 0x0015),
    "ImagePositionPatient":    (0x0020, 0x0032),
    "ImageOrientationPatient": (0x0020, 0x0037),
    "PixelSpacing":            (0x0028, 0x0030),
    "SliceThickness":          (0x0018, 0x0050),
    "InstanceNumber":          (0x0020, 0x0013),
    "SeriesDescription":       (0x0008, 0x103E),
    "Rows":                    (0x0028, 0x0010),
    "Columns":                 (0x0028, 0x0011),
}


def find_input_root() -> Path:
    """Locate the competition mount without hardcoding a layout that may have changed."""
    base = Path("/kaggle/input")
    if not base.exists():
        sys.exit("no /kaggle/input -- this script is meant to run on Kaggle, not locally")
    cands = [p for p in base.iterdir() if p.is_dir() and "knee" in p.name.lower()]
    if not cands:
        cands = [p for p in base.iterdir() if p.is_dir()]
    if not cands:
        sys.exit(f"nothing mounted under {base}")
    root = cands[0]
    print(f"input root: {root}")
    for child in sorted(root.iterdir())[:20]:
        kind = "dir " if child.is_dir() else "file"
        print(f"  {kind} {child.name}")
    return root


def find_dicoms(root: Path, limit_studies: int) -> dict[str, list[Path]]:
    """Discover the image layout by globbing rather than assuming study/series/*.dcm."""
    img_dirs = [d for d in root.iterdir()
                if d.is_dir() and any(k in d.name.lower() for k in ("train", "image", "dcm"))]
    search = img_dirs or [root]
    print(f"\nsearching for DICOMs under: {[str(d.name) for d in search]}")

    by_study: dict[str, list[Path]] = defaultdict(list)
    seen = 0
    for d in search:
        for p in d.rglob("*.dcm"):
            # Layout is <...>/<StudyInstanceUID>/<SeriesInstanceUID>/<file>.dcm in the usual
            # RSNA packaging; fall back to the closest ancestor that looks like a UID.
            parts = [q for q in p.parts if q.count(".") > 3]
            study = parts[0] if parts else p.parent.parent.name
            if study not in by_study:
                seen += 1
                if seen > limit_studies:
                    break
            by_study[study].append(p)
        if seen > limit_studies:
            break

    if not by_study:
        sys.exit("found no .dcm files -- inspect the printed layout and adjust the glob")
    n_files = sum(len(v) for v in by_study.values())
    print(f"found {n_files:,} files across {len(by_study):,} studies (sampled)")
    return dict(by_study)


def audit_headers(by_study: dict[str, list[Path]]) -> dict:
    """One header per study: which tags survived, and what Laterality says if present."""
    present = Counter()
    lat_values, bodypart, syntaxes = Counter(), Counter(), Counter()
    all_tags = Counter()
    ipp_x, ipp_rows = [], 0
    n = 0

    for study, files in by_study.items():
        try:
            ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
        except Exception as e:
            print(f"  unreadable: {files[0].name}: {e}")
            continue
        n += 1
        for elem in ds:
            all_tags[f"{elem.tag} {elem.keyword or '<private>'}"] += 1
        for name, tag in CRITICAL.items():
            if tag in ds:
                present[name] += 1
        if (0x0020, 0x0060) in ds:
            lat_values[str(ds[(0x0020, 0x0060)].value).strip().upper() or "<empty>"] += 1
        if (0x0018, 0x0015) in ds:
            bodypart[str(ds[(0x0018, 0x0015)].value).strip().upper() or "<empty>"] += 1
        ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        syntaxes[f"{UID(ts).name if ts else 'unknown'}"] += 1
        if (0x0020, 0x0032) in ds:
            try:
                ipp_x.append(float(ds[(0x0020, 0x0032)].value[0])); ipp_rows += 1
            except Exception:
                pass

    print(f"\n{'='*70}\nQ1  TAG SURVIVAL -- {n} studies sampled\n{'='*70}")
    for name in CRITICAL:
        c = present[name]
        mark = "OK " if c == n else ("PARTIAL" if c else "GONE")
        print(f"  {mark:<8}{name:<26}{c:>5}/{n}")

    print(f"\nLaterality values: {dict(lat_values) or 'TAG ABSENT'}")
    print(f"BodyPartExamined : {dict(bodypart) or 'TAG ABSENT'}")

    geom = geometry_verdict(ipp_x, ipp_rows, n)

    print(f"\ntotal distinct tags seen: {len(all_tags)}  "
          f"(PLAN.md says the allowlist is 86)")
    return {
        "n_studies_sampled": n,
        "tag_presence": {k: present[k] for k in CRITICAL},
        "laterality_values": dict(lat_values),
        "bodypart_values": dict(bodypart),
        "distinct_tags_seen": len(all_tags),
        "all_tags": dict(all_tags.most_common()),
        "transfer_syntaxes": dict(syntaxes),
        "geometry": geom,
    }


def geometry_verdict(ipp_x: list[float], rows: int, n: int) -> dict:
    """Is the ImagePositionPatient x-sign a usable handedness fallback?

    DICOM patient coordinates put +x at the patient's LEFT. A knee volume centred at negative
    x is therefore the right knee. That only holds if de-identification left the frame intact
    -- many pipelines zero or re-origin it, which shows up as all-zero or single-signed values.
    """
    out = {"n_with_ipp": rows, "usable": False, "reason": ""}
    if rows < max(10, n // 4):
        out["reason"] = "ImagePositionPatient largely absent"
    else:
        x = np.array(ipp_x, float)
        neg, pos, zero = (x < -1).sum(), (x > 1).sum(), (np.abs(x) <= 1).sum()
        out.update(n_negative=int(neg), n_positive=int(pos), n_near_zero=int(zero),
                   x_min=float(x.min()), x_max=float(x.max()), x_median=float(np.median(x)))
        if zero > 0.5 * len(x):
            out["reason"] = "x is ~0 for most studies -- frame was re-origined by de-id"
        elif neg == 0 or pos == 0:
            out["reason"] = ("all studies share one x sign -- either a single-sided cohort "
                             "or a shifted frame; cannot separate L from R")
        else:
            out["usable"] = True
            out["reason"] = f"both signs present ({neg} negative, {pos} positive) -- usable"

    print(f"\n{'-'*70}\ngeometry fallback: {'USABLE' if out['usable'] else 'NOT USABLE'}"
          f" -- {out['reason']}")
    return out


def benchmark_decode(by_study: dict[str, list[Path]]) -> dict:
    """Decode cost per transfer syntax. Drives the preprocessing budget (PLAN.md 9.4)."""
    buckets: dict[str, list[Path]] = defaultdict(list)
    for files in by_study.values():
        for p in files[:3]:
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
                name = UID(ts).name if ts else "unknown"
            except Exception:
                continue
            if len(buckets[name]) < N_DECODE_PER_SYNTAX:
                buckets[name].append(p)

    print(f"\n{'='*70}\nQ2  DECODE COST PER TRANSFER SYNTAX\n{'='*70}")
    print(f"{'syntax':<44}{'n':>4}{'ms/slice':>10}")
    print("-" * 58)
    out = {}
    for name, paths in sorted(buckets.items()):
        times = []
        for p in paths:
            try:
                t0 = time.perf_counter()
                _ = pydicom.dcmread(p, force=True).pixel_array
                times.append((time.perf_counter() - t0) * 1000)
            except Exception as e:
                print(f"  DECODE FAILED {name}: {type(e).__name__}: {e}")
                print("   -> pip install pylibjpeg pylibjpeg-openjpeg pylibjpeg-libjpeg gdcm")
                break
        if times:
            ms = float(np.median(times))
            out[name] = {"n": len(times), "median_ms": ms}
            print(f"{name:<44}{len(times):>4}{ms:>10.1f}")

    if out:
        worst = max(out.items(), key=lambda kv: kv[1]["median_ms"])
        est = worst[1]["median_ms"] * 819_640 / 1000 / 3600
        print(f"\nslowest syntax: {worst[0]} at {worst[1]['median_ms']:.1f} ms/slice")
        print(f"single-process upper bound over all 819,640 files: {est:.1f} h "
              f"-- multiprocess the cache build accordingly")
    return out


def main() -> None:
    root = find_input_root()
    by_study = find_dicoms(root, N_STUDIES)
    result = audit_headers(by_study)
    result["decode_benchmark"] = benchmark_decode(by_study)

    out = Path("/kaggle/working/dicom_audit.json")
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {out}")

    print(f"\n{'='*70}\nVERDICT\n{'='*70}")
    n = result["n_studies_sampled"]
    lat = result["tag_presence"]["Laterality"]
    ilat = result["tag_presence"]["ImageLaterality"]
    if lat == n or ilat == n:
        tag = "Laterality" if lat == n else "ImageLaterality"
        print(f"  {tag} SURVIVED on every sampled study. Canonicalise handedness from it,")
        print("  then drop the pixel-classifier fallback from PLAN.md 3.2.")
    elif lat or ilat:
        print(f"  Laterality is PARTIAL ({lat}/{n}). Use it where present and fall back")
        print("  elsewhere -- do not assume a default side.")
    elif result["geometry"]["usable"]:
        print("  Laterality is GONE, but ImagePositionPatient x-sign works: negative x is the")
        print("  right knee. Derive handedness from geometry and audit ~50 studies visually.")
    else:
        print("  Laterality is GONE and geometry is unusable. PLAN.md 3.2's last resort is")
        print("  now the only option: train a small left/right classifier on pixels.")
        print("  Until that exists, the four side-specific labels are unreliable.")
    print("\n  Reminder: hflip TTA stays invalid unless Medial<->Lateral outputs are swapped.")


if __name__ == "__main__":
    main()
