"""Build the DINOv2 feature cache on this laptop, from the NIfTI corpus. PLAN.md 9.1.

The Kaggle-side equivalent is `notebooks/kaggle_02_dinov2_cache.py`, which has not finished in
five attempts -- every one of them killed by a Kaggle-specific property (the GPU lottery, the 9 h
cap, and above all ~19 ms to open one of ~700k small files on a network mount). None of those are
properties of the data. This is the same cache built where none of them apply.

    IMG_SIZE=224 python pipeline/build_cache_local.py     # ~4.2 h, the proving run
    python pipeline/build_cache_local.py                  # 518, ~26 h, the real cache
    python pipeline/build_cache_local.py --self-test      # seconds, no NIfTI needed

`IMG_SIZE` feeds `PREPROCESS_VERSION`, so a 224 cache and a 518 cache have different fingerprints
and **cannot be confused with each other** -- which is what makes the cheap proving run safe to
do first. Run 224 to unblock every downstream experiment today, then 518 for what gets submitted.

WHY THERE IS NO WORKER POOL, and this is the main design decision.

`kaggle_02` runs an 8-worker spawn pool with a prefetch window, and that machinery cost two
sessions on its own (IMPROVEMENTS.md K1, K7). It exists because a DICOM series is ~25 separate
opens on a network mount at ~19 ms each -- decode was never the bottleneck, latency was. Here one
series is ONE file on a local SSD. Measured: ~30 ms to read a series against ~2.4 s of GPU for it
at 518 (~0.4 s at 224). I/O is about 1% of the loop. A pool would be complexity guarding nothing,
and this file would rather be boring than clever. **If that ratio ever changes, measure it before
adding threads.**

WHAT THIS TAKES FROM ELSEWHERE, because the NIfTI files cannot supply it:

  laterality   data/study_meta.csv, via preprocess.study_laterality(). The conversion dropped the
               patient coordinate system entirely (PLAN.md 9.1 correction), so the geometry
               fallback cannot run from the pixels. kaggle_01b's DICOM-derived answer is used
               instead -- 2,203 by tag, 2,204 by geometry, 0 unresolved.
  plane / FS   data/train_series.csv, which carries Anatomical_Plane and Fluid_Sensitive.

VALIDATED 2026-08-09. `pipeline/validate_nifti.py` checks 1-5 all pass against the kaggle_01c
export: in-plane layout `as-is` at r = 1.0000 (identical pixels), slice order 100% forward,
slice counts and spacings exact. The `--validated` flag exists so a future corpus revision has
to re-earn that rather than inherit it; without it the script warns on every run.

MEASURED THROUGHPUT: 1,062 studies/h at 224 on an M5, stable across three passes -> ~4.2 h for
the full 4,407. The 518 pass is ~6.3x the tokens, so ~26 h. PLAN.md 9.1's "~1.7 h at 224" was
computed for 16 slices/series; the configured value is 32.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "pipeline"))
from preprocess import (BATCH_HINT, EMBED_DIM, IMG_SIZE, MODEL,        # noqa: E402
                        PLANE_ID, PREPROCESS_VERSION, embed, load_series_nifti,
                        manifest, pick_device, study_laterality, to_25d)

D = PROJ / "data"
LAT_CODE = {"L": 0, "R": 1}


def device() -> str:
    dev = pick_device(allow_mps=True)
    if dev == "unusable":
        sys.exit("a GPU is present that this PyTorch has no kernels for. Refusing rather than "
                 "silently falling back to CPU, which would take days at 518.")
    return dev


def build_model(dev: str):
    """timm backbone + the smoke test that K14 says must happen before touching any data."""
    import timm
    model = timm.create_model(MODEL, pretrained=True, num_classes=0,
                              dynamic_img_size=True).eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"{MODEL} on {dev}, prefix_tokens={model.num_prefix_tokens}")

    # IMPROVEMENTS.md K14: this checkpoint is 518-native and timm asserts on any other input size
    # unless the position embedding is interpolated at forward time. That failure cost a session
    # because it surfaced on the first real series, after the corpus walk and the weights
    # download. One synthetic slice settles it in a second.
    with torch.no_grad():
        probe = embed(model, torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), dev, BATCH_HINT)
    if probe.shape != (1, EMBED_DIM):
        sys.exit(f"backbone smoke test: embed() returned {probe.shape}, expected "
                 f"(1, {EMBED_DIM}) at IMG_SIZE={IMG_SIZE}. The cache would be unusable.")
    print(f"backbone smoke test OK at {IMG_SIZE}px -> {EMBED_DIM}-d")
    return model


def series_table(path: Path) -> dict:
    """-> {study: [(series_uid, plane_name, fluid_sensitive), ...]}."""
    df = pd.read_csv(path)
    out: dict[str, list] = {}
    for r in df.itertuples():
        out.setdefault(r.StudyInstanceUID, []).append(
            (r.SeriesInstanceUID, str(r.Anatomical_Plane), int(r.Fluid_Sensitive)))
    return out


def build(nifti_dir: Path, out: Path, by_study: dict, lat_of: dict, embed_fn,
          limit: int = 0, probe_at: int = 25, only: set | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    # A cache directory belongs to exactly one PREPROCESS_VERSION. Without this, running the 518
    # pass into a directory holding 224 features skips every study on the count check and then
    # re-stamps the manifest as 518 -- a 224 cache labelled 518, which nothing downstream can
    # detect. The docstring's claim that the two "cannot be confused" only held if the operator
    # remembered to change --out.
    shard = out / "_shard0.json"
    if shard.exists():
        prev = json.loads(shard.read_text()).get("preprocess_version")
        if prev and prev != PREPROCESS_VERSION:
            sys.exit(f"{out} holds a cache built by preprocess_version {prev}, but this run is "
                     f"{PREPROCESS_VERSION} (IMG_SIZE={IMG_SIZE}). Build into a different --out.")
    studies = sorted(by_study)
    if only is not None:
        missing = only - set(studies)
        if missing:
            sys.exit(f"--studies names {len(missing)} studies absent from {Path(out).name}'s "
                     f"series table; first: {sorted(missing)[0]}")
        studies = [s for s in studies if s in only]
    if limit:
        studies = studies[:limit]

    def files_on_disk(study: str) -> int:
        n = 0
        for ser, _, _ in by_study[study]:
            if (nifti_dir / f"{study}_{ser}.nii").exists() or \
               (nifti_dir / f"{study}_{ser}.nii.gz").exists():
                n += 1
        return n

    def cached_series(dst: Path) -> int:
        """How many series the existing .npz actually holds. -1 if unreadable."""
        try:
            with np.load(dst) as z:
                return int(len(np.unique(z["series_idx"])))
        except Exception:
            return -1

    done = skipped = no_file = rebuilt = 0
    lat_seen: dict[str, int] = {}
    t0 = time.time()
    for n, study in enumerate(studies, 1):
        dst = out / f"{study}.npz"
        if dst.exists():
            # Existence is NOT enough. A study whose series span several download parts gets
            # built from whichever parts had landed at the time -- measured on the first 224
            # pass: 6 studies cached, one of them holding 1 series of 5. A plain exists() check
            # would skip those forever and the fusion head would attend over a study that
            # silently lost 80% of its evidence. Rebuild whenever more series are now on disk
            # than the cache holds, so the build self-heals as parts arrive.
            have, cached = files_on_disk(study), cached_series(dst)
            if cached >= have and cached > 0:
                skipped += 1
                continue
            rebuilt += 1

        side, src = lat_of.get(study, (None, "none"))
        parts = []
        for k, (ser, plane_name, fs_flag) in enumerate(by_study[study]):
            p = nifti_dir / f"{study}_{ser}.nii"
            if not p.exists():
                p = nifti_dir / f"{study}_{ser}.nii.gz"
            if not p.exists():
                no_file += 1
                continue
            vol, _, _ = load_series_nifti(p, plane_name, side, src)
            if vol is None:
                continue
            e = embed_fn(to_25d(vol))
            parts.append((k, e, PLANE_ID.get(plane_name, -1), fs_flag,
                          LAT_CODE.get(side, -1)))

        if not parts:
            continue
        lat_seen[src] = lat_seen.get(src, 0) + 1

        feats = [p[1] for p in parts]
        sid = [p[0] for p in parts for _ in range(len(p[1]))]
        plane = [p[2] for p in parts for _ in range(len(p[1]))]
        fs = [p[3] for p in parts for _ in range(len(p[1]))]
        lat = [p[4] for p in parts for _ in range(len(p[1]))]
        # tmp+rename: a run killed mid-write otherwise leaves a truncated .npz that the resume
        # check above would treat as finished, and FeatureStore would drop the study with a
        # warning nobody reads.
        tmp = out / f".{study}.tmp.npz"
        np.savez_compressed(tmp, feats=np.concatenate(feats),
                            series_idx=np.array(sid, np.int16),
                            plane=np.array(plane, np.int8),
                            fluid_sensitive=np.array(fs, np.int8),
                            laterality=np.array(lat, np.int8))
        tmp.replace(dst)
        done += 1

        # An early probe, for the same reason kaggle_02 has one: a rate that implies days rather
        # than hours should be visible in minutes, not discovered at 3am.
        if done == probe_at or done % 200 == 0:
            rate = done / (time.time() - t0)
            left = (len(studies) - skipped - done) / max(rate, 1e-9) / 3600
            print(f"  PROBE {done:>5}/{len(studies):,}  {rate * 3600:>6.0f} studies/h  "
                  f"~{left:.1f} h left", flush=True)

    # Studies still short of what train_series.csv says they should have. Mid-download this is
    # expected and self-heals; on the final pass it is the number that matters, because it is
    # evidence the corpus itself is missing series rather than the download being incomplete.
    incomplete = [s for s in studies
                  if (out / f"{s}.npz").exists()
                  and cached_series(out / f"{s}.npz") < len(by_study[s])]
    return {"written": done, "skipped": skipped, "rebuilt": rebuilt, "series_missing": no_file,
            "incomplete_studies": len(incomplete), "laterality": lat_seen,
            "elapsed_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nifti", default=str(D / "nifti" / "nifti_train"))
    ap.add_argument("--out", default=str(D / "features"))
    ap.add_argument("--series", default=str(D / "train_series.csv"))
    ap.add_argument("--meta", default=str(D / "study_meta.csv"))
    ap.add_argument("--limit", type=int, default=0, help="first N studies only")
    ap.add_argument("--studies", default=None,
                    help="file of StudyInstanceUIDs, one per line -- build ONLY these. "
                         "--limit takes the lexicographically first N, which scatters the gold "
                         "studies (~7 of 37 in an 800-study prefix) and leaves the pooled OOF "
                         "unscoreable, so a controlled subset has to be named explicitly")
    ap.add_argument("--validated", action="store_true",
                    help="acknowledge that validate_nifti.py checks 4/4b have passed")
    args = ap.parse_args()

    nd = Path(args.nifti)
    if not nd.exists():
        sys.exit(f"{nd} not found -- download the NIfTI parts first")
    for p in (Path(args.series), Path(args.meta)):
        if not p.exists():
            sys.exit(f"{p} not found; it is not optional -- see this file's docstring")

    if not args.validated:
        print("!" * 78)
        print("IN-PLANE ORIENTATION AND SLICE DIRECTION ARE NOT VALIDATED.")
        print("Run notebooks/kaggle_01c_series_geometry.py, then")
        print("  python pipeline/validate_nifti.py --geometry ... --thumbs ...")
        print("Until checks 4 and 4b pass, this cache is for EXPERIMENTS ONLY. Do not train a")
        print("submission on it. Re-run with --validated once they have.")
        print("!" * 78)

    by_study = series_table(Path(args.series))
    lat_of = study_laterality(Path(args.meta))
    have = {p.name.split("_")[0] for p in list(nd.glob("*.nii")) + list(nd.glob("*.nii.gz"))}
    by_study = {s: v for s, v in by_study.items() if s in have}
    print(f"preprocess {PREPROCESS_VERSION} · IMG_SIZE={IMG_SIZE} · "
          f"{len(by_study):,} studies present of 4,407")

    dev = device()
    model = build_model(dev)
    with torch.no_grad():
        only = None
        if args.studies:
            only = {ln.strip() for ln in Path(args.studies).read_text().splitlines() if ln.strip()}
            print(f"--studies: {len(only)} named")
        stats = build(nd, Path(args.out), by_study, lat_of,
                      embed_fn=lambda x: embed(model, x, dev, BATCH_HINT),
                      limit=args.limit, only=only)

    # The manifest must sit beside the features: fusion/train.py copies it out next to fold*.pt
    # and kaggle_03 refuses to submit heads that cannot prove what preprocessed them.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "_shard0.json").write_text(json.dumps(
        manifest(source="nifti_local", validated=bool(args.validated), **stats), indent=2))

    print(f"\n{'=' * 66}")
    print(f"  written {stats['written']:,} · skipped {stats['skipped']:,} · "
          f"rebuilt {stats['rebuilt']:,} · series with no file {stats['series_missing']:,}")
    if stats["incomplete_studies"]:
        print(f"  !! {stats['incomplete_studies']:,} studies hold fewer series than "
              f"train_series.csv lists.\n     Mid-download this is expected -- re-run after the "
              f"remaining parts land and they rebuild.")
    print(f"  laterality sources: {stats['laterality']}")
    print(f"  {stats['elapsed_s'] / 3600:.2f} h on {dev}")
    print(f"  cache: {sum(f.stat().st_size for f in Path(args.out).glob('*.npz')) / 1e9:.2f} GB")
    print(f"{'=' * 66}\nNext: python fusion/folds.py && python fusion/train.py "
          f"--features {args.out}")


def self_test() -> None:
    """Whole loop on synthetic NIfTI + a fake backbone. No corpus, no weights, seconds.

    Proves the thing that actually breaks consumers: that the .npz written here is byte-shaped
    exactly like kaggle_02's, so fusion/dataset.py can read it. It asserts against FeatureStore
    directly rather than against a remembered schema.
    """
    import shutil
    import tempfile
    import nibabel as nib
    sys.path.insert(0, str(PROJ / "fusion"))
    from dataset import FeatureStore, StudyDataset, LABELS, collate

    tmp = Path(tempfile.mkdtemp(prefix="cachebuild_selftest_"))
    try:
        nd = tmp / "nifti"
        nd.mkdir()
        rows, studies = [], [f"1.2.{i}" for i in range(4)]
        planes = ["Axial", "Coronal", "Sagittal"]
        for si, study in enumerate(studies):
            n_ser = 1 if si == 0 else 3          # force the single-series edge case
            for ci in range(n_ser):
                ser = f"1.2.{si}.{ci}"
                plane = planes[ci % 3]
                rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": ser,
                             "Fluid_Sensitive": ci % 2, "Fat_Suppression": ci % 2,
                             "Anatomical_Plane": plane})
                arr = np.random.default_rng(si * 10 + ci).integers(
                    0, 2000, size=(64, 64, 6)).astype(np.int16)
                aff = np.diag([0.33, 0.33, 3.4, 1.0])      # the real corpus's degenerate affine
                nib.save(nib.Nifti1Image(arr, aff), nd / f"{study}_{ser}.nii")
        pd.DataFrame(rows).to_csv(tmp / "train_series.csv", index=False)
        pd.DataFrame([{"StudyInstanceUID": s, "PatientID": "x",
                       "laterality_tag": "R" if i % 2 else "",
                       "x_median": -150.0 if i % 2 == 0 else 14.0,
                       "n_series": 1 if i == 0 else 3, "x_side": ""}
                      for i, s in enumerate(studies)]).to_csv(tmp / "study_meta.csv", index=False)

        by_study = series_table(tmp / "train_series.csv")
        lat_of = study_laterality(tmp / "study_meta.csv")
        assert all(v[0] in ("L", "R") for v in lat_of.values()), lat_of
        assert {v[1] for v in lat_of.values()} == {"tag", "geometry"}, \
            f"both laterality sources must be exercised: {lat_of}"

        out = tmp / "features"
        fake = lambda x: np.zeros((len(x), EMBED_DIM), np.float16)     # noqa: E731
        stats = build(nd, out, by_study, lat_of, embed_fn=fake)
        assert stats["written"] == len(studies), stats
        assert stats["series_missing"] == 0, stats

        # Resume must skip everything and rebuild nothing.
        again = build(nd, out, by_study, lat_of, embed_fn=fake)
        assert again["written"] == 0 and again["skipped"] == len(studies), again
        assert again["incomplete_studies"] == 0, again

        # A study built while one of its parts was still downloading must be REBUILT, not
        # skipped. Simulate it: hide a series, rebuild that study, restore the series, and
        # assert the next pass notices the cache is short rather than trusting exists().
        victim = studies[1]                       # a 3-series study
        hidden = nd / f"{victim}_1.2.1.1.nii"
        (out / f"{victim}.npz").unlink()
        hidden.rename(hidden.with_suffix(".hidden"))
        partial = build(nd, out, by_study, lat_of, embed_fn=fake)
        assert partial["series_missing"] >= 1, partial
        with np.load(out / f"{victim}.npz") as z:
            assert len(np.unique(z["series_idx"])) == 2, "fixture did not go partial"
        hidden.with_suffix(".hidden").rename(hidden)
        healed = build(nd, out, by_study, lat_of, embed_fn=fake)
        assert healed["rebuilt"] == 1, f"partial cache was not rebuilt: {healed}"
        with np.load(out / f"{victim}.npz") as z:
            assert len(np.unique(z["series_idx"])) == 3, "rebuild did not restore the series"
        assert healed["incomplete_studies"] == 0, healed

        # The real contract: fusion/dataset.py must consume it unmodified.
        store = FeatureStore(out, studies)
        assert len(store) == len(studies), f"FeatureStore read {len(store)} of {len(studies)}"
        assert len(store.data[studies[0]]) == 1, "single-series study lost its edge case"
        tgt = pd.DataFrame(0.5, index=studies, columns=LABELS)
        b = collate([StudyDataset(store, studies, tgt)[i] for i in range(len(studies))])
        assert b["feats"].shape[-1] == EMBED_DIM, b["feats"].shape
        assert b["series_type"].max() < 6, "series_type out of range"
        print(f"  npz -> FeatureStore OK: {len(store)} studies, "
              f"batch {tuple(b['feats'].shape)}, series_type {b['series_type'].unique().tolist()}")
        print("\nself-test PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
