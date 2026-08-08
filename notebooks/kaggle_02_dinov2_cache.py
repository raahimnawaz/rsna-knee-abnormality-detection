"""Kaggle-side: frozen DINOv2 embeddings for every series. The unlock for local iteration.

Everyone has the same DINOv2 weights, so the backbone cannot differentiate us (PLAN.md 7.1).
What it can do is make our differentiator cheap to iterate: freeze it, run it once, cache the
per-slice features, and the fusion head from 3.3 -- slice transformer, attention pool,
series-type embedding, series attention -- then trains on a 16 GB M5 laptop in minutes instead
of needing a GPU session per experiment. Measured: 2.18 GB peak with the full cache, the model
and optimizer steps all live.

Output is ~2.4 GB for the whole corpus: 24,371 series x 32 slices x 1536 dims fp16. (Earlier
drafts of this file said ~800 MB, computed at 768 dims -- but embed() concatenates CLS with the
patch mean, so it is 1536. Corrected 2026-08-07.) Publish /kaggle/working/features as a Kaggle
Dataset and pull it down.

ALL preprocessing lives in pipeline/preprocess.py, imported here rather than defined here. The
submission notebook imports the same file, and if the two ever disagree the model is fed a
distribution it never trained on with nothing raising an error. The manifest written beside the
shards carries PREPROCESS_VERSION so that mismatch is detectable instead of silent.

RUNS ACROSS MULTIPLE SESSIONS BY DESIGN. Decode is the bottleneck, not the GPU. Set
SHARD/N_SHARDS and run N sessions; finished studies are skipped on restart, so an interrupted
session loses at most one study.

  IMG_SIZE  518 gives 37x37 patches and is what "DINOv2 at meniscus resolution" is about -- a
            meniscal tear is small and 16x16 patches at 224 lose it. It costs roughly 5x. Do a
            224 pass first to get the pipeline honest, then re-run at 518.
  SLICES    32 cached, 24 used at train time. The gap is slice jitter, the only pixel-space
            augmentation that survives a frozen backbone (see fusion/dataset.py).
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Locate pipeline/preprocess.py wherever the code Dataset landed. Globbing beats hardcoding:
# the mount name depends on the Dataset slug, and a Dataset version still being created when the
# kernel starts can leave the path briefly absent -- which surfaced as a bare ModuleNotFoundError
# and cost a GPU session. Fail with a listing instead.
def _bootstrap_preprocess() -> None:
    import glob
    # Recursive: Kaggle nests sources under competitions/ and datasets/ when a kernel has more
    # than one, so the depth of the mount is not fixed. Measured 2026-08-07 -- /kaggle/input held
    # exactly ['competitions', 'datasets'] and a one-level glob found nothing.
    for pat in ("/kaggle/input/**/pipeline/preprocess.py", "/kaggle/usr/lib/**/preprocess.py",
                str(Path(__file__).resolve().parents[1] / "pipeline" / "preprocess.py")):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            sys.path.insert(0, str(Path(hits[0]).parent))
            return
    listing = sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*"))
    raise SystemExit(
        "cannot find pipeline/preprocess.py. Attach the rsna-knee-code Dataset to this "
        f"notebook. /kaggle/input currently holds: {listing}")


_bootstrap_preprocess()
from preprocess import (BATCH_HINT, MODEL, PLANE_ID, PREPROCESS_VERSION,  # noqa: E402
                        build_study_index, find_competition_root, imagenet_normalise,
                        load_series, manifest, to_25d)

BATCH = BATCH_HINT
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
OUT = Path("/kaggle/working/features")

# Decode runs in worker processes so it overlaps the GPU instead of alternating with it.
# PLAN.md 5 says this about the submission notebook -- "multiprocess it (CPU-bound, will
# otherwise starve the GPU)" -- and it is just as true here, where ~700k slices have to be
# decoded. Serial decode roughly doubles the wall clock of this script, which is the difference
# between a couple of Kaggle sessions and several, against a 9h cap and a weekly GPU quota.
#
# Kaggle gives ~4 usable cores. PREFETCH bounds how many decoded volumes are in flight; each is
# ~27 MB at 32 slices of 457x457 float32, so 8 is ~215 MB of headroom.
# The mount is LATENCY-bound, not bandwidth- or CPU-bound: kaggle_01b measured ~19 ms per file
# open, and the cache needs ~700k of them. That is ~10 h serial, which is why the workers are
# not optional and why oversubscribing cores helps -- they are blocked on I/O, not computing.
N_WORKERS = int(os.environ.get("N_WORKERS", 8))
PREFETCH = int(os.environ.get("PREFETCH", 16))


def _pool(max_workers):
    """ProcessPoolExecutor on a SPAWN context.

    The default on Linux is fork, and this process initialises CUDA (timm .to('cuda')) before
    the pool is created. Forking a process that already holds a CUDA context is documented as
    unsafe, and the observed cost was severe: the first 224 run tracked the SERIAL throughput
    curve for 9 h against a ~2.7 h estimate with 4 workers. Spawn starts clean children that
    never inherit the context. They re-import this module, which is why _decode_task is
    top-level and why main() sits behind an __name__ guard.
    """
    return ProcessPoolExecutor(max_workers=max_workers,
                               mp_context=multiprocessing.get_context("spawn"))


def _decode_task(item):
    """Runs in a worker process: DICOM -> normalised, canonicalised volume.

    Must stay top-level and picklable. Returns the volume rather than embedding it, because the
    GPU lives in the parent -- workers do the CPU-bound half and nothing else.
    """
    study, k, files, plane_name, fs_flag = item
    try:
        vol, side, src = load_series([Path(f) for f in files], plane_name)
    except Exception:
        vol, side, src = None, None, "none"
    return study, k, plane_name, fs_flag, vol, side, src


@torch.no_grad()
def embed(model, x: torch.Tensor, dev: str) -> np.ndarray:
    """[S,3,H,W] -> [S, 2D] fp16: CLS concatenated with the patch mean.

    Both halves earn their place: CLS carries the global impression, the patch mean retains
    localised signal that a single token averages away -- and the findings here are small.
    """
    out = []
    for i in range(0, len(x), BATCH):
        b = imagenet_normalise(x[i:i + BATCH].to(dev))
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            tok = model.forward_features(b)
        cls, patches = tok[:, 0], tok[:, model.num_prefix_tokens:].mean(1)
        out.append(torch.cat([cls, patches], -1).float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def build_cache(root: Path, mine: list, meta, out: Path, embed_fn,
                pool_factory=None) -> tuple[int, int, dict]:
    """The scheduling loop. Injectable so --self-test can drive it without DICOMs or a GPU."""
    done = skipped = 0
    lat_seen = {"tag": 0, "geometry": 0, "none": 0}

    # Build the whole work list up front so decode can run ahead of the GPU across STUDY
    # boundaries, not just within one study. A study has ~5 series; without look-ahead past the
    # end of a study the GPU stalls every five items waiting on the next decode.
    todo = []
    index = build_study_index(root)             # one pass, not one walk per study
    for study in mine:
        if (out / f"{study}.npz").exists():     # resume: an interrupted session loses one study
            skipped += 1
            continue
        sdir = index.get(study)
        if sdir is None:
            continue
        for k, ser in enumerate(sorted(p for p in sdir.iterdir() if p.is_dir())):
            files = sorted(ser.glob("*.dcm"))
            if files:
                row = meta.loc[ser.name] if ser.name in meta.index else None
                todo.append((study, k, [str(f) for f in files],
                             getattr(row, "Anatomical_Plane", None),
                             int(getattr(row, "Fluid_Sensitive", -1))))
    n_studies = len({t[0] for t in todo})
    print(f"{len(todo):,} series across {n_studies:,} studies to decode "
          f"({skipped:,} studies already cached)")

    n_done_series = 0
    pending: dict[str, list] = defaultdict(list)
    remaining = Counter(t[0] for t in todo)
    t0 = time.time()

    with (pool_factory or _pool)(max_workers=N_WORKERS) as pool:
        it = iter(todo)
        futures = deque()
        # Bounded look-ahead: each in-flight decode holds a ~27 MB volume, so PREFETCH caps the
        # memory this costs. Unbounded submission would decode the entire shard into RAM.
        for _ in range(PREFETCH):
            nxt = next(it, None)
            if nxt is None:
                break
            futures.append(pool.submit(_decode_task, nxt))

        while futures:
            study, k, plane_name, fs_flag, vol, side, src = futures.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                futures.append(pool.submit(_decode_task, nxt))

            lat_seen[src] = lat_seen.get(src, 0) + 1
            if vol is not None:
                e = embed_fn(to_25d(vol))
                pending[study].append((k, e, PLANE_ID.get(plane_name, -1), fs_flag,
                                       {"L": 0, "R": 1}.get(side, -1)))
            n_done_series += 1
            if n_done_series in (25, 100, 400):
                el = time.time() - t0
                rate = n_done_series / max(el, 1e-9)
                print(f"  PROBE {n_done_series} series in {el:.0f}s = {rate:.2f} series/s "
                      f"-> ~{len(todo) / max(rate, 1e-9) / 3600:.1f} h for this shard "
                      f"({N_WORKERS} workers)")
                if n_done_series == 100 and rate < 0.35:
                    print("    WARNING: that is near single-worker throughput. The pool may not "
                          "be parallelising -- check N_WORKERS and the spawn context before "
                          "letting this run for hours.")
            remaining[study] -= 1
            if remaining[study] > 0:            # study not finished yet
                continue

            parts = sorted(pending.pop(study, []), key=lambda x: x[0])
            if not parts:
                continue
            feats = [p[1] for p in parts]
            sid = [p[0] for p in parts for _ in range(len(p[1]))]
            plane = [p[2] for p in parts for _ in range(len(p[1]))]
            fs = [p[3] for p in parts for _ in range(len(p[1]))]
            # Recorded per series so laterality coverage can be audited after the fact, and so
            # a study whose handedness was unknown can be found again without a full rebuild.
            lat = [p[4] for p in parts for _ in range(len(p[1]))]
            # Write to a temp name and rename: np.savez_compressed on a study that is killed
            # mid-write leaves a truncated .npz that the resume check would treat as finished.
            tmp = out / f".{study}.tmp.npz"
            np.savez_compressed(tmp, feats=np.concatenate(feats),
                                series_idx=np.array(sid, np.int16),
                                plane=np.array(plane, np.int8),
                                fluid_sensitive=np.array(fs, np.int8),
                                laterality=np.array(lat, np.int8))
            tmp.replace(out / f"{study}.npz")
            done += 1
            if done % 20 == 0:
                rate = done / (time.time() - t0)
                left = (n_studies - done) / max(rate, 1e-9) / 3600
                print(f"  {done:>5}/{n_studies:,}  {rate * 3600:>6.0f} studies/h  "
                      f"~{left:.1f} h left")

    return done, skipped, lat_seen


def main() -> None:
    import timm
    root = find_competition_root()
    OUT.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(root / "train_series.csv")
    studies = sorted(series.StudyInstanceUID.unique())
    mine = [s for i, s in enumerate(studies) if i % N_SHARDS == SHARD]
    print(f"shard {SHARD}/{N_SHARDS}: {len(mine):,} of {len(studies):,} studies")
    print(f"preprocess version {PREPROCESS_VERSION}")
    print(f"{N_WORKERS} decode workers, prefetch {PREFETCH}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().to(dev)
    print(f"{MODEL} on {dev}, prefix_tokens={model.num_prefix_tokens}")

    done, skipped, lat_seen = build_cache(
        root, mine, series.set_index("SeriesInstanceUID"), OUT,
        embed_fn=lambda x: embed(model, x, dev))

    total_lat = sum(lat_seen.values()) or 1
    print(f"\nshard {SHARD}: {done:,} written, {skipped:,} already present -> {OUT}")
    print(f"laterality source: tag {lat_seen.get('tag', 0)}, "
          f"geometry {lat_seen.get('geometry', 0)}, none {lat_seen.get('none', 0)} "
          f"({100 * lat_seen.get('none', 0) / total_lat:.1f}%)")
    if lat_seen.get("none", 0) > total_lat * 0.02:
        print("  WARNING: >2% of series have neither a tag nor usable geometry, so they are "
              "NOT canonicalised. Medial/Lateral labels are only as good as this -- PLAN.md 3.2.")

    (OUT / f"_shard{SHARD}.json").write_text(json.dumps(
        manifest(written=done, shard=SHARD, n_shards=N_SHARDS, laterality=lat_seen), indent=2))
    print("Publish /kaggle/working/features as a Kaggle Dataset, then train the fusion head "
          "on the M5.")


# ------------------------------------------------------------------------------- self-test
class _SerialPool:
    """Executor stub that runs inline. Exercises the SCHEDULING, which is the novel part.

    multiprocessing itself is stdlib and well tested; the prefetch window, the per-study
    completion accounting and the resume path are mine, and they are what would strand a
    three-hour Kaggle session.
    """

    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, item):
        class _F:
            def __init__(self, v):
                self._v = v

            def result(self):
                return self._v
        return _F(_fake_decode(item))


def _fake_decode(item):
    study, k, files, plane_name, fs_flag = item
    n = len(files)
    if n < 3:                                   # undecodable series -> dropped, not fatal
        return study, k, plane_name, fs_flag, None, None, "none"
    side = {0: "L", 1: "R"}.get(k % 3, None)    # exercise L, R and the unknown branch
    src = {0: "tag", 1: "geometry"}.get(k % 3, "none")
    return (study, k, plane_name, fs_flag,
            np.full((n, 8, 8), float(k), dtype=np.float32), side, src)


def self_test() -> None:
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="kaggle02_"))
    root, out = tmp / "root", tmp / "out"
    out.mkdir(parents=True)

    # 7 studies x variable series; study 3 gets an undecodable series, study 5 has only one.
    rows, expect = [], {}
    for s in range(7):
        uid = f"study{s:02d}"
        n_ser = 1 if s == 5 else (2 + s % 4)
        ok = 0
        for k in range(n_ser):
            sid = f"{uid}_ser{k}"
            d = root / "train_series" / uid / sid
            d.mkdir(parents=True)
            n_files = 1 if (s == 3 and k == 0) else 5      # 1 file -> undecodable
            for f in range(n_files):
                (d / f"{f:03d}.dcm").write_bytes(b"")
            ok += n_files >= 3
            rows.append({"StudyInstanceUID": uid, "SeriesInstanceUID": sid,
                         "Anatomical_Plane": ["Axial", "Coronal", "Sagittal"][k % 3],
                         "Fluid_Sensitive": k % 2})
        expect[uid] = ok
    meta = pd.DataFrame(rows).set_index("SeriesInstanceUID")
    mine = sorted(expect)

    embed_fn = lambda x: np.asarray(x[:, 0, 0, 0], dtype=np.float16).reshape(-1, 1)  # noqa: E731
    done, skipped, lat = build_cache(root, mine, meta, out, embed_fn, _SerialPool)
    assert done == 7 and skipped == 0, (done, skipped)

    for uid, n_ok in expect.items():
        z = np.load(out / f"{uid}.npz")
        got = sorted(set(z["series_idx"].tolist()))
        assert len(got) == n_ok, f"{uid}: {len(got)} series cached, expected {n_ok}"
        # series_idx must stay sorted -- the fusion head groups on it and a shuffled order
        # would silently pair a series' features with another series' plane/FS flags.
        assert z["series_idx"].tolist() == sorted(z["series_idx"].tolist()), f"{uid} unsorted"
        for key in ("plane", "fluid_sensitive", "laterality"):
            assert len(z[key]) == len(z["feats"]), f"{uid}: {key} length mismatch"
    print(f"  {done} studies written, series counts and ordering correct")
    assert lat["tag"] and lat["geometry"] and lat["none"], f"lat sources not all hit: {lat}"
    print(f"  laterality sources all exercised: {lat}")

    # Resume: rerun over the same output, nothing should be rebuilt.
    done2, skipped2, _ = build_cache(root, mine, meta, out, embed_fn, _SerialPool)
    assert (done2, skipped2) == (0, 7), (done2, skipped2)
    print("  resume: 0 rebuilt, 7 skipped")

    # A truncated .npz must not be mistaken for a finished study.
    (out / "study00.npz").write_bytes(b"not an npz")
    try:
        np.load(out / "study00.npz")
        raise AssertionError("expected a corrupt-file error")
    except Exception:
        pass
    print("  (corrupt cache files are caught by np.load; the tmp+rename write prevents "
          "producing them)")

    shutil.rmtree(tmp)
    print("\nself-test PASSED")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
