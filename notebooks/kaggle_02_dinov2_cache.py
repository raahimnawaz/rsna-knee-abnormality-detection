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
PREPROCESS_DIR_ENV = "RSNA_PREPROCESS_DIR"
SEARCHED_FOR_PREPROCESS = False     # False = took the cached path; see _pool_self_test


def _repo_preprocess() -> str | None:
    """This repo's own copy, for a checkout rather than a Kaggle mount.

    Lazy and guarded on purpose. __main__ has no __file__ in a notebook cell or under exec(),
    and evaluating this eagerly while building the pattern list turned "no code Dataset
    attached" into a bare NameError raised before the helpful SystemExit below could run.
    """
    f = globals().get("__file__")
    return str(Path(f).resolve().parents[1] / "pipeline" / "preprocess.py") if f else None


def _bootstrap_preprocess() -> None:
    import glob
    # Spawned decode workers re-import this module (see _pool), so this runs once per worker as
    # well as in the parent. The env var is how the parent hands them the answer: without it
    # every worker repeats the search, and a `**` pattern under /kaggle/input descends into the
    # competition data -- ~29k series directories holding 819,640 files on a mount measured at
    # ~19 ms per open. Paid 8 times at pool startup that is minutes of dead time, against a 9 h
    # session cap and a weekly GPU quota.
    global SEARCHED_FOR_PREPROCESS
    cached = os.environ.get(PREPROCESS_DIR_ENV)
    if cached and (Path(cached) / "preprocess.py").exists():
        sys.path.insert(0, cached)
        return
    SEARCHED_FOR_PREPROCESS = True

    # BOUNDED DEPTH FIRST, recursion only as a fallback. Kaggle nests sources under
    # competitions/ and datasets/ when a kernel has more than one, so the depth is not fixed
    # (measured 2026-08-07 -- /kaggle/input held exactly those two and a one-level glob found
    # nothing) -- but it is shallow, and every depth below 3 is image data.
    pats = [f"/kaggle/input/{'*/' * d}pipeline/preprocess.py" for d in range(4)]
    pats += ["/kaggle/usr/lib/*/preprocess.py", "/kaggle/usr/lib/**/preprocess.py",
             "/kaggle/input/**/pipeline/preprocess.py", _repo_preprocess()]
    for pat in pats:
        hits = sorted(glob.glob(pat, recursive=True)) if pat else []
        if hits:
            found = str(Path(hits[0]).parent)
            sys.path.insert(0, found)
            os.environ[PREPROCESS_DIR_ENV] = found      # workers inherit this; see above
            return
    listing = sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*"))
    raise SystemExit(
        "cannot find pipeline/preprocess.py. Attach the rsna-knee-code Dataset to this "
        f"notebook. /kaggle/input currently holds: {listing}")


_bootstrap_preprocess()
from preprocess import (BATCH_HINT, EMBED_DIM, IMG_SIZE, MODEL, PLANE_ID,  # noqa: E402
                        PREPROCESS_VERSION, build_study_index, find_competition_root,
                        embed, load_series, manifest, pick_device, to_25d)

BATCH = BATCH_HINT
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
OUT = Path("/kaggle/working/features")

# Decode runs in worker processes so it overlaps the GPU instead of alternating with it.
# PLAN.md 5 says this about the submission notebook -- "multiprocess it (CPU-bound, will
# otherwise starve the GPU)" -- and it is just as true here, where ~700k slices have to be
# decoded. Serial decode roughly doubles the wall clock of this script, which is the difference
# between a couple of Kaggle sessions and several, against a 9h cap and a weekly GPU quota.
#
# The mount is LATENCY-bound, not bandwidth- or CPU-bound: kaggle_01b measured ~19 ms per file
# open, and the cache needs ~700k of them. That is ~10 h serial, which is why the workers are
# not optional and why oversubscribing cores helps -- they are mostly blocked on I/O. Kaggle
# gives ~4 usable cores, hence 8; capped off Kaggle so a 2-core box does not spawn 8 processes.
# Decode is not PURELY I/O though (normalise_and_resample runs torch.quantile and F.interpolate
# over the whole volume), so _worker_init pins each child to one intra-op thread -- 8 children
# at torch's default of one thread per core would be ~32 compute threads on 4 cores.
N_WORKERS = int(os.environ.get("N_WORKERS", min(8, 2 * (os.cpu_count() or 2))))
# PREFETCH bounds how many decoded volumes are in flight; each is ~27 MB at 32 slices of
# 457x457 float32, so 16 is ~430 MB. That sits on top of ~250-400 MB of RSS per spawned worker
# (they re-import torch/numpy/pandas and share nothing, unlike fork) against Kaggle's ~13 GB.
PREFETCH = int(os.environ.get("PREFETCH", 16))

# Series counts at which build_cache prints a throughput probe. A module constant so --self-test
# can lower it; the first threshold used to sit above anything the self-test produced, so the
# probe and its "not parallelising" warning shipped without ever having been executed.
PROBE_AT = (25, 100, 400)


def _worker_init() -> None:
    """Runs once per spawned child. Keeps one decode process to one compute thread."""
    torch.set_num_threads(1)


def _pool(max_workers):
    """ProcessPoolExecutor on a SPAWN context.

    The default on Linux is fork, and this process initialises CUDA (timm .to('cuda')) before
    the pool is created. Forking a process that already holds a CUDA context is documented as
    unsafe, and the observed cost was severe: the first 224 run tracked the SERIAL throughput
    curve for 9 h against a ~2.7 h estimate with 4 workers. Spawn starts clean children that
    never inherit the context. They re-import this module, which is why _decode_task is
    top-level, why main() sits behind an __name__ guard, and why _bootstrap_preprocess caches
    its answer in the environment -- re-import means each child would otherwise repeat the
    search across the 570 GB mount.
    """
    return ProcessPoolExecutor(max_workers=max_workers,
                               mp_context=multiprocessing.get_context("spawn"),
                               initializer=_worker_init)


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
    t0 = t_steady = time.time()

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
            if n_done_series == 0:
                # Steady-state clock. Spawning 8 children costs a few seconds of interpreter
                # startup and torch imports; folding that into the probe below made the
                # 25-series rate startup-dominated and could trip the warning spuriously.
                t_steady = time.time()
            nxt = next(it, None)
            if nxt is not None:
                futures.append(pool.submit(_decode_task, nxt))

            lat_seen[src] = lat_seen.get(src, 0) + 1
            if vol is not None:
                e = embed_fn(to_25d(vol))
                pending[study].append((k, e, PLANE_ID.get(plane_name, -1), fs_flag,
                                       {"L": 0, "R": 1}.get(side, -1)))
            n_done_series += 1
            if n_done_series in PROBE_AT:
                el = time.time() - t_steady
                rate = (n_done_series - 1) / max(el, 1e-9)      # first result started the clock
                print(f"  PROBE {n_done_series} series in {el:.0f}s = {rate:.2f} series/s "
                      f"-> ~{len(todo) / max(rate, 1e-9) / 3600:.1f} h for this shard "
                      f"({N_WORKERS} workers)")
                if n_done_series == PROBE_AT[1] and rate < 0.35:
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
    # Check the GPU before ANYTHING else -- before indexing 4,410 directories and before timm
    # pulls DINOv2 weights. A bad draw should cost seconds so retrying is nearly free.
    dev = pick_device()
    if dev != "cuda":
        raise SystemExit(f"refusing to start on '{dev}'. Re-run to draw a different GPU.")

    import timm
    root = find_competition_root()
    OUT.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(root / "train_series.csv")
    studies = sorted(series.StudyInstanceUID.unique())
    mine = [s for i, s in enumerate(studies) if i % N_SHARDS == SHARD]
    print(f"shard {SHARD}/{N_SHARDS}: {len(mine):,} of {len(studies):,} studies")
    print(f"preprocess version {PREPROCESS_VERSION}")
    print(f"{N_WORKERS} decode workers, prefetch {PREFETCH}")

    # dynamic_img_size is NOT optional. This checkpoint is 518-native (1,369 position tokens),
    # and timm asserts on any other input size unless the position embedding can be interpolated
    # at forward time: "Input height (224) doesn't match model (518)". IMG_SIZE defaults to 224
    # here, so without this flag the very first embed() raises -- after the GPU guard passes,
    # after the corpus walk, and after the weights download. The self-test never caught it
    # because it injects a fake embed_fn and so never builds the real backbone.
    #
    # The flag rather than img_size=IMG_SIZE, because pos_embed then stays at its native shape
    # and kaggle_03 can load the same checkpoint at either resolution. Measured 2026-08-08 on
    # timm 1.0.28: identical output to img_size=518 at 518, max|diff| 2.5e-05 at 224 -- an order
    # of magnitude under the fp16 the cache is stored in.
    model = timm.create_model(MODEL, pretrained=True, num_classes=0,
                              dynamic_img_size=True).eval().to(dev)
    print(f"{MODEL} on {dev}, prefix_tokens={model.num_prefix_tokens}")

    # Smoke the real embed path on one synthetic slice before touching a single DICOM. This is
    # pick_device()'s argument applied to the backbone: the failure above cost nothing to detect
    # and a whole session to discover. Checks the resolution AND the 1536-wide concat.
    probe = embed(model, torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), dev, BATCH)
    if probe.shape != (1, EMBED_DIM):
        raise SystemExit(f"backbone smoke test: embed() returned {probe.shape}, expected "
                         f"(1, {EMBED_DIM}) at IMG_SIZE={IMG_SIZE}. The cache would be unusable.")
    print(f"backbone smoke test OK at {IMG_SIZE}px -> {EMBED_DIM}-d")

    done, skipped, lat_seen = build_cache(
        root, mine, series.set_index("SeriesInstanceUID"), OUT,
        embed_fn=lambda x: embed(model, x, dev, BATCH))

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


def _pool_probe_task(delay):
    """Trivial picklable payload, only for _pool_self_test. Must stay top-level."""
    time.sleep(delay)
    return os.getpid(), torch.get_num_threads(), SEARCHED_FOR_PREPROCESS


def _pool_self_test() -> None:
    """Exercise the real spawn pool. This is the mechanism that cost 9 h when it was wrong.

    _SerialPool covers the scheduling, but it cannot see either of the two things that actually
    burned a session: whether the pool parallelises AT ALL -- the first 224 run tracked the
    serial throughput curve for 9 h -- and whether a spawned child repeats the /kaggle/input
    search on re-import instead of reading the answer out of the environment.
    """
    with _pool(max_workers=2) as pool:
        # Warm up first: the timing below must measure the work, not two interpreters starting
        # and importing torch. The sleep is what forces BOTH children up.
        for f in [pool.submit(_pool_probe_task, 0.2) for _ in range(2)]:
            f.result()
        t0 = time.time()
        results = [f.result() for f in [pool.submit(_pool_probe_task, 0.25) for _ in range(4)]]
        el = time.time() - t0

    pids = {r[0] for r in results}
    assert len(pids) == 2, f"spawn pool did not fan out across workers: {pids}"
    assert el < 0.8, f"4 x 0.25s on 2 workers took {el:.2f}s -- serial, not parallel"
    assert all(r[1] == 1 for r in results), f"workers not pinned to one thread: {results}"
    assert not any(r[2] for r in results), (
        f"a child re-ran the preprocess search instead of reading {PREPROCESS_DIR_ENV}; on "
        f"Kaggle that is a full walk of the 570 GB mount per worker: {results}")
    print(f"  spawn pool: 4 x 0.25s tasks in {el:.2f}s over {len(pids)} workers, 1 thread each, "
          f"preprocess dir inherited (no re-glob)")


def self_test() -> None:
    import shutil
    import tempfile

    global PROBE_AT
    # The real thresholds start at 25 series and the corpus below holds 21, so the probe block
    # -- including its "not parallelising" warning -- never ran under the self-test that is this
    # file's stated correctness gate. Lower them instead of inflating the corpus.
    PROBE_AT = (3, 5, 400)

    _pool_self_test()

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
