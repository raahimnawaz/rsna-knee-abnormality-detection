"""The slot-pixel cache: PIXELS, not embeddings, so the backbone can be fine-tuned.

PLAN 9 / README Phase 0 step 4. `IMPROVEMENTS.md` 2e is the reason this file exists: the frozen
DINOv2 embedding cache in `data/features_*` cannot fine-tune at any resolution, under any head,
with any labels -- and `pilkwang`, the 0.891 fork, opens its last six encoder blocks
(`UNFREEZE_LAST=6`, `LR_BACKBONE=8e-6`). Trainability is the primary constraint, and a cache of
1536-d vectors can never supply it. So this stores uint8 pixels and the encoder runs inside the
training loop.

WHAT A SLOT IS. One study becomes a fixed list of SLOT IMAGES, each a 3-channel 336x336 uint8
tile: three adjacent slices stacked as the channels, exactly the 2.5D convention `to_25d` already
uses. Six PROTOCOL slots reproduce the fork's layout -- one per (plane x fluid-sensitive) -- and
that set, and only that set, is what the reproduction gate in step 5 runs against. The ANATOMICAL
slots below are the divergence, and they are built here rather than retrofitted because
retrofitting them would mean a second full pass over 458 GB of NIfTI.

Cost, measured rather than budgeted (`pipeline/bench_cache_build.py`, IMPROVEMENTS 2h): the six
protocol slots are ~16 min for the whole corpus. It is a NIfTI read and an in-plane resample, not
an encoder pass, which is why the entry price for the architecture that CAN fine-tune is a coffee
break while the one that cannot cost 21 h, 9 h and four Kaggle sessions.

    26,442 tiles (4,407 studies x 6) x 3 x 336 x 336 bytes = 8.96 GB

WHY THE ANATOMICAL SLOTS ARE SHAPED THE WAY THEY ARE. IMPROVEMENTS 2d measured that the labels
which fail are exactly the ones whose finding is millimetre-scale -- Fracture 0.494, Lateral
Meniscus 0.526, Contusion 0.603 -- while centimetre-scale ones already work (Baker's 0.919,
Medial OA 0.913). At 160 mm across 336 px a tear line is about one pixel. A crop is the cheapest
possible answer: half the field of view at the same 336 px is **twice the effective resolution**
over the compartment where the finding lives, for identical compute. Two unrelated fields arrive
at the same answer independently (`REFERENCE.md` 4.3-4.4: localize, then embed crops).

The crops are FIXED, with no detector, and the reason that is legitimate is measured here:

  1. `normalise_and_resample` puts every series on one grid -- 0.35 mm/px, 160 mm FOV, 457 px --
     and the knee is protocol-centred, so image coordinates are already millimetres from the
     joint.
  2. The in-plane axes are canonical per plane. Measured over the 396-series geometry sample,
     the nearest signed LPS axis is unanimous **132/132 for every plane and every axis**, with
     median obliquity 2.4-8.2 deg and p90 under 20 deg:

         Axial     col+ = +x (Left)      row+ = +y (Posterior)   normal = +z
         Coronal   col+ = +x (Left)      row+ = -z (Inferior)    normal = +y
         Sagittal  col+ = +y (Posterior) row+ = -z (Inferior)    normal = -x

     The "374 distinct IOP rows" the geometry kernel reported is float obliquity, not a mixture
     of conventions. (Re-measure this if the corpus grows: it is the whole licence for fixed
     boxes.)
  3. `canonicalise` mirrors left knees onto CANONICAL_SIDE = 'R'. A right knee's medial side
     faces the midline, which is +x, so after canonicalisation **increasing column index is
     medial** for axial and coronal in every study.

Sagittal is different and it is where K16 binds. Medial/lateral is the SLICE axis there, the
normal is -x, so ascending spatial order runs medial -> lateral for a canonical right knee. That
ordering is only true of a volume known to be in ascending order -- and 33% of the NIfTI series
are stored back-to-front with nothing in the file to say which (K16, `validate_nifti` check 4b,
n=51 stratified: Axial 12/12 forward, Coronal 14/18, Sagittal 8/21). So a sagittal medial slab
without the direction bit is a coin flip on the axis four of the twelve labels depend on.

This module therefore REFUSES to build the sagittal anatomical slots when the bit is absent,
rather than building them wrong. That is the same choice `canonicalise` makes with unknown
laterality: honest beats safe, because a silently mirrored slab has no symptom.

    python pipeline/slot_cache.py --slots protocol          # the 6 the gate needs, ~16 min
    python pipeline/slot_cache.py --slots anatomical        # the divergence, second pass
    python pipeline/slot_cache.py --slots all --limit 40    # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("TARGET_MM", "0.35")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from pipeline import preprocess as pp  # noqa: E402

D = PROJ / "data"
NII = D / "nifti" / "nifti_train"

TILE = 336            # the fork's input size; dinov2-small at patch-14 -> 24x24 patches
GROUP = 3             # adjacent slices stacked as channels, matching to_25d
GRID = int(round(pp.FOV_MM / pp.TARGET_MM))     # 457 px, the shared post-resample grid


@dataclass(frozen=True)
class Slot:
    """One tile: which series supplies it, which slices, and which box in millimetres.

    `depth` indexes the slice axis as a fraction of the series, so it is resolution-independent
    and survives a series with 11 slices as well as one with 320.

    `box_mm` is the side of a SQUARE field of view. Square on purpose: a half-width strip resized
    to 336x336 would stretch the anatomy 2:1 along one axis, and the backbone has never seen a
    knee stretched 2:1. None means the full 160 mm.

    `centre_mm` is (row, col) offset from the image centre, in the canonicalised frame described
    in the module docstring -- so +col is medial for axial/coronal, -row is anterior for axial,
    -col is anterior for sagittal.
    """
    name: str
    plane: str
    fluid: int
    depth: float = 0.5
    box_mm: float | None = None
    centre_mm: tuple[float, float] = (0.0, 0.0)
    needs_direction: bool = False


# --- the six the reproduction gate runs against. Do not change these without re-running step 5.
PROTOCOL = [
    Slot("ax_fs", "Axial", 1), Slot("ax_nf", "Axial", 0),
    Slot("cor_fs", "Coronal", 1), Slot("cor_nf", "Coronal", 0),
    Slot("sag_fs", "Sagittal", 1), Slot("sag_nf", "Sagittal", 0),
]

# --- the divergence. Each one names the label it is aimed at; none of them is a claim yet.
# 84 mm boxes at 336 px = 0.25 mm/px, against 0.48 mm/px for the full field of view.
ANATOMICAL = [
    # Sagittal slabs: the compartment IS the slice position, so these take the full field of
    # view and vary only in depth. Both need K16 -- see the module docstring.
    Slot("sag_med", "Sagittal", 1, depth=0.25, needs_direction=True),   # medial meniscus, medial OA
    Slot("sag_lat", "Sagittal", 1, depth=0.75, needs_direction=True),   # LATERAL MENISCUS, 0.526
    # Coronal compartments: medial/lateral is in-plane here, so a box does it and no bit is
    # needed. +30 mm in col is medial after canonicalisation.
    Slot("cor_med", "Coronal", 1, box_mm=84.0, centre_mm=(0.0, 30.0)),
    Slot("cor_lat", "Coronal", 1, box_mm=84.0, centre_mm=(0.0, -30.0)),
    # Patellofemoral: anterior. -row is anterior on axial, -col on sagittal.
    Slot("ax_pf", "Axial", 1, box_mm=84.0, centre_mm=(-30.0, 0.0)),     # patellar fracture, PF OA
    Slot("sag_pf", "Sagittal", 1, box_mm=84.0, centre_mm=(-10.0, -30.0)),
]

BY_NAME = {s.name: s for s in PROTOCOL + ANATOMICAL}


# Flags that change what a tile CONTAINS rather than which tiles exist. Two caches that
# disagree on any of these cannot be fed to one model: the disagreement is per-study and
# silent, which is the same class of failure PREPROCESS_VERSION exists to catch.
CACHE_COMPAT_KEYS = ("preprocess_version", "tile", "group", "grid", "target_mm", "fov_mm",
                     "canonical_side", "sagittal_lr_slice_flip", "slices_per_series")


def assert_caches_compatible(*manifest_paths) -> None:
    """Raise unless every manifest agrees on the keys that change pixel values.

    CALL THIS BEFORE ANY RUN THAT CONSUMES MORE THAN ONE CACHE TAG. The live instance of this
    hazard, 2026-08-10: `tiles_protocol` was built before the K16 bit existed
    (`sagittal_lr_slice_flip=False`, `direction_bits=0`) and the anatomical tiles need
    `SAGITTAL_LR=1`, because "slice 25% is medial" is exactly inverted for the 43% of studies
    that are left knees. Mixing them puts half the sagittal slabs on the wrong compartment for
    those studies, per study, with no symptom other than four labels quietly failing to improve.

    `direction_bits` is deliberately NOT in the compat set: it is a count of how many series had
    a resolved bit, so it grows as the corpus downloads and is not a property of the arithmetic.
    `sagittal_lr_slice_flip` is the flag that actually changes the pixels.
    """
    import json as _json
    mans = [(str(p), _json.loads(Path(p).read_text())) for p in manifest_paths]
    if len(mans) < 2:
        return
    ref_name, ref = mans[0]
    for name, m in mans[1:]:
        bad = {k: (ref.get(k), m.get(k)) for k in CACHE_COMPAT_KEYS if ref.get(k) != m.get(k)}
        if bad:
            lines = "\n".join(f"    {k}: {a!r} vs {b!r}" for k, (a, b) in bad.items())
            raise SystemExit(
                f"incompatible caches -- these were built under different preprocessing:\n"
                f"  {ref_name}\n  {name}\n{lines}\n\n"
                "Rebuild the older one with the newer flags (~21 min for the whole corpus) "
                "rather than\nmixing them. See IMPROVEMENTS.md 2n, 'HAZARD'.")


def slot_set(which: str) -> list[Slot]:
    if which == "protocol":
        return list(PROTOCOL)
    if which == "anatomical":
        return list(ANATOMICAL)
    if which == "all":
        return PROTOCOL + ANATOMICAL
    names = [n.strip() for n in which.split(",") if n.strip()]
    missing = [n for n in names if n not in BY_NAME]
    if missing:
        raise SystemExit(f"unknown slot(s) {missing}; known: {sorted(BY_NAME)}")
    return [BY_NAME[n] for n in names]


def crop_box(slot: Slot) -> tuple[int, int, int, int]:
    """-> (r0, r1, c0, c1) in the 457 px grid. Clamped to the grid, so an off-edge box shrinks
    rather than wrapping or padding with black that the model would read as anatomy."""
    if slot.box_mm is None:
        return 0, GRID, 0, GRID
    half = int(round(slot.box_mm / pp.TARGET_MM / 2))
    cr = GRID // 2 + int(round(slot.centre_mm[0] / pp.TARGET_MM))
    cc = GRID // 2 + int(round(slot.centre_mm[1] / pp.TARGET_MM))
    r0, r1 = max(0, cr - half), min(GRID, cr + half)
    c0, c1 = max(0, cc - half), min(GRID, cc + half)
    return r0, r1, c0, c1


def tile_from(vol: np.ndarray, slot: Slot) -> np.ndarray | None:
    """[S,457,457] float32 in [0,1] -> [3,336,336] uint8, or None if the series is too thin.

    Quantised to uint8 at the very end. The volume is already in [0,1] from the shared
    percentile normalisation, so this is a 1/255 quantisation of a bounded signal, not a window
    choice -- and it is what makes 9 GB rather than 36 GB.
    """
    s = len(vol)
    if s < GROUP:
        return None
    c = int(round(slot.depth * (s - 1)))
    c = min(max(c, GROUP // 2), s - 1 - GROUP // 2)
    sl = vol[c - GROUP // 2: c + GROUP // 2 + 1]           # [3,457,457]
    r0, r1, c0, c1 = crop_box(slot)
    sl = sl[:, r0:r1, c0:c1]
    t = torch.from_numpy(np.ascontiguousarray(sl))[None]   # [1,3,h,w]
    t = F.interpolate(t, size=(TILE, TILE), mode="bilinear", align_corners=False)[0]
    return (t.clamp(0, 1) * 255).round().to(torch.uint8).numpy()


def load_direction() -> dict[str, str]:
    """SeriesInstanceUID -> 'forward'|'reversed', from the resolved K16 export. {} if absent."""
    p = D / "slice_direction_resolved.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "direction" not in df.columns:
        return {}
    df = df[df.direction.isin(["forward", "reversed"])]
    return dict(zip(df.SeriesInstanceUID, df.direction))


def build(which: str, limit: int | None, out_dir: Path, workers: int) -> None:
    slots = slot_set(which)
    direction = load_direction()

    gated = [s.name for s in slots if s.needs_direction and not direction]
    if gated:
        raise SystemExit(
            f"slots {gated} need the K16 per-series direction bit and "
            f"data/slice_direction_resolved.csv is missing or empty.\n"
            "Medial/lateral is the SLICE axis for sagittal and 33% of series are stored "
            "back-to-front (validate_nifti check 4b), so these slabs would be a coin flip on "
            "the axis four of the twelve labels depend on.\n"
            "Run notebooks/kaggle_01d_slice_direction.py on Kaggle, then "
            "pipeline/resolve_slice_direction.py. Or build --slots protocol, which does not "
            "need the bit.")

    ser = pd.read_csv(D / "train_series.csv")
    path_of, have = {}, set()
    for p in NII.glob("*.nii"):
        st, _, se = p.name[:-4].partition("_")
        path_of[(st, se)] = p
        have.add(st)
    ser = ser[ser.StudyInstanceUID.isin(have)].copy()

    lat = pp.study_laterality(D / "study_meta.csv")
    studies = sorted(ser.StudyInstanceUID.unique())
    if limit:
        studies = studies[:limit]

    # One row per study, one plane in the memmap per slot. Studies missing a series type get a
    # zero tile and a False in the mask; the training loader must consult the mask rather than
    # trusting that a zero tile is black anatomy. Axial non-FS exists for only 971 of the series
    # on disk, so this is the common case, not a corner one.
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = which if which in ("protocol", "anatomical", "all") else "custom"
    mm_path = out_dir / f"tiles_{tag}.u8"
    shape = (len(studies), len(slots), GROUP, TILE, TILE)
    mm = np.lib.format.open_memmap(mm_path.with_suffix(".npy"), mode="w+",
                                   dtype=np.uint8, shape=shape)
    mask = np.zeros((len(studies), len(slots)), dtype=bool)

    # Group the slots by the series they read, so a study opens each .nii once no matter how
    # many tiles come out of it. sag_fs, sag_med, sag_lat and sag_pf are four tiles from one
    # file; reading it four times would quadruple the only expensive step.
    by_series: dict[tuple[str, int], list[tuple[int, Slot]]] = {}
    for j, s in enumerate(slots):
        by_series.setdefault((s.plane, s.fluid), []).append((j, s))

    t0 = time.time()
    n_tiles = 0
    for i, st in enumerate(studies):
        rows = ser[ser.StudyInstanceUID == st]
        side = lat.get(st, (None, "none"))
        for (plane, fluid), members in by_series.items():
            m = rows[(rows.Anatomical_Plane == plane) & (rows.Fluid_Sensitive == fluid)]
            if m.empty:
                continue
            se = m.SeriesInstanceUID.iloc[0]        # first matching series; protocols repeat
            p = path_of.get((st, se))
            if p is None:
                continue
            try:
                vol, _, _ = pp.load_series_nifti(
                    p, plane=plane, laterality=side[0], laterality_source=side[1],
                    slice_direction=direction.get(se))
            except Exception as e:                                     # noqa: BLE001
                print(f"    ! {st[:16]} {plane}_{fluid}: {type(e).__name__} {str(e)[:50]}")
                continue
            if vol is None:
                continue
            for j, slot in members:
                tile = tile_from(vol, slot)
                if tile is None:
                    continue
                mm[i, j] = tile
                mask[i, j] = True
                n_tiles += 1
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(studies) - i - 1)
            print(f"  [{i + 1:>5}/{len(studies)}] {n_tiles:,} tiles  "
                  f"{el / 60:.1f} min elapsed, ETA {eta / 60:.1f} min", flush=True)

    mm.flush()
    idx = pd.DataFrame({"StudyInstanceUID": studies, "row": np.arange(len(studies))})
    for j, s in enumerate(slots):
        idx[f"has_{s.name}"] = mask[:, j]
    idx.to_csv(out_dir / f"index_{tag}.csv", index=False)

    man = {
        "preprocess_version": pp.PREPROCESS_VERSION, "tile": TILE, "group": GROUP,
        "grid": GRID, "target_mm": pp.TARGET_MM, "fov_mm": pp.FOV_MM,
        "canonical_side": pp.CANONICAL_SIDE,
        "sagittal_lr_slice_flip": pp.SAGITTAL_LR_SLICE_FLIP,
        "slices_per_series": pp.SLICES_PER_SERIES,
        "direction_bits": len(direction),
        "slots": [asdict(s) for s in slots],
        "n_studies": len(studies), "n_tiles": int(n_tiles),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / f"manifest_{tag}.json").write_text(json.dumps(man, indent=2))

    el = (time.time() - t0) / 60
    gb = mm.nbytes / 1e9
    print(f"\n{'=' * 66}\nwrote {mm_path.with_suffix('.npy').name}  {gb:.2f} GB  in {el:.1f} min"
          f"\n{'=' * 66}")
    print(f"  studies {len(studies):,}   slots {len(slots)}   tiles {n_tiles:,} "
          f"({n_tiles / (len(studies) * len(slots)):.1%} of the grid filled)")
    for j, s in enumerate(slots):
        print(f"    {s.name:<9} {mask[:, j].sum():>6,} / {len(studies):,}  "
              f"({mask[:, j].mean():>5.1%})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slots", default="protocol",
                    help="protocol | anatomical | all | a comma-separated list of slot names")
    ap.add_argument("--limit", type=int, default=None, help="first N studies, for a smoke test")
    ap.add_argument("--out", default=str(D / "tiles336"))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    print(__doc__.split("\n\n")[0])
    print(f"slots={a.slots}  grid={GRID}px @ {pp.TARGET_MM} mm  tile={TILE}  group={GROUP}\n")
    build(a.slots, a.limit, Path(a.out), a.workers)


if __name__ == "__main__":
    main()
