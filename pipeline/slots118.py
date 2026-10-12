"""Stage C slice layout: Stage B with the coronal and axial fluid series sampled as densely as the sagittal.

Stage B keeps a median 87% of the sagittal slices but 53% of the coronal and 44% of the axial (one slice every
6.5 and 9.0 mm). The findings read on those planes -- Lateral OA and MCL on the coronal, PF OA on the axial -- are the
ones the image model reads worse than the report does (`docs/REVIEW-2026-10-12.md` §2). Here cor_1 and ax_1 take 26
slices each (about every 4 and 5 mm); every other slot is Stage B's.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

import slots64 as _A
from slots64 import FOV_MM, GROUP, Slot, pick_depths, source_step  # noqa: F401

SPEC = "c8-118slice-384"
IMG = 384
TARGET_MM = FOV_MM / IMG

SLOTS: tuple[Slot, ...] = (
    Slot("sag_1", "Sagittal", 26, 1),
    Slot("sag_2", "Sagittal", 20, 0),
    Slot("cor_1", "Coronal",  26, 1),       # Stage B: 16
    Slot("cor_2", "Coronal",  10, None),
    Slot("ax_1",  "Axial",    26, None),    # Stage B: 14
    Slot("sag_3", "Sagittal",  6, None),
    Slot("ax_2",  "Axial",     4, None),
)
N_SLICES = sum(s.n_slices for s in SLOTS)      # 118
N_WINDOWS = sum(s.n_windows for s in SLOTS)    # 104
SLOT_INDEX = {s.name: i for i, s in enumerate(SLOTS)}


def slot_offsets() -> list[tuple[int, int]]:
    return _A.slot_offsets(SLOTS)


def window_ids():
    return _A.window_ids(SLOTS)


def assign_series(series: list[dict]) -> dict[str, dict | None]:
    """`slots64.assign_series` over these slots: the same series reach the same slots as on Stage B."""
    return _A.assign_series(series, SLOTS)


def manifest(depth_range: tuple[float, float] = (_A.DEPTH_LO, _A.DEPTH_HI)) -> dict:
    return {"spec": SPEC, "fov_mm": FOV_MM, "target_mm": TARGET_MM, "img": IMG,
            "depth_range": list(depth_range), "group": GROUP,
            "n_slices": N_SLICES, "n_windows": N_WINDOWS,
            "window": "consecutive sampled slices",
            "per_series": ["n_available", "source_step"],
            "slots": [asdict(s) for s in SLOTS]}


def _self_test() -> None:
    import slots96 as B
    assert N_SLICES == 118, N_SLICES
    assert N_WINDOWS == 104, N_WINDOWS
    sid, pid, ctr = window_ids()
    assert len(ctr) == N_WINDOWS and ctr.max() < N_SLICES
    for (a, b), s in zip(slot_offsets(), SLOTS):
        own = ctr[(ctr >= a) & (ctr < b)]
        assert len(own) == s.n_windows and own.min() > a and own.max() < b - 1, s.name
    # same slots, planes and preferences as Stage B; only two sizes differ
    assert [(s.name, s.plane, s.fluid) for s in SLOTS] == [(s.name, s.plane, s.fluid) for s in B.SLOTS]
    assert {s.name for s, t in zip(SLOTS, B.SLOTS) if s.n_slices != t.n_slices} == {"cor_1", "ax_1"}
    ser = [{"plane": "Sagittal", "fluid": 1, "n_slices": 30, "id": "S1"},
           {"plane": "Coronal", "fluid": 1, "n_slices": 30, "id": "C1"},
           {"plane": "Coronal", "fluid": 0, "n_slices": 28, "id": "C2"},
           {"plane": "Axial", "fluid": 1, "n_slices": 32, "id": "A1"}]
    assert {k: (v or {}).get("id") for k, v in assign_series(ser).items()} == \
           {k: (v or {}).get("id") for k, v in B.assign_series(ser).items()}
    # a 30-slice coronal: Stage B takes every 1.8th slice, this takes every 1.1th
    assert source_step(30, 16, 0.02, 0.98) > 1.7 and source_step(30, 26, 0.02, 0.98) < 1.2
    print(f"slots118 self-test OK  ({N_SLICES} slices -> {N_WINDOWS} windows, {IMG} px, "
          f"{FOV_MM:.0f} mm = {TARGET_MM:.3f} mm/px)")


if __name__ == "__main__":
    _self_test()
