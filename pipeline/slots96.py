"""Stage B slice layout: seven series slots, 96 slices at 384 px, orientation-canonical."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

import slots64 as _A
from slots64 import FOV_MM, GROUP, Slot, pick_depths, source_step  # noqa: F401

SPEC = "c7-96slice-384"
IMG = 384
TARGET_MM = FOV_MM / IMG

SLOTS: tuple[Slot, ...] = (
    Slot("sag_1", "Sagittal", 26, 1),       # the meniscus/ACL plane: densest
    Slot("sag_2", "Sagittal", 20, 0),
    Slot("cor_1", "Coronal",  16, 1),
    Slot("cor_2", "Coronal",  10, None),
    Slot("ax_1",  "Axial",    14, None),
    Slot("sag_3", "Sagittal",  6, None),    # the third sagittal, present in ~22% of studies
    Slot("ax_2",  "Axial",     4, None),    # the second axial, present in ~25%
)
N_SLICES = sum(s.n_slices for s in SLOTS)      # 96
N_WINDOWS = sum(s.n_windows for s in SLOTS)    # 82
SLOT_INDEX = {s.name: i for i, s in enumerate(SLOTS)}


def slot_offsets() -> list[tuple[int, int]]:
    return _A.slot_offsets(SLOTS)


def window_ids():
    return _A.window_ids(SLOTS)


def assign_series(series: list[dict]) -> dict[str, dict | None]:
    """`slots64.assign_series` over these slots. Order matters and is deliberate: the five Stage A
    slots pick first, so sag_3 / ax_2 only ever take a series that would otherwise go unused."""
    return _A.assign_series(series, SLOTS)


def manifest(depth_range: tuple[float, float] = (_A.DEPTH_LO, _A.DEPTH_HI)) -> dict:
    return {"spec": SPEC, "fov_mm": FOV_MM, "target_mm": TARGET_MM, "img": IMG,
            "depth_range": list(depth_range), "group": GROUP,
            "n_slices": N_SLICES, "n_windows": N_WINDOWS,
            "window": "consecutive sampled slices",
            "per_series": ["n_available", "source_step"],
            "slots": [asdict(s) for s in SLOTS]}


def _self_test() -> None:
    assert N_SLICES == 96, N_SLICES
    assert N_WINDOWS == 82, N_WINDOWS
    assert IMG % 16 == 0
    sid, pid, ctr = window_ids()
    assert len(ctr) == N_WINDOWS and ctr.max() < N_SLICES
    for (a, b), s in zip(slot_offsets(), SLOTS):
        own = ctr[(ctr >= a) & (ctr < b)]
        assert len(own) == s.n_windows and own.min() > a and own.max() < b - 1, s.name
    # the extra slots take only what the Stage A five leave behind
    ser = [{"plane": "Sagittal", "fluid": 1, "n_slices": 30, "id": "S1"},
           {"plane": "Sagittal", "fluid": 0, "n_slices": 28, "id": "S2"},
           {"plane": "Sagittal", "fluid": 0, "n_slices": 26, "id": "S3"},
           {"plane": "Axial", "fluid": 0, "n_slices": 40, "id": "A1"},
           {"plane": "Axial", "fluid": 1, "n_slices": 20, "id": "A2"}]
    a = assign_series(ser)
    assert (a["sag_1"]["id"], a["sag_2"]["id"], a["sag_3"]["id"]) == ("S1", "S2", "S3")
    assert (a["ax_1"]["id"], a["ax_2"]["id"]) == ("A1", "A2")      # longer axial first
    assert a["cor_1"] is None and a["cor_2"] is None
    two = assign_series(ser[:2])
    assert two["sag_3"] is None and two["ax_2"] is None
    print(f"slots96 self-test OK  ({N_SLICES} slices -> {N_WINDOWS} windows, {IMG} px, "
          f"{FOV_MM:.0f} mm = {TARGET_MM:.3f} mm/px)")


if __name__ == "__main__":
    _self_test()
