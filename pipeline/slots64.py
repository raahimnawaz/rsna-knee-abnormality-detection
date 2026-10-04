"""Stage A slice layout: five series slots, 64 slices, 140 mm field of view."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

FOV_MM = 140.0
TARGET_MM = 0.35
IMG = 336

# --- sampling -----------------------------------------------------------------------------
DEPTH_LO, DEPTH_HI = 0.06, 0.94
GROUP = 3


@dataclass(frozen=True)
class Slot:
    name: str
    plane: str
    n_slices: int
    fluid: int | None      # 1 prefer fluid-sensitive, 0 prefer NOT, None don't care

    @property
    def n_windows(self) -> int:
        """Stride 1, dropping the two edge positions -- their neighbour lies outside the slot."""
        return max(self.n_slices - 2, 0)


SLOTS: tuple[Slot, ...] = (
    Slot("sag_1", "Sagittal", 18, 1),
    Slot("sag_2", "Sagittal", 14, 0),
    Slot("cor_1", "Coronal",  12, 1),
    Slot("cor_2", "Coronal",   8, None),     # the second coronal: `used` excludes cor_1's pick
    Slot("ax",    "Axial",    12, None),
)

N_SLICES = sum(s.n_slices for s in SLOTS)      # 64
N_WINDOWS = sum(s.n_windows for s in SLOTS)    # 54
SLOT_INDEX = {s.name: i for i, s in enumerate(SLOTS)}


def pick_depths(n_available: int, n_want: int,
                lo_frac: float = DEPTH_LO, hi_frac: float = DEPTH_HI) -> np.ndarray:
    """Evenly across 6-94% of the stack (or `lo_frac`-`hi_frac`), not only its middle."""
    if n_available <= 0:
        return np.zeros(n_want, dtype=int)
    if n_available == 1:
        return np.zeros(n_want, dtype=int)
    lo = lo_frac * (n_available - 1)
    hi = hi_frac * (n_available - 1)
    return np.clip(np.round(np.linspace(lo, hi, n_want)), 0, n_available - 1).astype(int)


def source_step(n_available: int, n_want: int,
                lo_frac: float = DEPTH_LO, hi_frac: float = DEPTH_HI) -> float:
    """Mean distance, in SOURCE slices, between consecutive sampled slices of one series."""
    d = pick_depths(n_available, n_want, lo_frac, hi_frac)
    return float(np.diff(d).mean()) if len(d) > 1 else 0.0


def window_ids(slots: tuple[Slot, ...] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed (slot_id, position_id, centre-slice index) for all 54 windows, same for every study."""
    slot_id, pos_id, centre = [], [], []
    base = 0
    for si, s in enumerate(slots or SLOTS):
        for p in range(s.n_windows):
            slot_id.append(si)
            pos_id.append(p)
            centre.append(base + p + 1)      # centre of the triple; +1 skips the dropped edge
        base += s.n_slices
    return (np.asarray(slot_id, dtype=np.int64),
            np.asarray(pos_id, dtype=np.int64),
            np.asarray(centre, dtype=np.int64))


def slot_offsets(slots: tuple[Slot, ...] | None = None) -> list[tuple[int, int]]:
    """(start, stop) of each slot inside the 64-slice cache row."""
    out, base = [], 0
    for s in slots or SLOTS:
        out.append((base, base + s.n_slices))
        base += s.n_slices
    return out


def assign_series(series: list[dict], slots: tuple[Slot, ...] | None = None) -> dict[str, dict | None]:
    """Map each slot to a series, or None when the study has nothing suitable."""
    out: dict[str, dict | None] = {}
    used: set[int] = set()
    for s in slots or SLOTS:
        cands = [(i, c) for i, c in enumerate(series)
                 if c.get("plane") == s.plane and i not in used]
        if not cands:
            out[s.name] = None
            continue
        def key(ic):
            i, c = ic
            miss = 0 if s.fluid is None else int(c.get("fluid", 0) != s.fluid)
            return (miss, -int(c.get("n_slices", 0)))
        cands.sort(key=key)
        pick = cands[0]
        used.add(pick[0])
        out[s.name] = pick[1]
    return out


def manifest(depth_range: tuple[float, float] = (DEPTH_LO, DEPTH_HI)) -> dict:
    """Stamped into every cache. `PREPROCESS_VERSION`'s purpose is to make drift loud."""
    return {"spec": "c6-64slice", "fov_mm": FOV_MM, "target_mm": TARGET_MM, "img": IMG,
            "depth_range": list(depth_range), "group": GROUP,
            "n_slices": N_SLICES, "n_windows": N_WINDOWS,
            "window": "consecutive sampled slices",
            "per_series": ["n_available", "source_step"],
            "slots": [asdict(s) for s in SLOTS]}


def _self_test() -> None:
    assert N_SLICES == 64, N_SLICES
    assert N_WINDOWS == 54, N_WINDOWS

    sid, pid, ctr = window_ids()
    assert len(sid) == len(pid) == len(ctr) == 54
    assert ctr.max() < N_SLICES, "a window centre points past the cache row"
    for (a, b), s in zip(slot_offsets(), SLOTS):
        own = ctr[(ctr >= a) & (ctr < b)]
        assert len(own) == s.n_windows, (s.name, len(own), s.n_windows)
        assert own.min() > a and own.max() < b - 1, f"{s.name}: window triple leaves the slot"

    # Depth sampling reaches the outer thirds.
    d = pick_depths(100, 18)
    assert d.min() <= 6 and d.max() >= 93, f"depths do not span 6-94%: {d.min()}..{d.max()}"
    assert len(d) == 18 and (np.diff(d) >= 0).all()
    # What a window spans in the source stack: neighbours only when the stack is ~1.1x the slot.
    assert abs(source_step(20, 18) - 1.0) < 0.1, source_step(20, 18)
    assert 2.5 < source_step(60, 18) < 3.5 and source_step(12, 18) < 1.0
    # Short and degenerate series must not raise.
    for n, want in ((1, 18), (2, 18), (5, 12), (0, 8)):
        got = pick_depths(n, want)
        assert len(got) == want and got.max() < max(n, 1), (n, want, got)

    # Slot assignment: preference is a sort key, not a filter.
    ser = [{"plane": "Sagittal", "fluid": 0, "n_slices": 30},
           {"plane": "Sagittal", "fluid": 1, "n_slices": 25},
           {"plane": "Axial", "fluid": 1, "n_slices": 20}]
    a = assign_series(ser)
    assert a["sag_1"]["fluid"] == 1, "sag_1 should prefer the fluid-sensitive sagittal"
    assert a["sag_2"]["fluid"] == 0, "sag_2 should take the non-fluid sagittal"
    assert a["cor_1"] is None and a["cor_2"] is None, "no coronal -> masked, never substituted"
    assert a["ax"] is not None
    # Three coronals: cor_1 takes the fluid one, cor_2 the longer of the other two.
    a3 = assign_series([{"plane": "Coronal", "fluid": 1, "n_slices": 30, "id": "A"},
                        {"plane": "Coronal", "fluid": 0, "n_slices": 28, "id": "B"},
                        {"plane": "Coronal", "fluid": 0, "n_slices": 12, "id": "C"}])
    assert (a3["cor_1"]["id"], a3["cor_2"]["id"]) == ("A", "B"), (a3["cor_1"], a3["cor_2"])
    # Only-non-fluid sagittals still fill sag_1 rather than leaving zeros.
    a2 = assign_series([{"plane": "Sagittal", "fluid": 0, "n_slices": 30}])
    assert a2["sag_1"] is not None and a2["sag_2"] is None
    print(f"slots64 self-test OK  ({N_SLICES} slices -> {N_WINDOWS} windows, "
          f"{FOV_MM:.0f} mm, {DEPTH_LO:.0%}-{DEPTH_HI:.0%})")


if __name__ == "__main__":
    _self_test()
