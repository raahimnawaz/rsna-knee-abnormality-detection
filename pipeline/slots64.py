"""The C6 cache spec: five slots, 64 slices, 6-94% of each stack, 140 mm. `MODEL.md` §2-3.

WHAT THIS CHANGES AND WHY EACH ONE IS A MEASURED DEFECT, NOT A PREFERENCE.

| | tiles336* (what shipped) | here | evidence |
|---|---|---|---|
| slices / study | 18 | **64** | `MODEL.md` §1, frontier takes 64 |
| where in the stack | **every slot `depth 0.5`** | **6-94%** | D2. Lat. Meniscus 0.6422, MCL 0.6433 vs midline ACL 0.8689 |
| crop | 160 mm -> 0.476 mm/px | **140 mm** | §R, +0.0268 at 4.1sigma; the CONFOUNDED-160 confound itself |
| built from | third-party NIfTI repackaging | **competition DICOMs** | D3, costs **0.085** at 14.1sigma |

D3 is the largest single defect this project measured and it is upstream of everything else, so
this spec is only ever built on Kaggle where the DICOMs mount directly. A cache built from any
repackaging -- ours or a public one -- reintroduces it. That is not a throughput argument; 0.085
is larger than every architecture delta in the record combined.

⛔ AMENDED 2026-09-23 (`DECISIONS.md` D3). "A repackaging reintroduces it" is not supported: the fork's
own weights score the same on our NIfTI files as on their DICOMs (§3m). What the port lacked is the
row above this one, coverage: it trained on ONE window per series at depth 0.5. Building on Kaggle
stands because the local corpus is gone. `RUNBOOK.md` §2b tests whether coverage is the 0.085.

⛔ THE SLICE AXIS IS THE POINT. On a sagittal stack the slice axis IS medial-lateral, so the outer
20% that `tiles336*` never sampled is exactly where Lateral Meniscus and the collateral ligaments
live. §3z pre-registered this before the first cache and it was never built. 6-94% rather than
0-100% because the end slices of an MR stack are routinely partial-volume or wrap, and the
frontier's own published range is 6-94%.

WHY 64 SLICES AND NOT 64 WINDOWS ARE CACHED. A window is three CONSECUTIVE SAMPLED slices (§3):
neighbours in this cache, but 0-3 slices apart in the source stack depending on its depth (see
`source_step`). So consecutive windows share two of their three cache slices, and caching windows
would store every slice three times -- 21.6 MB/study instead of 7.2 MB, 95 GB instead of 32 GB --
for no extra information. (Triples of TRUE stack neighbours around each centre would be extra
information and would cost exactly that 3x; `MODEL.md` §3 records why they were not chosen.)
Windows are assembled in the Dataset. This is the only reason the cache fits inside Kaggle's 20 GB per-kernel
output cap when sharded eight ways.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# --- geometry -----------------------------------------------------------------------------
# Crop in millimetres, never pixels: a knee then occupies the same fraction of frame at 0.3 and
# 0.5 mm/px, and the model is never asked to learn a scale difference carrying no medical
# information. `MODEL.md` §2.
FOV_MM = 140.0
TARGET_MM = 0.35
IMG = 336              # patch 16 divides 336 exactly (21x21); 518 does not (32.375)

# --- sampling -----------------------------------------------------------------------------
DEPTH_LO, DEPTH_HI = 0.06, 0.94
GROUP = 3              # 2.5D: three adjacent slices become the three channels of one window


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


# Fluid-sensitive sequences show effusion, synovitis and acute injury; T1/PD show anatomy and
# cartilage. The twelve findings split across both, so mixing the preference is deliberate --
# do NOT make all five fluid-sensitive. `MODEL.md` §2.
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


def pick_depths(n_available: int, n_want: int) -> np.ndarray:
    """Evenly across 6-94% of the stack. NOT the middle -- this is D2's fix.

    Returns `n_want` indices, with repeats when the series is shorter than the slot asks for.
    Repeats are deliberate: the slot's shape is fixed so the batch is rectangular, and a repeated
    slice is honest about there being no more anatomy to show. Dropping to a shorter slot instead
    would make the window count vary per study and break the fixed `(slot, position)` embedding.
    """
    if n_available <= 0:
        return np.zeros(n_want, dtype=int)
    if n_available == 1:
        return np.zeros(n_want, dtype=int)
    lo = DEPTH_LO * (n_available - 1)
    hi = DEPTH_HI * (n_available - 1)
    return np.clip(np.round(np.linspace(lo, hi, n_want)), 0, n_available - 1).astype(int)


def source_step(n_available: int, n_want: int) -> float:
    """Mean distance, in SOURCE slices, between consecutive sampled slices of one series.

    What a window's three channels span: ~1 on a stack about 1.1x the slot's slice count (true
    neighbours), 2-3 on a deep one, below 1 on a short one (channels repeat). It varies per series,
    so the Stage A builder records it per (study, slot) beside `n_available` -- decided 2026-09-23,
    when the spec was found to describe neighbours while the code sampled (AUDIT.md).
    """
    d = pick_depths(n_available, n_want)
    return float(np.diff(d).mean()) if len(d) > 1 else 0.0


def window_ids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed (slot_id, position_id, centre-slice index) for all 54 windows, same for every study.

    `centre` indexes the 64-slice cache row. The head's `(slot, position)` embedding is looked up
    by these, so they must not depend on the study -- a study-dependent layout would mean the
    embedding indexes different anatomy for different patients, which is §9h's wrong-table bug
    running perfectly and scoring wrongly.
    """
    slot_id, pos_id, centre = [], [], []
    base = 0
    for si, s in enumerate(SLOTS):
        for p in range(s.n_windows):
            slot_id.append(si)
            pos_id.append(p)
            centre.append(base + p + 1)      # centre of the triple; +1 skips the dropped edge
        base += s.n_slices
    return (np.asarray(slot_id, dtype=np.int64),
            np.asarray(pos_id, dtype=np.int64),
            np.asarray(centre, dtype=np.int64))


def slot_offsets() -> list[tuple[int, int]]:
    """(start, stop) of each slot inside the 64-slice cache row."""
    out, base = [], 0
    for s in SLOTS:
        out.append((base, base + s.n_slices))
        base += s.n_slices
    return out


def assign_series(series: list[dict]) -> dict[str, dict | None]:
    """Map each slot to a series, or None when the study has nothing suitable.

    `series` entries need `plane` ('Axial'|'Coronal'|'Sagittal'), `fluid` (0/1) and `n_slices`.
    A slot with no candidate returns None -> the cache writes zeros and clears the mask bit for
    every window in that slot. NEVER substitute a different plane: §9h's rule is that feeding a
    slot its neighbour runs perfectly and scores wrongly.
    """
    out: dict[str, dict | None] = {}
    used: set[int] = set()
    for s in SLOTS:
        cands = [(i, c) for i, c in enumerate(series)
                 if c.get("plane") == s.plane and i not in used]
        if not cands:
            out[s.name] = None
            continue
        # Preference is a sort key, not a filter: a study with only non-fluid sagittals should
        # still fill sag_1 rather than leave 18 slices of zeros. Ties break on slice count, since
        # a longer stack samples 6-94% more finely.
        def key(ic):
            i, c = ic
            miss = 0 if s.fluid is None else int(c.get("fluid", 0) != s.fluid)
            return (miss, -int(c.get("n_slices", 0)))
        cands.sort(key=key)
        # The best REMAINING candidate. A slot never needs a rank of its own: `used` already
        # removed every series an earlier slot took, so the head of this list IS "next best".
        # A `rank` field used to index into it as well, which skipped one series twice -- with
        # coronals of 30, 28 and 12 slices, cor_2 got the 12 (AUDIT.md, fixed 2026-09-23).
        pick = cands[0]
        used.add(pick[0])
        out[s.name] = pick[1]
    return out


def manifest() -> dict:
    """Stamped into every cache. `PREPROCESS_VERSION`'s purpose is to make drift loud."""
    return {"spec": "c6-64slice", "fov_mm": FOV_MM, "target_mm": TARGET_MM, "img": IMG,
            "depth_range": [DEPTH_LO, DEPTH_HI], "group": GROUP,
            "n_slices": N_SLICES, "n_windows": N_WINDOWS,
            # MODEL.md §3: consecutive SAMPLED slices, the fork's convention, not stack neighbours.
            # The builder must add `n_available` and `source_step` per (study, slot).
            "window": "consecutive sampled slices",
            "per_series": ["n_available", "source_step"],
            "slots": [asdict(s) for s in SLOTS]}


def _self_test() -> None:
    assert N_SLICES == 64, N_SLICES
    assert N_WINDOWS == 54, N_WINDOWS

    sid, pid, ctr = window_ids()
    assert len(sid) == len(pid) == len(ctr) == 54
    assert ctr.max() < N_SLICES, "a window centre points past the cache row"
    # Every centre must sit strictly inside its own slot, or the triple straddles two slots and
    # silently mixes planes -- the §9h failure this layout exists to prevent.
    for (a, b), s in zip(slot_offsets(), SLOTS):
        own = ctr[(ctr >= a) & (ctr < b)]
        assert len(own) == s.n_windows, (s.name, len(own), s.n_windows)
        assert own.min() > a and own.max() < b - 1, f"{s.name}: window triple leaves the slot"

    # Depth sampling reaches the outer thirds -- the entire point of D2.
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
    # Three coronals: cor_1 takes the fluid one, cor_2 the LONGER of the other two. The old
    # `rank` indexing handed cor_2 the 12-slice stack here; two coronals could never show it.
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
