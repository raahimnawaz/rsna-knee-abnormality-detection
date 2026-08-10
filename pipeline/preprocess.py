"""The single definition of how a series becomes model input. Imported by both sides.

Two programs must agree byte-for-byte on this: `kaggle_02_dinov2_cache.py`, which builds the
training features, and the submission notebook, which rebuilds them for the test set. If they
drift, nothing raises -- the model simply receives inputs from a distribution it never saw and
scores badly for reasons no traceback will show. That failure has no local symptom and would
surface as an unexplained CV/LB gap, so the defence is structural:

  1. ONE file. Neither program defines a constant of its own.
  2. FINGERPRINT. Every constant that changes a feature value is hashed into PREPROCESS_VERSION.
     The cache manifest records it; the submission notebook asserts on it. Paste a stale copy of
     this file into a Kaggle notebook and the assert fires instead of the model silently rotting.

Kaggle Script notebooks are single files, so to import this there either attach the repo as a
Dataset and `sys.path.insert(0, "/kaggle/input/<ds>/pipeline")`, or add it as a Kaggle Utility
Script. Both keep one source of truth; pasting a copy does not, which is what (2) is for.

pydicom is imported lazily inside the DICOM readers so this module stays importable on the Mac,
where the images never land and pydicom is deliberately not a dependency.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- constants
# Changing ANY of these invalidates an existing feature cache. See PREPROCESS_VERSION.
MODEL = os.environ.get("MODEL", "vit_base_patch14_reg4_dinov2.lvd142m")   # registers variant
# 518 -> 37x37 patches at patch-14. A meniscal tear is small and 224 loses it, but 518 costs
# ~5x, so the honest order is a 224 pass to prove the pipeline end-to-end on real features and
# then 518 once it is worth the GPU quota. Overridable by env so ONE code dataset serves both
# passes -- and because IMG_SIZE feeds the fingerprint, the two caches can never be confused.
IMG_SIZE = int(os.environ.get("IMG_SIZE", 518))
SLICES_PER_SERIES = int(os.environ.get("SLICES_PER_SERIES", 32))   # train on 24; see below
TARGET_MM = float(os.environ.get("TARGET_MM", 0.35))   # in-plane resample target, mm/px
FOV_MM = float(os.environ.get("FOV_MM", 160.0))        # centre crop/pad field of view
EMBED_DIM = 1536        # CLS(768) || patch-mean(768) -- what embed() concatenates

# Slices used per series at TRAIN time, out of the SLICES_PER_SERIES cached. The gap is the
# only pixel-space augmentation that survives a frozen backbone: everything else in PLAN 3.3
# (affine, gamma, bias field, Rician noise) is applied to pixels the backbone has already
# consumed. Caching 32 and sampling 24 buys slice jitter back for 33% more storage.
SLICES_PER_SERIES_TRAIN = 24

PLANE_ID = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
# Fluid_Sensitive and Fat_Suppression are perfectly redundant (FINDINGS.md 3.1), so one flag
# is the whole story -> 6 series types, not 12.
N_SERIES_TYPES = len(PLANE_ID) * 2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Slices per forward pass. Not a fingerprinted constant -- batching changes throughput, not
# feature values -- but shared so the cache build and the submission notebook are tuned once.
# 518px at patch-14 is ~1,374 tokens per slice, so this is bounded by activation memory.
BATCH_HINT = 16

# Every study is canonicalised to this handedness. 'Medial' is on opposite sides of the image
# for a left vs a right knee, so four of the twelve labels are noise without this (PLAN 3.2).
CANONICAL_SIDE = "R"

# Reverse the SLICE axis of sagittal left knees, where medial/lateral is the slice axis rather
# than the image x-axis. See canonicalise(). OFF until the K16 direction bit exists.
#
# One switch consulted by BOTH readers, rather than a per-call argument, because the correction
# is only meaningful if train and test apply it together. load_series knows its direction (it
# just sorted by IPP) while load_series_nifti does not, so a per-call decision would silently
# canonicalise the test set and not the training cache -- the preprocessing-parity failure with
# no symptom that PREPROCESS_VERSION exists to catch, arriving through the one door the
# fingerprint could not see (the NIfTI conversion sits upstream of it).
#
# Turning this on is PLAN 9 Phase 0 step 3: it needs the per-series direction bit from step 2, and
# it invalidates the 224 cache by design.
SAGITTAL_LR_SLICE_FLIP = bool(int(os.environ.get("SAGITTAL_LR", "0")))

# Boundary on the median ImagePositionPatient x-coordinate below which a knee is the RIGHT one.
#
# NOT zero, which is the obvious guess and is wrong. Measured over the 2,203 studies that carry
# a real Laterality tag (kaggle_01b): R knees sit at x median -150, L knees at +14, and the
# disagreements at a zero boundary all cluster at |x| ~ 10. The scanned knee is placed at
# isocentre rather than the patient being centred, so the frame is offset and the two modes
# straddle roughly -62 instead of 0.
#
#   threshold 0    -> 89.3% agreement with the tag
#   threshold -62  -> 97.7%; cross-validated (fit 80% / test 20%, x20) 97.32% +- 0.72%,
#                     threshold itself stable at -62.4 +- 4.5
#
# It transfers to the untagged half: the two halves have near-identical x distributions
# (below -62 / between / above: 51.2/8.9/40.0 tagged vs 53.8/9.9/36.3 untagged).
LATERALITY_X_THRESHOLD = -62.0


def _fingerprint() -> str:
    """Hash of everything that changes a cached feature value."""
    keys = {
        "model": MODEL, "img_size": IMG_SIZE, "slices": SLICES_PER_SERIES,
        "target_mm": TARGET_MM, "fov_mm": FOV_MM, "embed_dim": EMBED_DIM,
        "canonical_side": CANONICAL_SIDE, "norm": "pct_0.5_99.5", "stack": "2.5d_prev_cur_next",
        "lat_x_threshold": LATERALITY_X_THRESHOLD,
    }
    # Added only when ENABLED, so that switching it on invalidates every existing cache while
    # leaving it off is a genuine no-op. The alternative -- always hashing the flag -- would
    # renumber the fingerprint of the 224 cache without changing a single feature value, and
    # force a 2.5 h rebuild to reproduce bytes it already holds.
    if SAGITTAL_LR_SLICE_FLIP:
        keys["canon_sagittal_lr"] = True
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:12]


PREPROCESS_VERSION = _fingerprint()


# --------------------------------------------------------------------------- DICOM reading
# Decode is the bottleneck for both the cache build and the submission notebook -- PLAN.md
# 6.3.1 ranks decode throughput as the single largest efficiency lever, at zero AUC cost,
# because the corpus mixes uncompressed / JPEG Lossless / JPEG 2000 / Implicit VR and JPEG 2000
# is slow. `dicomsdl` is typically 3-5x pydicom on JPEG 2000; SimpleITK and NVIDIA DALI are the
# other candidates. But this is a MEASUREMENT, not a belief: kaggle_01's benchmark reports
# ms/slice per transfer syntax per backend, and that decides it.
#
# Until then the backend stays pluggable, defaulting to whatever is present. pydicom is
# preinstalled on the Kaggle image; dicomsdl is not, and the submission notebook has NO
# INTERNET -- so choosing it means shipping a pinned wheel in an attached Dataset. Do not adopt
# it on reputation; adopt it if kaggle_01 says it is worth the packaging.
DECODE_BACKEND = "auto"     # "auto" | "pydicom" | "dicomsdl"


def _backend() -> str:
    if DECODE_BACKEND != "auto":
        return DECODE_BACKEND
    try:
        import dicomsdl  # noqa: F401
        return "dicomsdl"
    except ImportError:
        return "pydicom"


def read_pixels(path) -> np.ndarray | None:
    """Decode one slice to a 2-D float32 array, via whichever backend is selected."""
    try:
        if _backend() == "dicomsdl":
            import dicomsdl
            return np.asarray(dicomsdl.open(str(path)).pixelData(), dtype=np.float32)
        import pydicom
        return pydicom.dcmread(path, force=True).pixel_array.astype(np.float32)
    except Exception:
        return None


CUDA_MIN_CAPABILITY = (7, 0)    # what Kaggle's current PyTorch reports as its floor


def _parse_arch(entry: str) -> tuple[int, int] | None:
    """'sm_75' -> (7, 5), 'sm_90a' -> (9, 0), 'sm_100' -> (10, 0). Anything else -> None.

    The last digit is the minor version, whatever is left is the major -- which is why this is
    not a two-character slice: Blackwell's sm_100 is compute 10.0, not 1.0.
    """
    if not entry.startswith("sm_"):
        return None                             # 'compute_90' (PTX) is not a binary arch
    digits = "".join(c for c in entry[3:] if c.isdigit())
    if len(digits) < 2:
        return None
    return int(digits[:-1]), int(digits[-1])


def pick_device(allow_mps: bool = False) -> str:
    """The ONE device picker. Returns 'cuda', 'mps' (only if allow_mps), 'cpu' or 'unusable'.

    'unusable' means a GPU is present but this PyTorch build has no kernels for it. That is a
    RETRY, not a fallback, and callers must not silently treat it as 'cpu' -- see kaggle_02,
    which exits, and kaggle_03, which writes a placeholder submission before degrading.

    `torch.cuda.is_available()` is TRUE on a GPU whose compute capability the installed wheel
    has no kernels for, and the failure then arrives much later as a bare
    `CUDA error: no kernel image is available for execution on the device`.

    Measured 2026-08-08: Kaggle hands out **Tesla P100 (capability 6.0)** alongside T4s, and its
    current PyTorch supports 7.0-12.0. A P100 session therefore cannot run this at all. The
    assignment is a coin flip and the `accelerator` field in kernel-metadata.json does not
    reliably override it, so the only defence is to detect it and say so in seconds rather than
    after the notebook has queued, downloaded weights, and started decoding.

    allow_mps exists for fusion/train.py, which is the only caller that can use Apple silicon.
    It is a parameter rather than a second function because two device pickers that disagree is
    exactly how fusion/train.py kept the unguarded is_available() this one was written to
    replace.
    """
    import torch
    if allow_mps and torch.backends.mps.is_available():
        return "mps"
    if not torch.cuda.is_available():
        return "cpu"

    # Capability check FIRST, and it fails closed. Two earlier versions of this guard tried to
    # infer support and both fell open on a P100 -- one swallowed an exception and returned
    # "cuda", the other's probe did not raise. Kaggle's PyTorch reports a 7.0 minimum, so
    # anything below that is unusable, full stop, with no inference and no fallthrough.
    #
    # The two queries get their own try blocks on purpose: sharing one meant a transient failure
    # of the NAME lookup zeroed a capability that had already been read successfully, and a
    # perfectly good T4 was then condemned as "compute 0.0".
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        major, minor = 0, 0
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "unknown GPU"
    print(f"GPU: {name}, compute {major}.{minor}")
    if (major, minor) < CUDA_MIN_CAPABILITY:
        print(f"WARNING: {name} (compute {major}.{minor}) is below the "
              f"{CUDA_MIN_CAPABILITY[0]}.{CUDA_MIN_CAPABILITY[1]} minimum this PyTorch "
              f"supports. Kaggle still assigns P100s and the accelerator field in "
              f"kernel-metadata.json does not override the draw. Re-run for a different GPU.")
        return "unusable"

    # The floor above only catches a card too OLD for the *Kaggle* wheel. get_arch_list() is
    # what the installed wheel was actually compiled for, so it catches the same failure on any
    # machine (a Maxwell sm_52 box, say) and it is the only thing that can see a card too NEW.
    #
    # Below the compiled minimum there is no escape hatch -- fail closed. ABOVE it, CUDA can JIT
    # the highest embedded PTX forward, so sm_86 on an sm_80 wheel is fine and hard-failing that
    # would strand working sessions. Warn and let the probe below decide.
    try:
        archs = sorted(filter(None, (_parse_arch(a) for a in torch.cuda.get_arch_list())))
    except Exception:
        archs = []                              # a source build can report nothing; don't condemn
    if archs and (major, minor) not in archs:
        if (major, minor) < min(archs):
            print(f"WARNING: {name} (compute {major}.{minor}) is below sm_{min(archs)[0]}"
                  f"{min(archs)[1]}, the oldest architecture this PyTorch was compiled for "
                  f"({torch.cuda.get_arch_list()}). There is no forward-JIT path downwards, so "
                  f"this would die with 'no kernel image is available for execution on the "
                  f"device'. Use a newer GPU or a PyTorch build that targets this one.")
            return "unusable"
        print(f"NOTE: {name} (compute {major}.{minor}) is newer than anything this PyTorch was "
              f"compiled for ({torch.cuda.get_arch_list()}); it will run via PTX JIT if the "
              f"wheel embeds PTX. First launch may be slow.")

    try:
        torch.zeros(8, 8, device="cuda").sum().item()
        torch.cuda.synchronize()    # kernel-launch errors are reported ASYNCHRONOUSLY
        return "cuda"
    except Exception as e:
        # name/major/minor are already in scope and already have fallbacks. Re-querying CUDA
        # after a failed launch risks a second exception on a poisoned context for no gain.
        detail = (str(e).splitlines() or [""])[0]       # str(SomeError()) is "" -> [] , not [""]
        print(f"WARNING: {name} (compute {major}.{minor}) cannot run this PyTorch build -- "
              f"{type(e).__name__}: {detail}\nKaggle still assigns Tesla P100s (compute 6.0) "
              f"while its PyTorch requires >= 7.0, and the accelerator field in "
              f"kernel-metadata.json does not reliably override the draw. Re-run to get a "
              f"different GPU; a T4 works.")
        return "unusable"


def find_competition_root(hint: str = "knee"):
    """Locate the mounted competition data. Do NOT assume it is a direct child of /kaggle/input.

    Measured 2026-08-07: it mounts at /kaggle/input/competitions/rsna-knee-abnormality-detection,
    one level deeper than the obvious guess. A find_root that only scans direct children exits
    with 'competition data not found' -- after the notebook has already queued for a GPU. So
    search two levels and require a marker file rather than trusting the directory name.
    """
    from pathlib import Path as _P
    base = _P("/kaggle/input")
    if not base.exists():
        raise SystemExit("no /kaggle/input -- this runs on Kaggle, not locally")

    def looks_right(d) -> bool:
        return (d / "train_series.csv").exists() or (d / "test_series.csv").exists()

    cands = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        cands.append(p)
        try:
            cands.extend(sorted(c for c in p.iterdir() if c.is_dir()))
        except OSError:
            pass
    for d in cands:
        if looks_right(d) and hint in d.name.lower():
            return d
    for d in cands:                     # marker file beats the name hint
        if looks_right(d):
            return d
    raise SystemExit(f"competition data not found under {base}; looked in "
                     f"{[str(c) for c in cands[:8]]}")


def build_study_index(root) -> dict:
    """StudyInstanceUID -> directory, built in ONE filesystem pass.

    The obvious `next(d for d in root.rglob(uid) if d.is_dir())` inside a per-study loop walks
    the entire tree once per study: O(n_studies x n_files) over 4,407 studies and 819,640 files
    spanning 570 GB. On Kaggle's mounted dataset that alone can outlast the session.

    The layout is <split>_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm, so listing one
    level under each top-level directory builds the whole map. Falls back to rglob only if that
    finds nothing, so a layout change degrades to slow rather than to broken.
    """
    from pathlib import Path as _P
    root = _P(root)
    idx: dict = {}
    for parent in sorted(root.iterdir()):
        if not parent.is_dir():
            continue
        try:
            for d in parent.iterdir():
                if d.is_dir():
                    idx.setdefault(d.name, d)
        except OSError:
            continue
    if not idx:
        for d in root.rglob("*"):
            if d.is_dir():
                idx.setdefault(d.name, d)
    return idx


def read_headers(paths):
    """-> (headers, kept_paths). Unreadable files are dropped rather than failing the series.

    Always pydicom: header reads are cheap with stop_before_pixels=True, and pydicom's
    attribute access is what slice_order/read_laterality are written against. The backend
    choice only matters for pixel decode, which is where the time actually goes.
    """
    import pydicom
    headers, keep = [], []
    for p in paths:
        try:
            headers.append(pydicom.dcmread(p, stop_before_pixels=True, force=True))
            keep.append(p)
        except Exception:
            pass
    return headers, keep


def slice_order(headers) -> list[int]:
    """Sort indices along the true through-plane axis (PLAN.md 3.1 step 1).

    The slice normal is the cross product of the row and column direction cosines from
    ImageOrientationPatient; projecting ImagePositionPatient onto it gives a real spatial
    coordinate. InstanceNumber is NOT reliably spatial, and a mis-ordered volume silently
    destroys the slice transformer's positional signal. Falls back to it only when the
    geometry is missing.
    """
    keys = []
    for i, ds in enumerate(headers):
        try:
            iop = np.asarray(ds.ImageOrientationPatient, float)
            ipp = np.asarray(ds.ImagePositionPatient, float)
            normal = np.cross(iop[:3], iop[3:])
            keys.append((float(np.dot(ipp, normal)), i))
        except Exception:
            keys.append((float(getattr(ds, "InstanceNumber", i) or i), i))
    return [i for _, i in sorted(keys)]


def read_laterality(headers) -> tuple[str | None, str]:
    """-> ('L' | 'R' | None, source) where source is 'tag' | 'geometry' | 'none'.

    The tag is only half the story. kaggle_01b measured (0020,0060) Laterality present on
    2,203 of 4,407 studies -- the rest carry the tag with an EMPTY value, which must not be
    read as a side. Leaving that half uncanonicalised would make Medial/Lateral Meniscus and
    Medial/Lateral OA noise for half the corpus (PLAN.md 3.2), which is worse than a fallback
    that is right ~97% of the time.

    So: tag first, geometry second. The source is returned and stored per series in the cache,
    so the geometry-derived subset stays identifiable and the decision can be re-measured
    against the model later rather than being baked in invisibly.
    """
    votes = []
    for ds in headers:
        v = getattr(ds, "Laterality", None) or getattr(ds, "ImageLaterality", None)
        if not v:
            bp = str(getattr(ds, "BodyPartExamined", "") or "").upper()
            v = ("L" if bp.endswith("L") and "KNEE" in bp
                 else ("R" if bp.endswith("R") and "KNEE" in bp else None))
        v = str(v).strip().upper()[:1] if v else None
        if v in ("L", "R"):
            votes.append(v)
    if votes:
        return max(set(votes), key=votes.count), "tag"

    xs = []
    for ds in headers:
        try:
            xs.append(float(np.asarray(ds.ImagePositionPatient, float)[0]))
        except Exception:
            pass
    if xs:
        return ("R" if float(np.median(xs)) < LATERALITY_X_THRESHOLD else "L"), "geometry"
    return None, "none"


# --------------------------------------------------------------------------- volume ops
def canonicalise(vol: np.ndarray, laterality: str | None, plane: str | None,
                 slice_direction: str | None = None) -> np.ndarray:
    """Mirror left knees onto CANONICAL_SIDE so 'medial' is always the same place.

    `vol` is [S,H,W]. Medial/lateral lives on a DIFFERENT AXIS depending on the plane, and this
    function has to handle both:

      Axial, Coronal   medial/lateral is the image x-axis  -> mirror axis 2 (in-plane)
      Sagittal         medial/lateral is the SLICE axis    -> reverse axis 0

    The sagittal case was missed until 2026-08-09. The previous docstring argued that "flipping a
    sagittal series would mirror the knee front-to-back for no gain", which is true of the
    in-plane axis -- sagittal's image x-axis IS anterior-posterior -- and simply never considered
    the slice axis. But `spatial_order` sorts every series ascending along the slice normal, and
    for sagittal that normal is the patient's left-right axis, on which medial is +x for a right
    knee and -x for a left one. So the same sort yields lateral->medial for one knee and
    medial->lateral for the other, and `FusionHead.slice_pos` is a LEARNED per-index embedding
    (fusion/model.py:77), so slice 5 meant lateral in one study and medial in the next.

    Measured exposure: sagittal is the largest plane at 9,864 of 24,371 series (40.5%), and 1,894
    of 4,407 studies resolve to left knees (43.0%, tag first then geometry). Four of the twelve
    labels are medial/lateral pairs.

    `slice_direction` is how this composes with K16 (a third of NIfTI series are stored
    back-to-front and the file cannot say which). The sagittal correction is defined relative to
    ascending spatial order, so it must be applied to a volume KNOWN to be in that order:

      'forward'   already ascending -- what spatial_order guarantees on the DICOM path
      'reversed'  restore ascending first, then correct
      None        unknown; the slice axis is left alone entirely, including for sagittal

    None is honest rather than safe, matching how unknown laterality is handled: with the K16 bit
    missing, a sagittal reversal would land on a substrate that is itself reversed for ~62% of
    sagittal series (8/21 forward, README 1) and would be wrong exactly where it fired. Pass the
    bit from kaggle_01c once it exists (PLAN 9 Phase 0 step 2) and both corrections apply together.
    """
    flip_slices = slice_direction == "reversed"

    left = laterality is not None and laterality != CANONICAL_SIDE
    if SAGITTAL_LR_SLICE_FLIP and left and plane == "Sagittal" and slice_direction is not None:
        # XOR: restoring ascending order and then reversing for handedness cancel out.
        flip_slices = not flip_slices

    if flip_slices:
        vol = vol[::-1]

    if left and plane in ("Axial", "Coronal"):
        vol = vol[:, :, ::-1]

    return np.ascontiguousarray(vol)


def center_fit(v: torch.Tensor, side: int) -> torch.Tensor:
    """Centre crop or zero-pad [1,S,H,W] to side x side. The knee is protocol-centred."""
    _, _, h, w = v.shape
    if h > side:
        t = (h - side) // 2
        v = v[:, :, t:t + side, :]
    if w > side:
        l = (w - side) // 2
        v = v[:, :, :, l:l + side]
    _, _, h, w = v.shape
    if h < side or w < side:
        v = F.pad(v, (0, max(0, side - w), 0, max(0, side - h)))
    return v


def normalise_and_resample(vol: np.ndarray, spacing: float | None) -> np.ndarray:
    """[S,H,W] raw -> [S,side,side] float32 in [0,1]. PLAN.md 3.1 steps 3-4.

    MRI has no HU standard, so intensity is per-volume robust percentiles rather than a fixed
    window.
    """
    v = torch.from_numpy(np.ascontiguousarray(vol)).float()[None]      # [1,S,H,W]
    # np.percentile, NOT torch.quantile. torch.quantile raises
    #   RuntimeError: quantile() input tensor is too large
    # above 2**24 (16,777,216) elements, and a 32-slice series at 768x768 is 18,874,368. The
    # corpus contains 768x768 series, so this is not a corner case -- it is every large series.
    #
    # Found 2026-08-09 on the first real run of the local builder. It sits in the SHARED path,
    # so kaggle_02 would have hit it too, mid-build, after hours of GPU. The five failed cache
    # attempts died on the GPU lottery and on mount latency before ever reaching a series this
    # big, which is the only reason it had not surfaced.
    #
    # Same statistic, same interpolation (both linear by default), no size ceiling. The
    # "pct_0.5_99.5" in PREPROCESS_VERSION still describes it exactly -- and note the
    # fingerprint hashes that DESCRIPTION, not this implementation, so it could not have caught
    # a change here. Nothing had been cached yet, so there is no old cache to invalidate.
    lo, hi = np.percentile(np.asarray(vol, dtype=np.float32), [0.5, 99.5])
    v = ((v - float(lo)) / (float(hi) - float(lo) + 1e-6)).clamp(0, 1)
    if spacing and spacing > 0:
        scale = spacing / TARGET_MM
        if abs(scale - 1) > 0.02:
            v = F.interpolate(v, scale_factor=scale, mode="bilinear", align_corners=False)
    return center_fit(v, int(round(FOV_MM / TARGET_MM)))[0].numpy()


def pick_slices(n_available: int, n_want: int = SLICES_PER_SERIES) -> np.ndarray:
    """Even spread across the volume. A knee series is protocol-centred, so the middle is signal."""
    return np.unique(np.linspace(0, max(n_available - 1, 0), n_want).round().astype(int))


def to_25d(vol: np.ndarray) -> torch.Tensor:
    """[S,H,W] -> [S,3,H,W]: three adjacent slices as RGB, matching DINOv2's 3-channel stem.

    PLAN 3.3 says 'groups of 3-5 adjacent slices'; the ViT stem takes exactly 3, so 3 it is.
    """
    t = torch.from_numpy(np.ascontiguousarray(vol)).float()
    prev = torch.cat([t[:1], t[:-1]])
    nxt = torch.cat([t[1:], t[-1:]])
    return torch.stack([prev, t, nxt], dim=1)


def imagenet_normalise(b: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] in [0,1] -> resized to IMG_SIZE and standardised for the DINOv2 stem."""
    b = F.interpolate(b, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return (b - IMAGENET_MEAN.to(b.device)) / IMAGENET_STD.to(b.device)


@torch.no_grad()
def embed(model, x: torch.Tensor, dev: str, batch: int = BATCH_HINT) -> np.ndarray:
    """[S,3,H,W] -> [S, 2D] fp16: CLS concatenated with the patch mean.

    Both halves earn their place: CLS carries the global impression, the patch mean retains
    localised signal that a single token averages away -- and the findings here are small.

    Lives here rather than in a builder because it decides FEATURE VALUES, which makes it
    preprocessing by this file's own definition. It is called by the Kaggle cache builder, the
    local NIfTI builder and the submission notebook; three copies of it would be three ways for
    the train and test distributions to drift apart, which is the failure the whole
    PREPROCESS_VERSION mechanism exists to prevent and the one it could not see.

    autocast is CUDA-only on purpose. MPS fp16 reductions are uneven, and the cache is stored in
    fp16 either way -- the accumulation should not also be.

    THE DECORATOR IS LOAD-BEARING. When this was lifted out of kaggle_02 the @torch.no_grad()
    stayed behind on build_cache, and kaggle_02 only calls .eval() -- it never sets
    requires_grad_(False). Its K14 smoke probe then hit `Can't call numpy() on Tensor that
    requires grad`, after the GPU draw and the weights download. Neither self-test caught it
    because both inject a fake embed_fn and never build the real backbone -- which is K13's
    lesson arriving for the third time.
    """
    out = []
    for i in range(0, len(x), batch):
        b = imagenet_normalise(x[i:i + batch].to(dev))
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            tok = model.forward_features(b)
        cls, patches = tok[:, 0], tok[:, model.num_prefix_tokens:].mean(1)
        out.append(torch.cat([cls, patches], -1).float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def load_series(paths, plane: str | None = None):
    """DICOM paths -> ([S,side,side] float32 in [0,1], laterality, laterality_source).

    The whole of PLAN.md 3.1 in one call, so the cache builder and the submission notebook
    cannot diverge on any step of it.
    """
    headers, keep = read_headers(paths)
    if len(keep) < 3:
        return None, None, "none"

    lat, lat_src = read_laterality(headers)
    order = slice_order(headers)
    idx = [order[i] for i in pick_slices(len(order))]

    try:
        spacing = float(np.asarray(headers[idx[0]].PixelSpacing, float)[0])
    except Exception:
        spacing = None

    # Decode ONLY the slices that are kept -- never touch pixel data twice (PLAN.md 6.3.1).
    vol = [a for a in (read_pixels(keep[i]) for i in idx) if a is not None]
    if len(vol) < 3:
        return None, lat, lat_src

    # Mixed in-series geometry happens; keep the dominant shape rather than dropping the series.
    shape = max({a.shape for a in vol}, key=lambda s: s[0] * s[1])
    vol = [a for a in vol if a.shape == shape]
    if len(vol) < 3:
        return None, lat, lat_src

    v = normalise_and_resample(np.stack(vol), spacing)
    # 'forward' is a fact here, not an assumption: idx came from spatial_order, which sorts
    # ascending by IPP projection onto the slice normal a few lines above.
    return canonicalise(v, lat, plane, slice_direction="forward"), lat, lat_src


# --------------------------------------------------------------------------- NIfTI source
#
# TRAIN ONLY, AND IT SITS UPSTREAM OF THE FINGERPRINT. Read this before trusting a feature
# built through it.
#
# `davidadekanmi/rsna-knee-nifti-part1..12` is the competition corpus repackaged as one NIfTI
# per series (PLAN.md 9.1). It exists because the DICOM path's real cost is not decode, it is
# ~19 ms to open one of ~700k small files on a network mount; one file per series is 24,371
# opens off a local SSD instead, which removes the wall rather than optimising it.
#
# The danger is specific and it is not "third-party data might be wrong". It is that the
# conversion happened BEFORE `PREPROCESS_VERSION` is computed, so the fingerprint -- the whole
# mechanism this file exists to provide -- structurally cannot see it. Train features would come
# through here and test features through `load_series()`, and if the two disagree on slice order
# or in-plane orientation the model is fed a distribution it never trained on and simply scores
# badly. No traceback. That is the exact failure README "Preprocessing parity" is about, one
# layer further up than the fingerprint can reach.
#
# So every convention below is NAMED and MEASURED by `pipeline/validate_nifti.py`, not assumed.

# MEASURED 2026-08-09, and it corrects PLAN.md 9.1. That section recorded "sform_code=2 with a
# populated affine ... real affine". The sform_code is indeed 2, and the affine is NOT real:
#
#     [[0.33 0    0    0]
#      [0    0.33 0    0]      <- diagonal spacing, identity rotation, ZERO translation
#      [0    0    3.4  0]
#      [0    0    0    1]]
#
# The converter wrote voxel spacing into the header and discarded the patient coordinate system.
# ImagePositionPatient and ImageOrientationPatient are both gone. Consequences:
#
#   - THE GEOMETRY LATERALITY FALLBACK CANNOT RUN FROM THESE FILES. It needs IPP[0], and there
#     is no position. That fallback is what canonicalises the 2,204 studies whose (0020,0060)
#     tag is empty -- half the corpus -- and without canonicalisation Medial/Lateral Meniscus
#     and Medial/Lateral OA are noise (PLAN.md 3.2). Four of twelve labels.
#   - Plane cannot be derived either. Recoverable: train_series.csv carries Anatomical_Plane.
#   - Slice direction relative to the DICOM normal is unknowable from the header.
#
# The fix is that none of this has to come from the NIfTI. `kaggle_01b` already resolved
# laterality for all 4,407 studies from the DICOM headers and wrote it to data/study_meta.csv --
# `laterality_tag` for the tagged half, `x_median` for every study. So laterality is PASSED IN
# here from that table rather than re-derived, which is strictly better than what the DICOM path
# does at run time: same answer, already measured, no per-series recomputation.
#
# In-plane orientation and slice direction could not be settled from headers at all once the
# affine turned out to carry no direction cosines. `kaggle_01c` thumbnails settled both in
# pixels instead: layout `as-is` at r = 1.0000 against the DICOM middle slice (runner-up 0.65),
# and slice order 100% forward. r = 1.0 means the pixel data is identical, so this is a faithful
# repackaging and the transpose below is correct as written -- FOR AXIAL_0. All 60 thumbnailed
# series turned out to be that one type, because kaggle_01c v2 took the head of a group-ordered
# frame. Coronal and Sagittal are unvalidated until kaggle_01c is re-run with the shuffle.
#
# There is deliberately NO module-level "the affine is empty" constant. `nifti_geometry()`
# reports has_position / has_orientation per file, because a later part of the dataset could be
# converted by a different pass and a global flag would hide that.


NIFTI_SLICE_AXIS = 2        # the converter writes (rows, cols, slices); measured, not assumed


def _nifti_axes(aff: np.ndarray, shape: tuple) -> tuple[int, int, int]:
    """-> (slice_axis, row_axis, col_axis). Fixed by the converter's layout, NOT by spacing.

    An earlier version picked the slice axis as argmax(voxel spacing), justified in a docstring
    as "~0.33 mm in-plane against ~3.4 mm through-plane, a 10x separation, so this is
    unambiguous". **Measured over all 19,859 downloaded series, that is false**: the ratio runs
    down to 1.005, 84 series are under 1.5x, and 27 series acquired near-isotropically at
    (0.625, 0.625, 0.60) put the LARGEST spacing in-plane. argmax then returned axis 0 and
    reformatted a 160-slice axial acquisition into 256 sagittal slices -- silently, and only on
    those 27, so no aggregate check would show it.

    The layout is a property of the converter, not of the voxel sizes: every file in this corpus
    is (rows, cols, slices). Assert it rather than infer it, so a differently-converted part
    fails loudly instead of being reformatted.
    """
    sa = NIFTI_SLICE_AXIS
    return sa, 1, 0


def nifti_geometry(path) -> dict | None:
    """Header-only read: shape, spacing, and the per-slice x in DICOM LPS. No pixel data.

    Separate from load_series_nifti because validate_nifti.py needs this over thousands of
    series and must not pay to decode any of them.
    """
    import nibabel as nib
    try:
        img = nib.load(str(path))
    except Exception:
        return None
    aff = np.asarray(img.affine, float)
    shape = tuple(int(s) for s in img.shape[:3])
    if len(shape) != 3 or min(shape) < 3:
        return None
    sa, ra, ca = _nifti_axes(aff, shape)
    spacing = np.linalg.norm(aff[:3, :3], axis=0)
    # The slice axis is asserted, not inferred (see _nifti_axes). A file whose slice axis is not
    # the third is from a different conversion pass and must not be silently reformatted.
    layout_ok = bool(shape[sa] <= min(shape[ra], shape[ca]) or spacing[sa] >= spacing[ra])

    # Is there a patient coordinate system in here at all, or only spacing? A zero translation
    # with a diagonal 3x3 is not "the scanner frame with the origin at isocentre" -- it is the
    # frame having been dropped. Reported per file rather than assumed, because a later part of
    # the dataset could have been converted by a different pass.
    has_position = bool(np.abs(aff[:3, 3]).max() > 1e-6)
    off_diag = aff[:3, :3] - np.diag(np.diag(aff[:3, :3]))
    has_orientation = bool(np.abs(off_diag).max() > 1e-6)

    return {"shape": shape, "slice_axis": sa, "row_axis": ra, "col_axis": ca,
            "n_slices": shape[sa], "in_plane_mm": float(spacing[ra]),
            "slice_mm": float(spacing[sa]),
            "has_position": has_position, "has_orientation": has_orientation,
            "layout_ok": layout_ok, "affine": aff}


def study_laterality(meta_path) -> dict:
    """study_meta.csv -> {StudyInstanceUID: (side, source)}. The DICOM answer, precomputed.

    kaggle_01b resolved this from the headers over all 4,407 studies: the (0020,0060) tag where
    it is non-empty, and the x < -62 geometry rule otherwise (FINDINGS.md 6.2). Reading it back
    here keeps the NIfTI path on exactly the same laterality decision the DICOM path makes, which
    matters more than usual because the NIfTI files cannot supply it themselves.
    """
    import csv
    out = {}
    with open(meta_path, newline="") as fh:
        for row in csv.DictReader(fh):
            uid = row.get("StudyInstanceUID")
            if not uid or uid in out:
                continue
            tag = (row.get("laterality_tag") or "").strip().upper()[:1]
            if tag in ("L", "R"):
                out[uid] = (tag, "tag")
                continue
            try:
                x = float(row.get("x_median") or "")
            except ValueError:
                out[uid] = (None, "none")
                continue
            # float("nan") does NOT raise, and `nan < -62` is False -- so a NaN coordinate would
            # fall through as a confident "L" and mirror that study's axial and coronal volumes,
            # breaking 4 of the 12 labels while validate_nifti counted it as resolved. Latent
            # today (0 of 4,407 rows), which is exactly when it is cheap to close.
            if not np.isfinite(x):
                out[uid] = (None, "none")
                continue
            out[uid] = (("R" if x < LATERALITY_X_THRESHOLD else "L"), "geometry")
    return out


def load_series_nifti(path, plane: str | None = None, laterality: str | None = None,
                      laterality_source: str = "none", slice_direction: str | None = None):
    """One .nii -> the same ([S,side,side] float32 in [0,1], laterality, source) as load_series.

    Everything after the read is the SHARED code path -- pick_slices, normalise_and_resample,
    canonicalise. Only the reader differs, which is the point: the parity-critical arithmetic has
    one definition and this cannot drift from it.

    `laterality` MUST be passed in, from study_laterality(). It is not derivable here: these
    files carry spacing and nothing else (see the section header). Passing None is honest rather
    than safe -- canonicalise() then leaves the volume alone and the study is recorded as
    uncanonicalised, which is the behaviour the DICOM path already has when both tag and geometry
    are missing.

    `plane` likewise comes from train_series.csv, not from the file.
    """
    import nibabel as nib
    g = nifti_geometry(path)
    if g is None:
        return None, laterality, laterality_source
    try:
        arr = np.asanyarray(nib.load(str(path)).dataobj)
    except Exception:
        return None, laterality, laterality_source

    vol = np.transpose(arr, (g["slice_axis"], g["row_axis"], g["col_axis"])).astype(np.float32)
    vol = vol[pick_slices(len(vol))]
    if len(vol) < 3:
        return None, laterality, laterality_source
    v = normalise_and_resample(np.ascontiguousarray(vol), g["in_plane_mm"])
    # None until kaggle_01c exports the per-series bit (K16). The NIfTI carries no direction
    # cosines, so there is nothing here to derive it from -- see the section header.
    return (canonicalise(v, laterality, plane, slice_direction=slice_direction),
            laterality, laterality_source)


def manifest(**extra) -> dict:
    """The reproducibility contract written beside every cache shard."""
    return {"preprocess_version": PREPROCESS_VERSION, "model": MODEL, "img_size": IMG_SIZE,
            "slices_per_series": SLICES_PER_SERIES,
            "slices_per_series_train": SLICES_PER_SERIES_TRAIN,
            "target_mm": TARGET_MM, "fov_mm": FOV_MM, "embed_dim": EMBED_DIM,
            "canonical_side": CANONICAL_SIDE, "decode_backend": DECODE_BACKEND, **extra}


def assert_matches(cache_manifest: dict) -> None:
    """Fail loudly when the features were built by a different version of this file.

    The synthetic check comes first because it is the failure with no symptom: a version
    mismatch scores badly and looks wrong, but heads trained on random tensors produce a
    perfectly well-formed submission of noise.
    """
    if cache_manifest.get("synthetic"):
        raise SystemExit(
            "these fusion heads were trained on SYNTHETIC features -- random tensors, so every "
            "AUC is chance by construction and this submission would be noise.\n"
            "fusion/train.py --synthetic writes this marker beside its checkpoints so a smoke "
            "run cannot reach the leaderboard. Re-train on the real cache:\n"
            "  python fusion/train.py --features data/features_224"
        )
    got = cache_manifest.get("preprocess_version")
    if got != PREPROCESS_VERSION:
        raise SystemExit(
            f"preprocessing mismatch: features were built with {got!r}, this file is "
            f"{PREPROCESS_VERSION!r}.\nThe test set would be preprocessed differently from the "
            f"training set and the model would silently score badly. Rebuild the cache or "
            f"restore the matching version of pipeline/preprocess.py."
        )


def self_test() -> None:
    """The canonicalise axis table, asserted rather than reasoned about.

    Worth its own test because the sagittal correction is an XOR against `slice_direction`, and
    the two properties that matter are both invisible to the existing self-tests: that leaving
    SAGITTAL_LR_SLICE_FLIP off is a byte-level no-op, and that with it on the two readers agree.
    build_cache_local and kaggle_02 both pass either way -- they assert on npz SHAPE, which no
    axis flip changes.
    """
    # slice i carries value i; column 0 is marked, so both axes are traceable through a flip.
    vol = np.zeros((4, 2, 3), np.float32)
    for i in range(4):
        vol[i] = i
    vol[:, :, 0] += 0.1

    def order(o):
        return "".join(str(int(s[0, 1])) for s in o)

    def xflipped(o):
        return bool(o[0, 0, 0] < o[0, 0, 2])

    # (laterality, plane, slice_direction) -> (slice order, in-plane mirrored)
    off = {("R", "Sagittal", "forward"): ("0123", False),
           ("L", "Sagittal", "forward"): ("0123", False),     # the bug: left runs opposite
           ("L", "Sagittal", None): ("0123", False),
           ("L", "Coronal", "forward"): ("0123", True),
           ("R", "Coronal", "forward"): ("0123", False)}
    on = {("R", "Sagittal", "forward"): ("0123", False),
          ("L", "Sagittal", "forward"): ("3210", False),      # corrected onto the right knee
          ("L", "Sagittal", "reversed"): ("0123", False),     # XOR: K16 flip cancels the fix
          ("R", "Sagittal", "reversed"): ("3210", False),     # K16 restore alone
          ("L", "Sagittal", None): ("0123", False),           # unknown direction -> hands off
          ("L", "Coronal", "forward"): ("0123", True),        # in-plane, unaffected by the switch
          ("L", "Coronal", "reversed"): ("3210", True)}

    expect = on if SAGITTAL_LR_SLICE_FLIP else off
    for (lat, plane, d), (want_order, want_x) in expect.items():
        o = canonicalise(vol, lat, plane, slice_direction=d)
        got = (order(o), xflipped(o))
        assert got == (want_order, want_x), (
            f"canonicalise({lat}, {plane}, {d!r}) -> {got}, expected {(want_order, want_x)}")
        assert o.flags["C_CONTIGUOUS"], f"{lat}/{plane}/{d} returned a non-contiguous view"
    print(f"  canonicalise table OK ({len(expect)} cases, "
          f"SAGITTAL_LR_SLICE_FLIP={SAGITTAL_LR_SLICE_FLIP})")

    # The fingerprint must move when the switch does, and must NOT move when it does not --
    # otherwise turning it on silently keeps a stale cache, or leaving it off forces a rebuild
    # of features that are already correct.
    import os as _os
    import subprocess
    here = str(Path(__file__).resolve())
    vs = {}
    for flag in ("0", "1"):
        env = {**_os.environ, "SAGITTAL_LR": flag}
        vs[flag] = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, {str(Path(here).parent)!r}); "
             "from preprocess import PREPROCESS_VERSION as v; print(v)"],
            env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert vs["0"] != vs["1"], f"fingerprint did not move with the switch: {vs}"
    print(f"  fingerprint off={vs['0']} on={vs['1']} -- distinct OK")

    print("\nself-test PASSED")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        print(json.dumps(manifest(), indent=2))
