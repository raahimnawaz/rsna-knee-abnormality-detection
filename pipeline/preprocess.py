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
    payload = json.dumps({
        "model": MODEL, "img_size": IMG_SIZE, "slices": SLICES_PER_SERIES,
        "target_mm": TARGET_MM, "fov_mm": FOV_MM, "embed_dim": EMBED_DIM,
        "canonical_side": CANONICAL_SIDE, "norm": "pct_0.5_99.5", "stack": "2.5d_prev_cur_next",
        "lat_x_threshold": LATERALITY_X_THRESHOLD,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


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


def pick_device() -> str:
    """'cuda' only if the assigned GPU can actually run this PyTorch build.

    `torch.cuda.is_available()` is TRUE on a GPU whose compute capability the installed wheel
    has no kernels for, and the failure then arrives much later as a bare
    `CUDA error: no kernel image is available for execution on the device`.

    Measured 2026-08-08: Kaggle hands out **Tesla P100 (capability 6.0)** alongside T4s, and its
    current PyTorch supports 7.0-12.0. A P100 session therefore cannot run this at all. The
    assignment is a coin flip and the `accelerator` field in kernel-metadata.json does not
    reliably override it, so the only defence is to detect it and say so in seconds rather than
    after the notebook has queued, downloaded weights, and started decoding.
    """
    import torch
    if not torch.cuda.is_available():
        return "cpu"

    # Launch an actual kernel. Inferring support from get_device_capability() against
    # get_arch_list() looks tidier but is indirect, and the first version of this function
    # wrapped it in try/except and fell through to "cuda" on any failure -- so it passed a
    # P100 straight through to the same crash it existed to prevent. Running a real op is the
    # ground truth, it costs microseconds, and there is nothing left to infer.
    # Capability check FIRST, and it fails closed. Two earlier versions of this guard tried to
    # infer support and both fell open on a P100 -- one swallowed an exception and returned
    # "cuda", the other's probe did not raise. Kaggle's PyTorch reports a 7.0 minimum, so
    # anything below that is unusable, full stop, with no inference and no fallthrough.
    try:
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    except Exception:
        major, minor, name = 0, 0, "unknown GPU"
    print(f"GPU: {name}, compute {major}.{minor}")
    if major < 7:
        print(f"WARNING: {name} (compute {major}.{minor}) is below the 7.0 minimum this "
              f"PyTorch supports. Kaggle still assigns P100s and the accelerator field in "
              f"kernel-metadata.json does not override the draw. Re-run for a different GPU.")
        return "unusable"

    try:
        torch.zeros(8, 8, device="cuda").sum().item()
        return "cuda"
    except Exception as e:
        try:
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            what = f"{name} (compute {major}.{minor})"
        except Exception:
            what = "the assigned GPU"
        print(f"WARNING: {what} cannot run this PyTorch build -- {type(e).__name__}: "
              f"{str(e).splitlines()[0]}\nKaggle still assigns Tesla P100s (compute 6.0) while "
              f"its PyTorch requires >= 7.0, and the accelerator field in kernel-metadata.json "
              f"does not reliably override the draw. Re-run to get a different GPU; a T4 works.")
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
def canonicalise(vol: np.ndarray, laterality: str | None, plane: str | None) -> np.ndarray:
    """Mirror left knees onto CANONICAL_SIDE so 'medial' is always the same image side.

    Only in-plane left-right matters, and only for planes that HAVE a left-right axis in the
    image: axial and coronal put medial/lateral along the image x-axis, sagittal does not (its
    x-axis is anterior-posterior), so flipping a sagittal series would mirror the knee
    front-to-back for no gain. Returns the volume unchanged when laterality is unknown --
    guessing is worse than a recorded miss, which the cache stores per series so coverage can
    be audited afterwards.
    """
    if laterality is None or laterality == CANONICAL_SIDE:
        return vol
    if plane not in ("Axial", "Coronal"):
        return vol
    return np.ascontiguousarray(vol[:, :, ::-1])


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
    lo, hi = torch.quantile(v.flatten(), torch.tensor([0.005, 0.995]))
    v = ((v - lo) / (hi - lo + 1e-6)).clamp(0, 1)
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
    return canonicalise(v, lat, plane), lat, lat_src


def manifest(**extra) -> dict:
    """The reproducibility contract written beside every cache shard."""
    return {"preprocess_version": PREPROCESS_VERSION, "model": MODEL, "img_size": IMG_SIZE,
            "slices_per_series": SLICES_PER_SERIES,
            "slices_per_series_train": SLICES_PER_SERIES_TRAIN,
            "target_mm": TARGET_MM, "fov_mm": FOV_MM, "embed_dim": EMBED_DIM,
            "canonical_side": CANONICAL_SIDE, "decode_backend": DECODE_BACKEND, **extra}


def assert_matches(cache_manifest: dict) -> None:
    """Fail loudly when the features were built by a different version of this file."""
    got = cache_manifest.get("preprocess_version")
    if got != PREPROCESS_VERSION:
        raise SystemExit(
            f"preprocessing mismatch: features were built with {got!r}, this file is "
            f"{PREPROCESS_VERSION!r}.\nThe test set would be preprocessed differently from the "
            f"training set and the model would silently score badly. Rebuild the cache or "
            f"restore the matching version of pipeline/preprocess.py."
        )


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2))
