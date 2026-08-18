"""PLAN.md C-4 Gate 1: reproduce aagatti/nnunet_knee's OWN prediction on their OWN test volume.

THIS GATE TESTS OUR HARNESS, NOT THEIR MODEL. That distinction is the whole point and it is
IMPROVEMENTS 9h's lesson: a wrong-table bug runs perfectly and scores wrongly. So the reference
here is `test_prediction.nii.gz` -- the output THEY published -- and not `test_ground_truth.nii.gz`.
Agreement with their output means we drive the model as they did. Agreement with ground truth
would only mean the model is good, which is their claim to make, not ours to verify.

Gate 2 (domain: does it produce sane masks on OUR corpus, read PER PLANE x SEQUENCE) is a
separate script and MUST NOT be folded into this one -- C-4 fixes that order.

PASS: mean Dice vs their prediction >= 0.99 across the foreground classes. Anything materially
below that is a harness difference (spacing, orientation, TTA, checkpoint) and must be found
before any box is calibrated, because a mis-registered mask moves every crop silently.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
D = PROJ / "data" / "_nnunet"


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    tot = a.sum() + b.sum()
    return 1.0 if tot == 0 else float(2.0 * inter / tot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--folds", default="1")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    a = ap.parse_args()

    import torch
    import nibabel as nib
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    dev = a.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {dev}")

    model = D / "model"
    ref_p = D / "test" / "test_prediction.nii.gz"
    img_p = D / "test" / "test_image.nii.gz"
    for p in (model / "plans.json", model / "dataset.json", img_p, ref_p):
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            return 2

    # nnU-Net v2 requires the _0000 channel suffix on inputs.
    ind, outd = D / "gate1_in", D / "gate1_out"
    ind.mkdir(exist_ok=True); outd.mkdir(exist_ok=True)
    staged = ind / "case_0000.nii.gz"
    if not staged.exists():
        shutil.copy(img_p, staged)

    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           device=torch.device(dev), verbose=False, allow_tqdm=True)
    pred.initialize_from_trained_model_folder(
        str(model), use_folds=tuple(int(f) for f in a.folds.split(",")),
        checkpoint_name=a.checkpoint)

    t0 = time.time()
    pred.predict_from_files([[str(staged)]], [str(outd / "case.nii.gz")],
                            save_probabilities=False, overwrite=True,
                            num_processes_preprocessing=1, num_processes_segmentation_export=1)
    mins = (time.time() - t0) / 60
    print(f"predicted in {mins:.1f} min")

    ours = np.asanyarray(nib.load(outd / "case.nii.gz").dataobj)
    theirs = np.asanyarray(nib.load(ref_p).dataobj)
    if ours.shape != theirs.shape:
        print(f"⛔ SHAPE MISMATCH ours {ours.shape} theirs {theirs.shape}")
        return 1

    labels = json.loads((model / "dataset.json").read_text()).get("labels", {})
    inv = {int(v): k for k, v in labels.items() if int(v) != 0} if labels else {}
    classes = sorted(set(np.unique(theirs)) | set(np.unique(ours)) - {0})
    classes = [int(c) for c in classes if int(c) != 0]

    print(f"\n{'class':<28}{'dice':>8}{'ours vox':>12}{'theirs vox':>12}")
    ds = []
    for c in classes:
        d = dice(ours == c, theirs == c)
        ds.append(d)
        print(f"{inv.get(c, str(c)):<28}{d:>8.4f}{int((ours==c).sum()):>12}{int((theirs==c).sum()):>12}")
    mean = float(np.mean(ds)) if ds else 0.0
    agree = float((ours == theirs).mean())
    print(f"\nmean foreground Dice vs THEIR prediction: {mean:.4f}")
    print(f"exact voxel agreement:                    {agree:.6f}")
    verdict = "PASS" if mean >= 0.99 else "FAIL"
    print(f"\nGATE 1 {verdict} (threshold 0.99, fixed in PLAN.md C-4 before the run)")
    (D / "gate1_result.json").write_text(json.dumps(
        {"mean_dice": mean, "exact_agreement": agree, "per_class": dict(zip(map(str, classes), ds)),
         "device": dev, "minutes": mins, "checkpoint": a.checkpoint, "verdict": verdict}, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
