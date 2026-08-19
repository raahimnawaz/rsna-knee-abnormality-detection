"""Build the C-4 Gate 1 kernel: reproduce aagatti/nnunet_knee's own prediction, on a GPU.

    .venv/bin/python notebooks/build_nnunet_gate1_kernel.py --out /tmp/g1
    .venv/bin/python -m kaggle kernels push -p /tmp/g1

WHY THIS MOVED OFF THE LAPTOP `MEASURED 2026-08-18`. Gate 1 ran locally on MPS and reached
127/150 sliding-window tiles in 1 h 29 m before it was killed -- one tile alone took 17 min 50 s.
The cause is not ordinary memory pressure: gate1's RSS was 0.42 GB while swap sat at 4.9 GB, which
is MPS unified-memory behaviour plus a CPU-side aggregation buffer that only exists because
**MPS forces `perform_everything_on_device=False`** ("only supported for cuda devices"). A CUDA T4
restores it, which is the whole point of moving.

⛔ THIS IS A PLUMBING GATE. The reference is THEIR `test_prediction.nii.gz`, not their
`test_ground_truth.nii.gz`. Agreeing with their output proves we drive the model as they did;
agreeing with ground truth would only prove their model is good, which is their claim, not ours.
§9h is why: a wrong-table bug runs perfectly and scores wrongly.

⚠️ NOT A SUBMISSION KERNEL. `enable_internet: true` so it can pip-install nnunetv2 and pull the
weights from the Hub. Nothing here touches the competition data or the scored path.

The title IS the slug, deliberately. Kaggle derives the slug from the title, and a mismatch
between that and `id` returns 409 on every push after the first -- the failure that cost a cycle
on the F6 kernel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SLUG = "rsna-knee-nnunet-gate1"
OWNER = "raahimnawaz"

SRC = r'''
# C-4 Gate 1 -- reproduce aagatti/nnunet_knee's OWN prediction on their OWN test volume.
# Reference is their test_prediction.nii.gz. This tests OUR harness, not their model.
#
# VERSION 1 FAILED HERE, and the failure is worth keeping. `pip install nnunetv2` pulled
# numpy 2.5.2 over the numpy this session had ALREADY IMPORTED, and the next import died with
# `cannot import name '_center' from 'numpy._core.umath'`. The install itself was fine -- it
# reported success at 18.8 s. Only the live interpreter was poisoned. Two independent guards now:
#   1. numpy is PINNED to whatever the image already ships, so pip has no reason to touch it.
#   2. the real work runs in a SUBPROCESS, so its imports happen fresh no matter what pip did.
# Either alone would probably do. Both cost nothing and this kernel is not cheap to re-run.
import subprocess, sys, time, os, textwrap
t0 = time.time()
# NO numpy pin. v2 pinned numpy to the image's 2.0.2 and the subprocess died in 0.1 s: nnunetv2's
# compiled deps (blosc2, acvl_utils, batchgeneratorsv2) ship wheels built against a NEWER numpy, so
# forcing the old one is a binary mismatch at import. The subprocess guard ALONE already fixes v1 --
# a fresh interpreter imports whatever pip left, consistently. The pin was redundant and harmful.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nnunetv2"], check=True)
print(f"[{time.time()-t0:6.1f}s] nnunetv2 installed", flush=True)

WORK = textwrap.dedent("""
    import json, time, os, shutil
    import numpy as np, torch, nibabel as nib
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    print(f"numpy {np.__version__} | torch {torch.__version__} | cuda {torch.cuda.is_available()} "
          f"| {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    REPO = "aagatti/nnunet_knee"
    M = "models/Dataset500_KneeMRI/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
    root = "/kaggle/working/model"; os.makedirs(f"{root}/fold_1", exist_ok=True)
    for rel, dst in [(f"{M}/plans.json", f"{root}/plans.json"),
                     (f"{M}/dataset.json", f"{root}/dataset.json"),
                     (f"{M}/fold_1/checkpoint_final.pth", f"{root}/fold_1/checkpoint_final.pth")]:
        shutil.copy(hf_hub_download(REPO, rel), dst)
    img_p = hf_hub_download(REPO, "test_data/test_image.nii.gz")
    ref_p = hf_hub_download(REPO, "test_data/test_prediction.nii.gz")
    print(f"[{time.time()-t0:6.1f}s] weights + test data pulled", flush=True)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # perform_everything_on_device is CUDA-only -- exactly what MPS refused locally (C-4).
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           perform_everything_on_device=(dev == "cuda"),
                           device=torch.device(dev), verbose=False, allow_tqdm=True)
    pred.initialize_from_trained_model_folder(root, use_folds=(1,),
                                              checkpoint_name="checkpoint_final.pth")
    arr, props = SimpleITKIO().read_images([img_p])
    print(f"[{time.time()-t0:6.1f}s] loaded {arr.shape} spacing {props['spacing']}", flush=True)
    t1 = time.time()
    ours = np.asarray(pred.predict_single_npy_array(arr, props, None, None, False))
    mins = (time.time() - t1) / 60
    print(f"[{time.time()-t0:6.1f}s] PREDICTED in {mins:.2f} min on {dev}", flush=True)

    theirs = np.asanyarray(nib.load(ref_p).dataobj)
    if ours.shape != theirs.shape:
        theirs = np.transpose(theirs, (2, 0, 1))      # nnU-Net returns z,y,x
    print(f"ours {ours.shape} theirs {theirs.shape}")

    labels = json.load(open(f"{root}/dataset.json"))["labels"]
    inv = {int(v): k for k, v in labels.items() if int(v) != 0}
    def dice(a, b):
        tot = a.sum() + b.sum()
        return 1.0 if tot == 0 else float(2.0 * np.logical_and(a, b).sum() / tot)
    print(f"\n{'class':<16}{'dice':>8}{'ours':>10}{'theirs':>10}")
    ds = []
    for c in sorted(inv):
        d = dice(ours == c, theirs == c); ds.append(d)
        print(f"{inv[c]:<16}{d:>8.4f}{int((ours==c).sum()):>10}{int((theirs==c).sum()):>10}")
    mean = float(np.mean(ds)); agree = float((ours == theirs).mean())
    print(f"\nmean foreground Dice vs THEIR prediction: {mean:.4f}")
    print(f"exact voxel agreement:                    {agree:.6f}")
    print(f"\nGATE 1 {'PASS' if mean >= 0.99 else 'FAIL'} "
          f"(threshold 0.99, fixed in PLAN.md C-4 before the run)")
    print(f"PER-VOLUME COST ON {dev.upper()}: {mins:.2f} min  <- the number C-4 Variant A needs")
    json.dump({"mean_dice": mean, "exact_agreement": agree, "minutes": mins, "device": dev,
               "per_class": {inv[c]: d for c, d in zip(sorted(inv), ds)}},
              open("/kaggle/working/gate1_result.json", "w"), indent=2)
""")
open("/kaggle/working/g1_work.py", "w").write(WORK)
# STREAM the child's output. v2 inherited its fds and papermill captured nothing, so a subprocess
# that died in 0.1 s reported only "exited 1" with no traceback -- blind. Never again: merge stderr
# into stdout and pump it line by line, so a failure explains itself in the log.
proc = subprocess.Popen([sys.executable, "-u", "/kaggle/working/g1_work.py"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
rc = proc.wait()
print(f"[{time.time()-t0:6.1f}s] work subprocess exited {rc}", flush=True)
sys.exit(rc)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": SRC.strip().splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))

    meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "keywords": ["gpu"], "dataset_sources": [], "kernel_sources": [],
            "competition_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"built {out}/{SLUG}.ipynb  (id {meta['id']}, title {meta['title']})")


if __name__ == "__main__":
    main()
