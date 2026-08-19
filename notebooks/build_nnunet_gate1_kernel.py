"""Build the C-4 Gate 1 kernel: reproduce aagatti/nnunet_knee's own prediction, on a GPU.

    .venv/bin/python notebooks/build_nnunet_gate1_kernel.py --out /tmp/g1
    .venv/bin/python -m kaggle kernels push -p /tmp/g1

WHY THIS MOVED OFF THE LAPTOP `MEASURED 2026-08-18`. Gate 1 ran locally on MPS and reached
127/150 sliding-window tiles in 1 h 29 m before it was killed -- one tile alone took 17 min 50 s.
Not swap: gate1's RSS was 0.42 GB. It is MPS unified memory plus a CPU-side aggregation buffer
that exists only because **MPS forces `perform_everything_on_device=False`** ("only supported for
cuda devices"). A CUDA T4 restores it, which is the whole point of moving.

⛔ THIS IS A PLUMBING GATE. The reference is THEIR `test_prediction.nii.gz`, not their
`test_ground_truth.nii.gz`. Agreeing with their output proves we drive the model as they did;
agreeing with ground truth would only prove their model is good, which is their claim, not ours.
§9h is why: a wrong-table bug runs perfectly and scores wrongly.

⚠️ NOT A SUBMISSION KERNEL. `enable_internet: true` so it can pip-install nnunetv2 and pull the
weights from the Hub. Nothing here touches the competition data or the scored path.

THREE FAILURES BOUGHT THE SHAPE OF THIS FILE. Each fix is kept because each failure is cheap to
repeat and expensive to diagnose:

  v1  pip pulled numpy 2.5.2 over the numpy the session had ALREADY IMPORTED; the next import died
      with `cannot import name '_center' from 'numpy._core.umath'`. The install itself succeeded.
      FIX: run the real work in a SUBPROCESS, so its imports happen fresh whatever pip did.
  v2  pinned numpy to the image's 2.0.2 to stop that -- and nnunetv2's compiled deps (blosc2,
      acvl_utils, batchgeneratorsv2) ship wheels built against a newer numpy, so the child died in
      0.1 s on a binary mismatch. FIX: no pin. The subprocess alone already covers v1.
      v2 also inherited the child's fds, so papermill captured NOTHING and the failure reported
      only "exited 1". FIX: merge stderr into stdout and pump it line by line.
  v3  the work script was embedded indented and rebuilt with `textwrap.dedent` at runtime. It
      dedented correctly LOCALLY and did not in the kernel -- `IndentationError` on line 2.
      FIX: no dedent anywhere. WORK_SRC is written flush-left here and embedded as a JSON-escaped
      string literal, so there is no indentation to strip and nothing to differ across the
      boundary. A mechanism that behaves differently in two places is not worth debugging when
      deleting it costs nothing.

The title IS the slug, deliberately: Kaggle derives the slug from the title and a mismatch with
`id` returns 409 on every push after the first -- the failure that cost a cycle on the F6 kernel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SLUG = "rsna-knee-nnunet-gate1"
OWNER = "raahimnawaz"

# --- the child. FLUSH-LEFT ON PURPOSE (see v3 above). Never indent this string.
WORK_SRC = '''
import json, os, shutil, time
import numpy as np, torch, nibabel as nib
from huggingface_hub import hf_hub_download

t0 = time.time()
print(f"numpy {np.__version__} | torch {torch.__version__} | cuda {torch.cuda.is_available()} "
      f"| {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

# FAIL FAST ON THE GPU. v4 answered this question only after installing, downloading 816 MB and
# loading the model -- 58 s to learn the device could not run a conv3d. Five seconds now instead.
if torch.cuda.is_available():
    print("arch list: " + str(torch.cuda.get_arch_list())
          + " | capability " + str(torch.cuda.get_device_capability(0)), flush=True)
    try:
        _x = torch.zeros(1, 1, 8, 8, 8, device="cuda")
        _c = torch.nn.Conv3d(1, 1, 3, padding=1).cuda()
        _ = _c(_x); torch.cuda.synchronize()
        print("conv3d smoke test OK on cuda", flush=True)
    except Exception as e:
        print("CONV3D SMOKE TEST FAILED: " + type(e).__name__ + ": " + str(e)[:200], flush=True)
        print("the installed torch has no kernels for this device -- pin the image's torch", flush=True)
        raise SystemExit(3)

REPO = "aagatti/nnunet_knee"
M = "models/Dataset500_KneeMRI/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
root = "/kaggle/working/model"
os.makedirs(root + "/fold_1", exist_ok=True)
for rel, dst in [(M + "/plans.json", root + "/plans.json"),
                 (M + "/dataset.json", root + "/dataset.json"),
                 (M + "/fold_1/checkpoint_final.pth", root + "/fold_1/checkpoint_final.pth")]:
    shutil.copy(hf_hub_download(REPO, rel), dst)
img_p = hf_hub_download(REPO, "test_data/test_image.nii.gz")
ref_p = hf_hub_download(REPO, "test_data/test_prediction.nii.gz")
print(f"[{time.time()-t0:6.1f}s] weights + test data pulled", flush=True)

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

dev = "cuda" if torch.cuda.is_available() else "cpu"
# perform_everything_on_device is CUDA-only -- exactly what MPS refused locally (PLAN.md C-4).
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
    theirs = np.transpose(theirs, (2, 0, 1))          # nnU-Net returns z,y,x
print("ours " + str(ours.shape) + " theirs " + str(theirs.shape), flush=True)

labels = json.load(open(root + "/dataset.json"))["labels"]
inv = {int(v): k for k, v in labels.items() if int(v) != 0}


def dice(a, b):
    tot = a.sum() + b.sum()
    return 1.0 if tot == 0 else float(2.0 * np.logical_and(a, b).sum() / tot)


print("")
print(f"{'class':<16}{'dice':>8}{'ours':>12}{'theirs':>12}")
ds = []
for c in sorted(inv):
    d = dice(ours == c, theirs == c)
    ds.append(d)
    print(f"{inv[c]:<16}{d:>8.4f}{int((ours==c).sum()):>12}{int((theirs==c).sum()):>12}")
mean = float(np.mean(ds))
agree = float((ours == theirs).mean())
print("")
print(f"mean foreground Dice vs THEIR prediction: {mean:.4f}")
print(f"exact voxel agreement:                    {agree:.6f}")
print("")
print(f"GATE 1 {'PASS' if mean >= 0.99 else 'FAIL'} (threshold 0.99, fixed in PLAN.md C-4 before the run)")
print(f"PER-VOLUME COST ON {dev.upper()}: {mins:.2f} min  <- the number C-4 Variant A needs")
json.dump({"mean_dice": mean, "exact_agreement": agree, "minutes": mins, "device": dev,
           "per_class": {inv[c]: d for c, d in zip(sorted(inv), ds)}},
          open("/kaggle/working/gate1_result.json", "w"), indent=2)
'''

PARENT = '''# C-4 Gate 1 -- reproduce aagatti/nnunet_knee's OWN prediction on their OWN test volume.
# Reference is their test_prediction.nii.gz. This tests OUR harness, not their model.
# See notebooks/build_nnunet_gate1_kernel.py for why this is shaped the way it is (v1-v3).
import subprocess, sys, time

t0 = time.time()
# No numpy pin: v2 pinned it and nnunetv2's compiled deps are built against a newer numpy.
# torch is pinned, but ⚠️ NOT for the reason v5 claimed. v5 blamed pip for replacing the image's
# torch; v5's own smoke test then showed the real cause was that Kaggle had assigned a P100 (sm_60)
# because machine_shape was missing. The pin is kept as cheap insurance against pip clobbering a
# working torch, which is a real failure mode -- it is just not what v4 hit. v4 died with
#   "CUDA error: no kernel image is available for execution on the device"
# after paying the full install + 816 MB download + model load. The image's torch is the one built
# for this hardware, so pin it and let pip solve around it.
import torch as _t
TORCH = _t.__version__.split("+")[0]
print(f"pinning torch=={TORCH} (the image's, built for this GPU)", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nnunetv2", f"torch=={TORCH}"],
               check=True)
print(f"[{time.time()-t0:6.1f}s] nnunetv2 installed", flush=True)

open("/kaggle/working/g1_work.py", "w").write(WORK)

# Stream the child. v2 inherited its fds, papermill captured nothing, and a child that died in
# 0.1 s reported only "exited 1" with no traceback. Never again.
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
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # WORK is embedded as a JSON-escaped literal -- no indentation, no dedent, nothing to differ.
    src = "WORK = " + json.dumps(WORK_SRC) + "\n\n" + PARENT
    compile(WORK_SRC, "work", "exec")          # the child must compile before it ships
    compile(src, "cell", "exec")               # and so must the cell

    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": src.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))

    meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "keywords": ["gpu"], "dataset_sources": [], "kernel_sources": [],
            "competition_sources": [], "model_sources": [],
            # ⛔ REQUIRED. Omitting this got a Tesla P100 (sm_60), and the image's own
            # torch 2.10.0+cu128 supports sm_70..sm_120 -- so the P100 is too OLD for it and no
            # pin can fix that. v5 died in the conv3d smoke test naming exactly this. The F6
            # submission kernel has always specified T4; this one simply forgot to.
            "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"built {out}/{SLUG}.ipynb  (id {meta['id']})")


if __name__ == "__main__":
    main()
