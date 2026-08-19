"""Build the OrthoFoundation COST PROBE kernel. PLAN.md §C-5.

    .venv/bin/python notebooks/build_of_cost_kernel.py --out /tmp/ofc
    .venv/bin/python -m kaggle kernels push -p /tmp/ofc

⛔ WHY THIS RUNS BEFORE THE STRENGTH SCREEN, AND NOT AFTER. OrthoFoundation-L is 303 M params
against the port's current ViT-S at ~22 M, and `SlotNet` pushes B*K images through the encoder per
study. §3e puts a submission at ~2 h of a 9 h ceiling ALREADY. So this arm can be strong and still
unshippable, and the order that finds that out cheapest is cost first. Gate 1 (§C-4) learned the
same lesson the expensive way: it spent 58 s and an 816 MB download to discover the device could
not run a conv3d.

⚠️ NO COMPETITION DATA, DELIBERATELY. Encoder throughput does not depend on pixel content, so this
measures what it needs to with synthetic tiles. That keeps the kernel free of the
internet-plus-competition-data question entirely, and makes it re-runnable at zero rules risk.
Decode cost is NOT measured here and is not this arm's problem -- §3e/§9g already establish decode
as the dominant term, and it is unchanged by swapping the encoder.

WHAT IT PRINTS, AND THE ONLY THING THAT MATTERS: seconds per STUDY at the port's real shape
(6 slots x 3-channel 336 px tiles), and that figure projected onto 1,322 test studies. Compared in
the same run against the current ViT-S baseline, because a ratio measured on one machine is worth
more than two absolute numbers measured on two.

The title IS the slug (Kaggle derives the slug from the title; a mismatch 409s every push after
the first), and machine_shape is pinned to T4 -- omitting it got a P100 whose sm_60 the image's
own torch does not support (§C-4 v5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SLUG = "rsna-knee-orthofoundation-cost"
OWNER = "raahimnawaz"

WORK_SRC = '''
import json, os, time, urllib.request
import torch, timm

t0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0) if dev=='cuda' else 'CPU'}", flush=True)
if dev == "cuda":
    print("capability " + str(torch.cuda.get_device_capability(0))
          + " | arch list " + str(torch.cuda.get_arch_list()), flush=True)
    x = torch.zeros(1, 3, 64, 64, device="cuda")
    torch.nn.Conv2d(3, 8, 3).cuda()(x); torch.cuda.synchronize()
    print("gpu smoke test OK", flush=True)

URL = "https://media.githubusercontent.com/media/ytrsk/OrthoFoundation/main/OrthoFoudation-L.pth"
DST = "/tmp/OrthoFoudation-L.pth"
if not os.path.exists(DST):
    urllib.request.urlretrieve(URL, DST)
print(f"[{time.time()-t0:6.1f}s] weights {os.path.getsize(DST):,} bytes", flush=True)
assert os.path.getsize(DST) == 1213056638, "size mismatch -- raw.githubusercontent gives a 135-byte LFS pointer"

# --- remap, identical to fusion/orthofoundation_model.py. Keep the two in step.
raw = torch.load(DST, map_location="cpu", weights_only=True)
sd, periods = {}, None
for k, v in raw.items():
    if k.startswith("backbone."):
        k = k[len("backbone."):]
    if k == "rope_embed.periods":
        periods = v; continue
    if k == "storage_tokens":
        sd["reg_token"] = v; continue
    if k == "mask_token":
        continue
    if k.endswith(".attn.qkv.bias") or k.endswith(".attn.qkv.bias_mask"):
        assert v.abs().sum() == 0, k + " is not zero"
        continue
    sd[k.replace(".ls1.gamma", ".gamma_1").replace(".ls2.gamma", ".gamma_2")] = v

IMG = 336
of = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0, img_size=IMG)
miss, unexp = of.load_state_dict(sd, strict=False)
assert not miss and not unexp, f"non-strict: {list(miss)[:3]} {list(unexp)[:3]}"
# ⛔ timm keeps rope.periods as a NON-PERSISTENT buffer: load_state_dict never sets it, and theirs
# are the bf16-rounded series. Without this the model runs perfectly and is conditioned wrongly.
with torch.no_grad():
    of.rope.periods.copy_(periods.to(of.rope.periods.dtype))
assert torch.allclose(of.rope.periods.float(), periods.float(), atol=0)
print(f"[{time.time()-t0:6.1f}s] OrthoFoundation-L strict, rope=theirs, "
      f"{sum(p.numel() for p in of.parameters()):,} params", flush=True)

base = timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=False, num_classes=0,
                         img_size=IMG, dynamic_img_size=True)
print(f"baseline ViT-S {sum(p.numel() for p in base.parameters()):,} params", flush=True)

K = 6                      # the port's six protocol slots, one tile each
def bench(model, name, batches=6, bs=6):
    model = model.to(dev).eval()
    if dev == "cuda":
        model = model.half()
    x = torch.randn(bs, 3, IMG, IMG, device=dev, dtype=torch.half if dev == "cuda" else torch.float)
    with torch.no_grad():
        for _ in range(2):                       # warmup
            model(x)
        if dev == "cuda": torch.cuda.synchronize()
        t = time.time()
        for _ in range(batches):
            model(x)
        if dev == "cuda": torch.cuda.synchronize()
        dt = time.time() - t
    per_tile = dt / (batches * bs)
    per_study = per_tile * K
    print(f"\\n{name}")
    print(f"  {per_tile*1000:8.2f} ms / tile      {per_study*1000:9.2f} ms / study ({K} slots)")
    print(f"  1,322 test studies -> {per_study*1322/60:7.2f} min encode")
    print(f"  4,407 OOF studies  -> {per_study*4407/60:7.2f} min encode")
    model.cpu()
    if dev == "cuda": torch.cuda.empty_cache()
    return per_study

o = bench(of, "OrthoFoundation-L (ViT-L/16, 303M)")
b = bench(base, "baseline DINOv2 ViT-S/14 (~22M)")
print(f"\\nRATIO OrthoFoundation / baseline: {o/b:.2f}x per study")
print(f"ADDED encode time on a 1,322-study submission: {(o-b)*1322/60:.1f} min")
print("\\n⚠️ ENCODE ONLY. Decode is the dominant term (§3e) and is unchanged by the encoder swap.")
print("   Read this against the 9 h ceiling with ~2 h already spent by the two-arm path.")
json.dump({"per_study_s_of": o, "per_study_s_base": b, "ratio": o/b, "img": IMG, "slots": K,
           "device": dev, "added_min_1322": (o-b)*1322/60},
          open("/kaggle/working/of_cost.json", "w"), indent=2)
'''

PARENT = '''import subprocess, sys, time
t0 = time.time()
open("/kaggle/working/of_work.py", "w").write(WORK)
proc = subprocess.Popen([sys.executable, "-u", "/kaggle/working/of_work.py"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
rc = proc.wait()
print(f"[{time.time()-t0:6.1f}s] exited {rc}", flush=True)
sys.exit(rc)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = "WORK = " + json.dumps(WORK_SRC) + "\n\n" + PARENT
    compile(WORK_SRC, "work", "exec")
    compile(src, "cell", "exec")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": src.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps(
        {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb",
         "language": "python", "kernel_type": "notebook", "is_private": True,
         "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
         "keywords": ["gpu"], "dataset_sources": [], "kernel_sources": [],
         "competition_sources": [], "model_sources": [],
         "machine_shape": "NvidiaTeslaT4"}, indent=2))
    print(f"built {out}/{SLUG}.ipynb")


if __name__ == "__main__":
    main()
