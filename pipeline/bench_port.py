"""Measure the training port's real cost before committing to a 9 GB cache build.

IMPROVEMENTS.md 2e estimates ~12 min/epoch and ~2 h for the fork's EPOCHS=10, inferred from
our measured 9.9 img/s for DINOv2-base@518 scaled by parameter and token counts. It has never
been run. Every route in this project that committed to a multi-hour build on an inferred cost
model lost the hours, so this runs it.

Two numbers, because two different things get committed to:

  A. TRAIN THROUGHPUT -- img/s for forward+backward on dinov2-small@336 with the last six
     blocks trainable. Synthetic tensors on purpose: this isolates compute from I/O, and the
     cache exists precisely so that training never touches NIfTI.
  B. CACHE BUILD COST -- NIfTI open + slot extract + resize per study, on real files.

Both are needed: A sets the per-experiment cost, B sets the one-time entry price.
"""
import time

import numpy as np
import torch

IMG = 336
GROUP = 3          # slices stacked as the three channels
N_SLOT = 6
BATCH_STUDIES = 8
UNFREEZE_LAST = 6
N_STUDIES = 4407

dev = torch.device("mps")


def build(unfreeze_last: int):
    import timm
    m = timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=True,
                          img_size=IMG, num_classes=0)
    for p in m.parameters():
        p.requires_grad = False
    for blk in m.blocks[len(m.blocks) - unfreeze_last:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in m.norm.parameters():
        p.requires_grad = True
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    print(f"dinov2-small @ {IMG}: {len(m.blocks)} blocks, last {unfreeze_last} open, "
          f"{n_tr / 1e6:.1f}M / {n_all / 1e6:.1f}M params trainable")
    return m


print(__doc__)
print("=" * 70)
print("A. TRAIN THROUGHPUT")
print("=" * 70)
model = build(UNFREEZE_LAST).to(dev).train()
head = torch.nn.Linear(model.num_features, 12).to(dev)
opt = torch.optim.AdamW([
    {"params": [p for p in model.parameters() if p.requires_grad], "lr": 8e-6},
    {"params": head.parameters(), "lr": 1e-3}])

B = BATCH_STUDIES * N_SLOT          # images per optimiser step
x = torch.randn(B, GROUP, IMG, IMG, device=dev)
y = torch.rand(B, 12, device=dev)


def step():
    opt.zero_grad(set_to_none=True)
    out = head(model(x))
    torch.nn.functional.binary_cross_entropy_with_logits(out, y).backward()
    opt.step()


print(f"\nbatch = {BATCH_STUDIES} studies x {N_SLOT} slots = {B} images of {GROUP}x{IMG}^2")
for _ in range(3):                   # warm up: first MPS steps pay graph compilation
    step()
torch.mps.synchronize()

t0 = time.time()
N = 12
for _ in range(N):
    step()
torch.mps.synchronize()
dt = time.time() - t0
ips = N * B / dt
print(f"  {N} steps in {dt:.1f}s -> {dt / N * 1000:.0f} ms/step, **{ips:.1f} img/s**")

imgs_per_epoch = N_STUDIES * N_SLOT
ep = imgs_per_epoch / ips
print(f"\n  corpus epoch = {N_STUDIES} studies x {N_SLOT} slots = {imgs_per_epoch:,} images")
print(f"  -> {ep / 60:.1f} min/epoch   |   10 epochs = {ep * 10 / 3600:.2f} h")
print(f"  -> 5 folds x 10 epochs = {ep * 50 / 3600:.1f} h")
print(f"\n  IMPROVEMENTS.md 2e estimated 12 min/epoch and ~2 h for 10 epochs.")
r = (ep / 60) / 12
print(f"  measured/estimated = {r:.2f}x  ->  "
      f"{'WITHIN GATE (<3x)' if r < 3 else 'GATE FAILED (>=3x) -- stop and re-plan'}")

# ---- inference, for the submission-time budget -----------------------------------------
model.eval()
with torch.no_grad():
    for _ in range(2):
        model(x)
    torch.mps.synchronize()
    t0 = time.time()
    for _ in range(8):
        model(x)
    torch.mps.synchronize()
    inf = 8 * B / (time.time() - t0)
print(f"\n  inference: {inf:.0f} img/s ({inf / ips:.1f}x training)")
