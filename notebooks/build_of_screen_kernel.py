"""Build the C-5 OrthoFoundation feature-cache kernel. PILOT SHARD FIRST.

    .venv/bin/python notebooks/build_of_screen_kernel.py --out /tmp/ofs --shard 0 --n-shards 8
    .venv/bin/python -m kaggle kernels push -p /tmp/ofs

⛔ A SHARD VALIDATES THE PIPELINE AND MEASURES THROUGHPUT. IT IS NOT A STRENGTH READ, and no
strength number may be quoted from one -- §3b, and [[never-select-on-a-tiny-set]]: a choice made on
a small set reports a gain and delivers a loss. The strength gate is §9e's rule on the full
4,407-study OOF.

WHY IT IMPORTS `build_cache` RATHER THAN COPYING IT. `kaggle_02_dinov2_cache.py` already takes
`embed_fn` as a parameter -- its own self-test injects a fake one -- so the tested decode
scheduler, the resume-on-restart behaviour and the laterality accounting are reused rather than
duplicated. Three copies of a decode path is the drift `PREPROCESS_VERSION` exists to catch and
the failure `preprocess.py` admits it cannot see. The module is now carried in the
`rsna-knee-code` Dataset for exactly this reason.

⛔ THE ENV IS SET BEFORE THE IMPORT, AND THAT ORDER IS LOAD-BEARING. `kaggle_02` reads MODEL,
EMBED_DIM and IMG_SIZE from `preprocess` at MODULE level, and `PREPROCESS_VERSION` is a hash of
those VALUES. Setting them afterwards would cache OrthoFoundation features under the ViT-B
fingerprint -- one version over two different caches, which is precisely the silent mismatch the
fingerprint exists to prevent.

    MODEL=orthofoundation-L   a LABEL, not a timm name: this kernel builds the model itself
    EMBED_DIM=2048            embed() concatenates CLS(1024) || patch-mean(1024)
    IMG_SIZE=336              patch 16 divides 336 exactly (21x21 = 441). 518 does NOT (32.375)

⚠️ `dynamic_img_size` is NOT needed here and that is a real difference from the DINOv2 path.
DINOv2 carries a learned position embedding that must be interpolated off its 518-native shape;
OrthoFoundation is **RoPE**, which is resolution-flexible by construction. `img_size=336` is passed
directly instead.

Cost is already measured (§C-5): 16.95 ms/tile, 101.71 ms/study, so the encode side of a full
4,407-study pass is 7.47 min. **This job is decode-bound like every cache build here (§3e)** and
the shard exists to measure that half on real DICOMs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SLUG = "rsna-knee-orthofoundation-screen"
OWNER = "raahimnawaz"

WORK_SRC = '''
# C-5 OrthoFoundation feature cache, pilot shard.
#
# ⛔ EVERYTHING HEAVY IS UNDER `if __name__ == "__main__"`, AND THAT GUARD IS THE WHOLE FIX.
# `build_cache` drives a spawn-context ProcessPoolExecutor, and spawn RE-IMPORTS __main__ in every
# worker -- __main__ being this file. v1 and v2 had no guard, so all four workers re-executed the
# script top to bottom: v1 had each of them load a 1.21 GB checkpoint and build a 303 M ViT-L until
# the box died (BrokenProcessPool at 453 s), and v2, which deletes the checkpoint after loading,
# had them die on FileNotFoundError instead (BrokenProcessPool at 458 s). One cause, two symptoms,
# and neither was the memory tuning I blamed first. kaggle_02 carries the same guard for the same
# reason.
#
# What stays at module level is only what a worker legitimately needs on re-import: the env vars
# preprocess reads at import time, and the sys.path entry. Both are cheap and both must be set
# BEFORE kaggle_02/preprocess are imported -- see the builder docstring on the fingerprint.
import glob, json, os, sys, time, urllib.request

os.environ["MODEL"] = "orthofoundation-L"
os.environ["EMBED_DIM"] = "2048"
os.environ["IMG_SIZE"] = "336"
os.environ.setdefault("N_WORKERS", "4")
os.environ.setdefault("PREFETCH", "6")

_CODE_DIR_ENV = "RSNA_OF_CODE_DIR"
_cached = os.environ.get(_CODE_DIR_ENV)
if _cached and os.path.exists(os.path.join(_cached, "kaggle_02_dinov2_cache.py")):
    sys.path.insert(0, _cached)                      # workers take this path, not the glob
else:
    _hits = sorted(glob.glob("/kaggle/input/**/notebooks/kaggle_02_dinov2_cache.py", recursive=True))
    if not _hits:
        print("cannot find kaggle_02_dinov2_cache.py -- attach the rsna-knee-code Dataset", flush=True)
        print("/kaggle/input holds: " + str(sorted(glob.glob("/kaggle/input/*"))), flush=True)
        raise SystemExit(2)
    _dir = os.path.dirname(_hits[0])
    sys.path.insert(0, _dir)
    os.environ[_CODE_DIR_ENV] = _dir

URL = "https://media.githubusercontent.com/media/ytrsk/OrthoFoundation/main/OrthoFoudation-L.pth"
DST = "/tmp/OrthoFoudation-L.pth"


def build_model(dev):
    import torch, timm
    if not os.path.exists(DST):
        urllib.request.urlretrieve(URL, DST)
    assert os.path.getsize(DST) == 1213056638, "size mismatch (raw.githubusercontent gives a pointer)"
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
    # RoPE, so no dynamic_img_size. 336/16 = 21 -> 441 patches.
    m = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0, img_size=336)
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp, "non-strict load"
    with torch.no_grad():
        m.rope.periods.copy_(periods.to(m.rope.periods.dtype))   # NON-PERSISTENT buffer
    assert torch.allclose(m.rope.periods.float(), periods.float(), atol=0)
    m = m.eval().to(dev)
    del raw, sd
    import gc; gc.collect()
    return m


def main():
    import torch
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("refusing to start on CPU -- re-run to draw a GPU", flush=True); raise SystemExit(2)
    print(f"{torch.cuda.get_device_name(0)} | capability {torch.cuda.get_device_capability(0)}", flush=True)

    model = build_model(dev)
    print(f"[{time.time()-t0:6.1f}s] OrthoFoundation-L strict, rope=theirs", flush=True)

    import kaggle_02_dinov2_cache as k2
    from preprocess import (BATCH_HINT, EMBED_DIM, IMG_SIZE, MODEL, PREPROCESS_VERSION,
                            embed, find_competition_root, manifest)
    print(f"MODEL={MODEL} EMBED_DIM={EMBED_DIM} IMG_SIZE={IMG_SIZE}", flush=True)
    print(f"PREPROCESS_VERSION={PREPROCESS_VERSION}   <- must NOT be the ViT-B 2eddb3ec68d0", flush=True)
    assert PREPROCESS_VERSION != "2eddb3ec68d0", "fingerprint collided with the ViT-B cache"

    probe = embed(model, torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), dev, BATCH_HINT)
    assert probe.shape == (1, EMBED_DIM), f"embed() gave {probe.shape}, expected (1,{EMBED_DIM})"
    print(f"[{time.time()-t0:6.1f}s] embed smoke test OK -> {EMBED_DIM}-d", flush=True)

    import pandas as pd
    root = find_competition_root()
    OUT = k2.OUT; OUT.mkdir(parents=True, exist_ok=True)
    series = pd.read_csv(root / "train_series.csv")
    studies = sorted(series.StudyInstanceUID.unique())
    S, N = int(os.environ["SHARD"]), int(os.environ["N_SHARDS"])
    mine = [s for i, s in enumerate(studies) if i % N == S]
    print("", flush=True)
    print(f"shard {S}/{N}: {len(mine):,} of {len(studies):,} studies", flush=True)
    print(f"decode workers {os.environ['N_WORKERS']}, prefetch {os.environ['PREFETCH']}", flush=True)
    try:
        import resource
        print(f"parent RSS before decode: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB", flush=True)
    except Exception:
        pass

    t1 = time.time()
    done, skipped, lat = k2.build_cache(root, mine, series.set_index("SeriesInstanceUID"), OUT,
                                        embed_fn=lambda x: embed(model, x, dev, BATCH_HINT))
    mins = (time.time() - t1) / 60
    print("", flush=True)
    print(f"shard {S}: {done:,} written, {skipped:,} skipped in {mins:.1f} min", flush=True)
    if done:
        print(f"THROUGHPUT: {mins*60/done:.2f} s/study -> 4,407 studies = {mins*4407/done/60:.2f} h", flush=True)
        print("  (decode-bound: C-5 measured the ENCODE half at 101.71 ms/study)", flush=True)
    print(f"laterality: {lat}", flush=True)
    (OUT / f"_shard{S}.json").write_text(json.dumps(
        manifest(written=done, shard=S, n_shards=N, laterality=lat), indent=2))
    print("", flush=True)
    print("⛔ A SHARD IS NOT A STRENGTH READ. Full 4,407 OOF via §9e, never gold-58 (§3b).", flush=True)


if __name__ == "__main__":
    main()
'''

PARENT = '''import subprocess, sys, time, os
t0 = time.time()
os.environ.setdefault("SHARD", "__SHARD__"); os.environ.setdefault("N_SHARDS", "__NSHARDS__")
open("/kaggle/working/of_work.py", "w").write(WORK)
proc = subprocess.Popen([sys.executable, "-u", "/kaggle/working/of_work.py"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        bufsize=1, env=os.environ)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
rc = proc.wait()
print(f"[{time.time()-t0:6.1f}s] exited {rc}", flush=True)
sys.exit(rc)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    parent = PARENT.replace("__SHARD__", str(a.shard)).replace("__NSHARDS__", str(a.n_shards))
    src = "WORK = " + json.dumps(WORK_SRC) + "\n\n" + parent
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
         "keywords": ["gpu"], "dataset_sources": ["raahimnawaz/rsna-knee-code"],
         "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"],
         "model_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=2))
    print(f"built {out}/{SLUG}.ipynb  shard {a.shard}/{a.n_shards}")


if __name__ == "__main__":
    main()
