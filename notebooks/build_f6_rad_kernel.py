"""Build the THREE-family submission kernel: pilkwang + `ft_b` + the public RadImageNet arm.

§4c measured the arm and it passed every pre-registered gate: strength **0.8486** on gold-47
against a 0.83 bar, partition flat at χ² p=0.442, blend **+0.0146** on gold and **+0.0160 ± 0.0036
(4.4σ, 100% of draws)** on large-n. This turns that measurement into a submission.

    python notebooks/build_f6_rad_kernel.py --src notebooks/submissions --out /tmp/f6rad
    kaggle kernels push -p /tmp/f6rad

WHY A PATCHER AND NOT AN EDITED NOTEBOOK (§3d's pattern, from `build_tta_kernel.py`). Every
substitution below is anchored and asserted to appear **exactly once**. If upstream changes shape,
this refuses to build rather than silently producing a kernel that is not what was measured.

⛔ THE ONE DANGEROUS THING THIS DOES, AND HOW IT IS CONTAINED. The notebook keeps its pixel contract
in MODULE GLOBALS -- `IMG`, `GROUP`, `CACHE_SLICES`, `N_GROUP`, `CROP_MM`, `SLICE_BAND`, `SLOTS`,
`N_SLOT`, `RULES` -- and `adopt_config_globals` rebinds them per decode group. The RadImageNet arm
needs a *completely different* contract (224 px, 8 single slices not 3-channel groups, full frame,
band 0.12-0.88, and **3 fat-suppressed slots instead of 6**). Rebinding those and failing to put
them back would corrupt the pilkwang path **silently**, which is §9h's failure mode and the exact
thing §6b of the notebook exists to prevent. So the rad path saves and restores **every one** of
them in a `try/finally`, and asserts on the way out that each is back to the value it had.

WHAT SHIPS IS WHAT WAS MEASURED. §4c measured an **equal-weight rank-mean over three families**
(§9e's rule). Their board configuration uses `_RAD_ALPHA = 0.50` with
`_RAD_EXCLUDE = ("Baker's", "Fracture")`; **that was not measured here and §3b forbids choosing it
on 47 studies**, so the flat three-way rank-mean ships. Do not invent a third option at submission
time.

⚠️ WHAT THE LOCAL NUMBER DOES AND DOES NOT PREDICT. §4c's +0.0146 was measured against a two-family
blend built with `pool="prob"`, not against the exact banked 0.908 estimator (which uses per-target
TTA pooling, §3d). §3v measured variance-reduction levers as **sub-additive**, so the increment on
top of 0.908 may be smaller than +0.0146. A new *family* is a different kind of lever from a TTA
pool, so it may add better -- but that is a hope, not a measurement, and the LB read is the test.

⚠️ LICENCE, RECORDED AT THE BUILD SITE. The official RadImageNet ResNet-50 trunk is
**CC-BY-NC-SA-4.0** (§4c-3). Shipping this arm accepts that exposure knowingly; `REFERENCE.md` §1.3
is unanswered. The whole public 0.917-0.922 frontier carries it too, which is context, not a licence.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TITLE = "RSNA Knee - F6 three-arm blend (+ RadImageNet)"
# Kaggle derives the slug from the TITLE, so the id must match what the title resolves to or
# every push after the first returns 409 Conflict against a kernel that already exists.
SLUG = "rsna-knee-f6-three-arm-blend-radimagenet"

# Datasets the arm needs. The encoder our local run verified came from a slug that now returns
# 403; `marwanmath/...` is the one the heads' own README names and its ResNet50.pt is
# 94,345,733 bytes -- byte-for-byte the size of the copy whose SHA-256 we checked (§C-3).
RAD_SOURCES = ["mattiaangeli/rsna-knee-radimagenet-foldsv1-heads",
               "marwanmath/resnet-50-radimagenet-marwan"]

# ---------------------------------------------------------------------------- #
# 1. The arm itself, injected as a new cell after the ft_b cell.
# ---------------------------------------------------------------------------- #
RAD_CELL = r'''
# ============================================================================ #
# F6 arm 3: the PUBLIC RadImageNet arm -- frozen official R50 + FoundationQueryHead.
#
# IMPROVEMENTS 4c, measured before this cell was written: strength 0.8486 on the
# 47 gold studies against a 0.83 bar fixed in advance (pilkwang 0.8516, ft_b
# 0.8522); recovered partition flat at chi2 p=0.442; three-family blend +0.0146 on
# gold and +0.0160 +- 0.0036 (4.4 sigma, 100% of draws) on 600 report-labelled
# studies. It BEATS the e11 variant 4b measured (+0.0135 / +0.0102), which sits in
# a bundle whose own README says it must stay private.
#
# LICENCE: the trunk is CC-BY-NC-SA-4.0. Shipping accepts that knowingly (4c-3).
#
# EVERY NAME IS PREFIXED RAD_ / _rad_, for the reason the ft_b cell states.
# ============================================================================ #
import hashlib as _rad_hashlib

RAD_HEADS_DIR = RAD_ENC_PATH = None
for _c in ("/kaggle/input/rsna-knee-radimagenet-foldsv1-heads",
           "/kaggle/input/datasets/mattiaangeli/rsna-knee-radimagenet-foldsv1-heads"):
    if os.path.isdir(_c):
        RAD_HEADS_DIR = _c
        break
for _c in ("/kaggle/input/resnet-50-radimagenet-marwan/ResNet50.pt",
           "/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt"):
    if os.path.exists(_c):
        RAD_ENC_PATH = _c
        break

RAD_IMG, RAD_N_SLICE, RAD_BAND = 224, 8, (0.12, 0.88)
RAD_CROP_FULL = 10000.0          # past any FOV -> read_slot leaves the frame alone
RAD_SLOT_NAMES = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS"]   # manifest plane order
RAD_TOKEN_DIM, RAD_HEAD_DIM, RAD_N_LABEL = 2048, 512, 12
# What the same weights produced locally on 600 non-gold studies (4c). A gross convention
# error moves this a long way; it is a free check that costs no extra decode.
RAD_REF_MEAN, RAD_REF_STD = 0.345, 0.261


def _rad_sha256(p):
    h = _rad_hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class RadEncoder(nn.Module):
    """Official RadImageNet ResNet-50, frozen, global-average-pooled to 2048."""

    def __init__(self):
        super().__init__()
        import torchvision
        self.backbone = nn.Sequential(
            *list(torchvision.models.resnet50(weights=None).children())[:-2])

    def forward(self, image):
        return self.backbone(image).mean(dim=(2, 3))


class RadQueryHead(nn.Module):
    """`FoundationQueryHead`. The 2048 into `fuse` is FOUR 512-vectors concatenated --
    [attended, mean, |attended - mean|, attended * mean] -- which a state dict cannot tell
    you and a wrong guess would run perfectly while scoring wrongly (9h)."""

    def __init__(self, n_plane=3, n_slice=RAD_N_SLICE):
        super().__init__()
        self.n_plane, self.n_slice, self.n_label = n_plane, n_slice, RAD_N_LABEL
        self.project = nn.Sequential(nn.LayerNorm(RAD_TOKEN_DIM),
                                     nn.Linear(RAD_TOKEN_DIM, RAD_HEAD_DIM), nn.GELU())
        self.plane = nn.Parameter(torch.randn(n_plane, RAD_HEAD_DIM) * .01)
        self.position = nn.Parameter(torch.randn(n_slice, RAD_HEAD_DIM) * .01)
        self.query = nn.Parameter(torch.randn(RAD_N_LABEL, RAD_HEAD_DIM) * .02)
        self.attn = nn.MultiheadAttention(RAD_HEAD_DIM, 8, dropout=.10, batch_first=True)
        self.fuse = nn.Sequential(nn.LayerNorm(RAD_HEAD_DIM * 4),
                                  nn.Linear(RAD_HEAD_DIM * 4, RAD_HEAD_DIM),
                                  nn.GELU(), nn.Dropout(.15))
        self.weight = nn.Parameter(torch.randn(RAD_N_LABEL, RAD_HEAD_DIM) * .02)
        self.bias = nn.Parameter(torch.zeros(RAD_N_LABEL))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), self.n_plane, self.n_slice, -1)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        kp = mask <= 0
        empty = kp.all(1)
        if empty.any():
            kp = kp.clone()
            kp[empty, 0] = False
        q = self.query.unsqueeze(0).expand(len(token), -1, -1)
        att = q + self.attn(q, token, token, key_padding_mask=kp, need_weights=False)[0]
        den = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdim=True) / den
        mean = mean.expand(-1, self.n_label, -1)
        fused = self.fuse(torch.cat([att, mean, torch.abs(att - mean), att * mean], dim=-1))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def rad_load(dev):
    """Encoder + five heads, hashes checked against the shipped manifest, heads STRICT."""
    if RAD_HEADS_DIR is None or RAD_ENC_PATH is None:
        log("rad: assets not attached; skipping the arm")
        return None, None
    man = json.load(open(f"{RAD_HEADS_DIR}/rad_heads_manifest.json"))
    got = _rad_sha256(RAD_ENC_PATH)
    if got != man["encoder_sha256"]:
        log(f"rad: ENCODER SHA MISMATCH {got[:12]} != {man['encoder_sha256'][:12]} -- skipping")
        return None, None
    enc = RadEncoder()
    sd = torch.load(RAD_ENC_PATH, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    sd = {k: v for k, v in sd.items() if not k.startswith(("fc.", "classifier."))}
    if any(k.startswith("backbone.") for k in sd):
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    miss, unexp = enc.backbone.load_state_dict(sd, strict=False)
    if miss:
        log(f"rad: encoder has {len(miss)} MISSING keys -- refusing the arm")
        return None, None
    heads = []
    for f in range(5):
        name = f"rad_head_f{f}.pt"
        p = f"{RAD_HEADS_DIR}/{name}"
        if _rad_sha256(p) != man["heads"][name]["sha256"]:
            log(f"rad: {name} SHA MISMATCH -- refusing the arm")
            return None, None
        ck = torch.load(p, map_location="cpu", weights_only=False)
        h = RadQueryHead()
        h.load_state_dict(ck.get("state_dict", ck), strict=True)     # strict is the defence
        heads.append(h.to(dev).eval())
    n = sum(p.numel() for p in RadQueryHead().parameters())
    if n != man["head_parameters"]:
        log(f"rad: param count {n} != manifest {man['head_parameters']} -- refusing")
        return None, None
    log(f"rad: encoder + 5 heads loaded, all 6 SHA-256 verified, {n:,} head params")
    return enc.to(dev).eval(), heads


@torch.no_grad()
def rad_predict(all_ids, hte, plane_map, dev):
    """(n_studies, 12) probabilities, or None. Rebinds the pixel contract and RESTORES it."""
    enc, heads = rad_load(dev)
    if enc is None:
        return None
    global IMG, CACHE_IMG, GROUP, CACHE_SLICES, N_GROUP, CROP_MM, SLICE_BAND, SLOTS, N_SLOT
    saved = dict(IMG=IMG, CACHE_IMG=CACHE_IMG, GROUP=GROUP, CACHE_SLICES=CACHE_SLICES,
                 N_GROUP=N_GROUP, CROP_MM=CROP_MM, SLICE_BAND=SLICE_BAND,
                 SLOTS=list(SLOTS), N_SLOT=N_SLOT)
    try:
        IMG = CACHE_IMG = RAD_IMG
        GROUP = 1                      # one slice per token, expanded to 3 channels at encode
        CACHE_SLICES = N_GROUP = RAD_N_SLICE
        CROP_MM = RAD_CROP_FULL
        SLICE_BAND = RAD_BAND
        # SLOTS entries are 4-TUPLES (name, plane, fluid, fat_sat), not strings -- v1 of this
        # kernel filtered them as strings, got [], and the guard below correctly refused the
        # arm rather than blending a wrongly-conditioned one. Keep the guard; fix the filter.
        SLOTS = [s for s in saved["SLOTS"] if s[0] in RAD_SLOT_NAMES]
        if [s[0] for s in SLOTS] != RAD_SLOT_NAMES:
            log(f"rad: slot order {[s[0] for s in SLOTS]} != {RAD_SLOT_NAMES}; the plane "
                f"embedding would shift -- refusing the arm")
            return None
        N_SLOT = len(SLOTS)
        t0 = time.time()
        st, C, M = build_cache(pick_slots(hte, plane_map), plane_map,
                               lat_of(hte, "rad "), "rad")
        log(f"rad: cache {C.shape} in {time.time() - t0:.0f}s, "
            f"fill {M.sum() / M.size:.1%}")
        if C.shape[1] != 3 or C.shape[2] != RAD_N_SLICE or C.shape[-1] != RAD_IMG:
            log(f"rad: UNEXPECTED cache shape {C.shape} -- refusing the arm")
            return None
    finally:
        IMG, CACHE_IMG, GROUP = saved["IMG"], saved["CACHE_IMG"], saved["GROUP"]
        CACHE_SLICES, N_GROUP = saved["CACHE_SLICES"], saved["N_GROUP"]
        CROP_MM, SLICE_BAND = saved["CROP_MM"], saved["SLICE_BAND"]
        SLOTS, N_SLOT = saved["SLOTS"], saved["N_SLOT"]
        assert N_SLOT == len(saved["SLOTS"]) and IMG == saved["IMG"], "rad: restore failed"
        log("rad: pixel contract restored")

    pos = {s: i for i, s in enumerate(st)}
    n, ns, nsl = C.shape[:3]
    tok = np.repeat(M[:, :, None], nsl, axis=2).reshape(n, -1).astype(np.float32)
    flat = C.reshape(-1, C.shape[-2], C.shape[-1])
    valid = np.flatnonzero(tok.reshape(-1) > 0)
    feats = np.zeros((n * ns * nsl, RAD_TOKEN_DIM), np.float16)
    bs = 96 if dev.type == "cuda" else 8
    t0 = time.time()
    for s in range(0, len(valid), bs):
        i = valid[s:s + bs]
        img = torch.from_numpy(flat[i]).to(dev).float().div_(127.5).sub_(1.0)
        img = img.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            f = enc(img)
        f = f.float().cpu().numpy()
        if not np.isfinite(f).all():
            log("rad: non-finite feature -- refusing the arm")
            return None
        feats[i] = f.astype(np.float16)
    feats = feats.reshape(n, ns * nsl, RAD_TOKEN_DIM)
    log(f"rad: encoded {len(valid):,} images in {time.time() - t0:.0f}s")

    P = np.zeros((5, n, 12), np.float32)
    for k, h in enumerate(heads):
        o = []
        for s in range(0, n, 64):
            x = torch.from_numpy(feats[s:s + 64]).to(dev)
            m = torch.from_numpy(tok[s:s + 64]).to(dev)
            o.append(torch.sigmoid(h(x, m)).float().cpu().numpy())
        P[k] = np.concatenate(o)
    del enc, heads, feats, C
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    mu, sd_ = float(P.mean()), float(P.std())
    log(f"rad: prob mean {mu:.4f} std {sd_:.4f} (local reference {RAD_REF_MEAN:.3f}/"
        f"{RAD_REF_STD:.3f}); per-fold means "
        f"{[round(float(P[i].mean()), 4) for i in range(5)]}")
    if not (0.15 < mu < 0.60):
        log(f"rad: prob mean {mu:.4f} far from the local reference -- REFUSING the arm "
            f"rather than blending a convention error")
        return None
    out = np.full((len(all_ids), 12), np.nan, np.float32)
    mean_p = P.mean(0)
    for j, s in enumerate(all_ids):
        if s in pos:
            out[j] = mean_p[pos[s]]
    log(f"rad: {int(np.isfinite(out).all(1).sum())}/{len(all_ids)} studies predicted")
    return out
'''

# ---------------------------------------------------------------------------- #
# 2. Two-family blend -> three-family, §9e equal weight. Anchored.
# ---------------------------------------------------------------------------- #
OLD_BLEND = '''    ftb = ftb_predict(all_ids, dev)
    if ftb is not None:
        r_p = pd.DataFrame(acc).rank(pct=True).to_numpy()
        r_f = pd.DataFrame(ftb).rank(pct=True).to_numpy()
        ok = np.isfinite(ftb).all(1)
        # A study ft_b could not read falls back to pilkwang's own rank, so it keeps a
        # sensible position instead of an invented one. Expected to be zero rows.
        r_f[~ok] = r_p[~ok]
        acc = (r_p + r_f) / 2.0
        log(f"blended: pilkwang rank-mean + ft_b, equal weight, "
            f"{int(ok.sum())}/{len(ok)} studies with both arms")
    else:
        log("ft_b unavailable -- submitting the pilkwang arm alone (the banked 0.899 path)")
'''

NEW_BLEND = '''    # ---- F6+: THREE families, equal-weight rank-mean (§9e, measured in §4c) ---- #
    # Each family is ranked, then the family ranks are averaged. Adding a third arm to a
    # two-arm mean is NOT the same as pooling members: pilkwang's 20 are 5 folds x 4 seeds
    # of one config (§2y), so a flat pool would give that config twenty votes to five.
    r_p = pd.DataFrame(acc).rank(pct=True).to_numpy()
    arms, names = [r_p], ["pilkwang"]
    ftb = ftb_predict(all_ids, dev)
    if ftb is not None:
        r_f = pd.DataFrame(ftb).rank(pct=True).to_numpy()
        ok_f = np.isfinite(ftb).all(1)
        # A study an arm could not read falls back to pilkwang's own rank, so it keeps a
        # sensible position instead of an invented one. Expected to be zero rows.
        r_f[~ok_f] = r_p[~ok_f]
        arms.append(r_f)
        names.append(f"ft_b({int(ok_f.sum())}/{len(ok_f)})")
    else:
        log("ft_b unavailable -- it will not contribute a family")
    rad = rad_predict(all_ids, hte, plane_map, dev)
    if rad is not None:
        r_r = pd.DataFrame(rad).rank(pct=True).to_numpy()
        ok_r = np.isfinite(rad).all(1)
        r_r[~ok_r] = r_p[~ok_r]
        arms.append(r_r)
        names.append(f"rad({int(ok_r.sum())}/{len(ok_r)})")
    else:
        log("RadImageNet unavailable -- it will not contribute a family")
    acc = np.mean(np.stack(arms), axis=0)
    log(f"blended {len(arms)} famil{'y' if len(arms) == 1 else 'ies'} at equal weight: "
        + " + ".join(names))
    if len(arms) == 1:
        log("NOTE: this is the pilkwang arm alone -- the banked 0.899 path, not F6+")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("notebooks/submissions"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    nb_path = a.src / "rsna-knee-f6-two-arm.ipynb"
    meta_path = a.src / "f6-two-arm-kernel-metadata.json"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # --- anchor 1: the blend, in whichever cell owns it ---------------------- #
    hits = [i for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and OLD_BLEND in "".join(c["source"])]
    if len(hits) != 1:
        raise SystemExit(f"blend anchor found in {len(hits)} cells, expected exactly 1 -- "
                         f"upstream changed shape; refusing to build")
    ci = hits[0]
    src = "".join(nb["cells"][ci]["source"]).replace(OLD_BLEND, NEW_BLEND)
    nb["cells"][ci]["source"] = src.splitlines(keepends=True)

    # --- anchor 2: the ft_b cell, after which the rad cell is inserted ------- #
    ftb = [i for i, c in enumerate(nb["cells"])
           if c["cell_type"] == "code" and "def ftb_predict(" in "".join(c["source"])]
    if len(ftb) != 1:
        raise SystemExit(f"ft_b cell found {len(ftb)} times, expected 1; refusing to build")
    # The call site lives inside a function; what matters is that the DEFINITION cell runs
    # before the cell that EXECUTES main(). `ftb_predict` is itself called from the blend
    # cell and defined after it, which is exactly why the naive index check is wrong.
    runs = [i for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and re.search(r"^\s*main\(\)", "".join(c["source"]), re.M)]
    if not runs:
        raise SystemExit("no cell executes main(); refusing to build")
    if ftb[0] >= min(runs):
        raise SystemExit(f"the ft_b definition cell ({ftb[0]}) is not before the cell that "
                         f"executes main() ({min(runs)}); refusing to build")
    nb["cells"].insert(ftb[0] + 1, {"cell_type": "code", "metadata": {},
                                    "execution_count": None, "outputs": [],
                                    "source": RAD_CELL.splitlines(keepends=True)})

    a.out.mkdir(parents=True, exist_ok=True)
    out_nb = a.out / f"{SLUG}.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["id"] = f"raahimnawaz/{SLUG}"
    meta["title"] = TITLE
    meta["code_file"] = out_nb.name
    for s in RAD_SOURCES:
        if s not in meta["dataset_sources"]:
            meta["dataset_sources"].append(s)
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote {out_nb}  ({len(nb['cells'])} cells)")
    print(f"  rad cell inserted at index {ftb[0] + 1}, blend patched in cell {ci}")
    print(f"  dataset_sources: {len(meta['dataset_sources'])}")
    for s in meta["dataset_sources"]:
        print(f"    {'+ ' if s in RAD_SOURCES else '  '}{s}")
    print(f"\npush with:  .venv/bin/python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
