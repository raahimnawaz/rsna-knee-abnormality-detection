"""Diff our training config against the config that PRODUCED the fork's weights.

    python fusion/contract_audit.py            # the table
    python fusion/contract_audit.py --strict   # exit 1 on any UNDECLARED divergence

WHY THIS FILE EXISTS, AND IT IS THE MOST EXPENSIVE LESSON IN THE REPO SO FAR. §2y recorded the
port at 0.7323 against the fork's 0.8434, called it "−0.1111, 15.4σ, 0/12 labels won", and closed
§2w step 4 on the strength of it: *"the port had genuine diversity to sell and is simply not good
enough to pay for a slot."* `train_port.py`'s own docstring calls the port "a 21st sample of the
same config".

**It was not the same config, and the true one shipped WITH the weights.**
`data/external/pilkwang_weights/manifest.json` carries, per member, the exact `config` and
`pixel_group` that produced a mean holdout of **0.8398** — and four fields disagree with ours:

    epochs      20/24/25/27/29/30/37/60 across the 20 members   vs  EPOCHS = 10
    backbone    facebook/dinov2-small (HF, NO registers)        vs  vit_small_patch14_reg4_dinov2
    prior       false                                           vs  SLOT_PRIOR_STRENGTH = 0.55
    slots       SAG_FLUID_FS/COR_FLUID_FS/AX_FLUID_FS/          vs  ax_fs/ax_nf/cor_fs/
                SAG_FLUID_NOFS/COR_T1/SAG_T1                        cor_nf/sag_fs/sag_nf

⛔ A COMPARISON IS ONLY AS GOOD AS ITS CONTRACT, AND NOBODY CHECKED THE CONTRACT. Each of those
four is documented *somewhere* — the slot divergence is in `train_port.py`'s docstring, the
backbone in `pilkwang_model.build_model`'s default — but they were never assembled in one place
and read against the manifest. §9h is the standing precedent: a wrong convention **runs perfectly
and scores wrongly**. This file is the cheap, repeatable guard that makes divergence #5 visible
before it costs another 15.4σ conclusion.

The audit imports the live constants rather than restating them, so it cannot drift from the code
it audits. DECLARED divergences are listed with a reason and pass; anything else fails --strict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
for p in ("fusion", "pipeline"):
    sys.path.insert(0, str(PROJ / p))

WEIGHTS = PROJ / "data" / "external" / "pilkwang_weights"

# Divergences we have decided to keep, each with the reason it is not a defect. Anything NOT in
# here that disagrees is a finding. Keeping this list short is the point: every entry is a place
# our reproduction is knowingly not a reproduction.
DECLARED: dict[str, str] = {
    # Measured, and too small to be the explanation: the port scored 0.7323 on `tiles336`
    # (SAGITTAL_LR=0, their raw order) and 0.7358 on `tiles336_lr1` (SAGITTAL_LR=1, K16 applied).
    # +0.0035 against a 0.11 gap. §2y calls their `order: normal` "signal they are leaving on the
    # floor"; keeping our bit is a deliberate improvement, not a reproduction defect.
    "rules.order": "K16 is ours by choice; measured at +0.0035 (0.7323 -> 0.7358), not the cause",
}


def theirs() -> tuple[dict, list[int]]:
    """The fork's own config, plus the per-member epoch counts the config dict does not carry."""
    m = json.loads((WEIGHTS / "manifest.json").read_text())["members"]
    cfgs = {json.dumps(x["config"], sort_keys=True) for x in m}
    if len(cfgs) != 1:
        raise SystemExit(f"the 20 members do not share one config ({len(cfgs)} distinct) -- "
                         f"this file assumes `distinct_configs: 1`, which §2y measured")
    return json.loads(cfgs.pop()), [x["epochs_done"] for x in m]


def ours() -> dict:
    """Live constants, imported from the modules that actually run."""
    import pilkwang_model as pk
    import pilkwang_pixels as px
    import slot_cache as sc
    import train_port as tp

    # ⛔ crop_mm MUST be read off the path train_port actually trains on, and the first version of
    # this file got that wrong. `pilkwang_pixels.CROP_MM` is 130.0 and would PASS -- but
    # train_port does not import pilkwang_pixels. It trains on `slot_cache` tiles, and for a
    # PROTOCOL slot `crop_box` has `box_mm is None` and returns the FULL grid, i.e. `pp.FOV_MM`.
    # Reading the constant from the faithful-but-unused module is how a false PASS gets printed
    # over a real divergence.
    import preprocess as pp
    port_crop = pp.FOV_MM if all(s.box_mm is None for s in sc.PROTOCOL) else None

    return {
        "img": sc.TILE,
        "slices": px.N_SLICE,
        "group": sc.GROUP,
        "crop_mm": port_crop,
        "band": list(px.SLICE_BAND),
        "slots": [s.name for s in sc.PROTOCOL],
        "backbone": tp.BACKBONE,
        "unfreeze_last": tp.UNFREEZE_LAST,
        "pool": "cls_mean",                      # pilkwang_model.Model default; POOL_PARTS key
        "prior": tp.SLOT_PRIOR_STRENGTH > 0,     # theirs is the bool `false`
        "epochs": tp.EPOCHS,
        "_slots_recovered": [s[0] for s in pk.SLOTS_RECOVERED],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any divergence is not in DECLARED")
    a = ap.parse_args()

    cfg, epochs = theirs()
    us = ours()

    # `slots` is compared by NAME SET and the names are from two different vocabularies, so a
    # literal comparison would report a difference that is partly just spelling. What actually
    # matters is arity and whether the fork's split is expressible in ours: theirs separates
    # fat-suppression from fluid-sensitivity (two sagittal fluid variants, no axial non-fluid);
    # ours is plane x fluid. FINDINGS.md 3.1 measured Fluid_Sensitive and Fat_Suppression
    # byte-identical over all 24,371 series, so ours CANNOT express theirs from competition
    # metadata alone. That is the divergence, not the naming.
    rows: list[tuple[str, object, object, bool]] = []
    for k in ("img", "slices", "group", "band", "unfreeze_last", "pool"):
        rows.append((k, cfg[k], us[k], cfg[k] == us[k]))

    # Reported with the mm/px it implies, because that is the quantity that matters and the
    # ratio hides it. slot_cache.py's own docstring line 27: "At 160 mm across 336 px a tear
    # line is about one pixel."
    rows.append((f"crop_mm  -> {cfg['crop_mm']/cfg['img']:.3f} vs {us['crop_mm']/cfg['img']:.3f} mm/px",
                 cfg["crop_mm"], us["crop_mm"], cfg["crop_mm"] == us["crop_mm"]))

    rows.append(("backbone", cfg["backbone"], us["backbone"], False if
                 "reg4" in str(us["backbone"]) else cfg["backbone"] in str(us["backbone"])))
    rows.append(("prior", cfg["prior"], us["prior"], cfg["prior"] == us["prior"]))

    ep_lo, ep_hi = min(epochs), max(epochs)
    rows.append((f"epochs ({ep_lo}-{ep_hi} over 20)", f"{ep_lo}-{ep_hi}", us["epochs"],
                 ep_lo <= us["epochs"] <= ep_hi))

    same_slots = cfg["slots"] == us["_slots_recovered"]
    rows.append(("slots (fork's, recovered)", "6", "6 recovered EXACT" if same_slots else "MISMATCH",
                 same_slots))
    rows.append(("slots (what train_port feeds)", ",".join(cfg["slots"]), ",".join(us["slots"]),
                 False))

    # The four `rules` are an enum in the fork's source (`RULES_NATIVE` / `RULES_LEGACY`, line 589
    # of the archived kernel) and every member records the NATIVE value. `pilkwang_pixels.py` is
    # transcribed from that pipeline and implements them -- but train_port trains on `slot_cache`
    # tiles, which were never audited against them. So these are UNVERIFIED, not PASS: claiming a
    # match here would repeat the crop_mm error one line up.
    for k, v in cfg.get("rules", {}).items():
        if k == "order":
            rows.append(("rules.order", v, "K16 direction applied (SAGITTAL_LR=1)", False))
        else:
            rows.append((f"rules.{k}", v, "native; slot_cache path UNVERIFIED", False))

    w = max(len(str(r[0])) for r in rows) + 1
    print(__doc__.splitlines()[0])
    print(f"\nmanifest: {WEIGHTS/'manifest.json'}   (20 members, mean holdout 0.8398)\n")
    print(f"  {'field':<{w}} {'FORK (scores 0.8398)':<46} {'OURS':<40} ")
    print("  " + "-" * (w + 90))
    bad = []
    for name, t, o, ok in rows:
        mark = "  PASS" if ok else ("  DECL" if name in DECLARED else "  DIFF")
        if not ok and name not in DECLARED:
            bad.append(name)
        print(f"  {name:<{w}} {str(t):<46} {str(o):<40} {mark}")

    print()
    if bad:
        print(f"  {len(bad)} UNDECLARED DIVERGENCE(S): {', '.join(bad)}")
        print("  Each one is a way our run is not the run that produced 0.8398.")
    else:
        print("  No undeclared divergences: our config matches the one that produced 0.8398.")
    if a.strict and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
