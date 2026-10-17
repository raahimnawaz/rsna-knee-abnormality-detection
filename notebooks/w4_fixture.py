"""Checks for W4's two additions to the trainer (PLAN.md): the loss focus on the weak four and the per-finding
location bias, including the round trip through the submission loader.

    python notebooks/w4_fixture.py          # ~10 s on a laptop CPU
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJ / "fusion"), str(PROJ / "pipeline")]
import infer_windows as IW  # noqa: E402
import train_windows as TW  # noqa: E402


def model(loc_bins: int, seed: int = 0):
    import timm
    torch.manual_seed(seed)
    enc = timm.create_model("convnext_atto", pretrained=False, num_classes=0)
    return TW.WindowMIL(enc, 64, n_slots=7, spatial=True, loc_slots=7 if loc_bins else 0, loc_bins=loc_bins).eval()


def batch(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 255, (2, 10, 3, 64, 64), generator=g, dtype=torch.uint8)
    slot = torch.randint(0, 7, (2, 10), generator=g)
    pos = torch.rand(2, 10, generator=g)
    return x, slot, pos


def main() -> int:
    n = 0
    # 1. focus: 3 on exactly the weak four, 1 elsewhere
    w = TW.focus_weights(3.0)
    assert sorted(f for f, v in zip(TW.LABELS, w) if v == 3.0) == sorted(TW.WEAK4) and (w == 1.0).sum() == 8
    assert np.all(TW.focus_weights(1.0) == 1.0)
    n += 2
    # 2. zero-initialised location bias: identical to the model without it
    m0, m1 = model(0), model(8)
    missing = m1.load_state_dict(m0.state_dict(), strict=False)
    assert missing.missing_keys == ["loc"] and not missing.unexpected_keys, missing
    x, slot, pos = batch()
    with torch.no_grad():
        assert torch.allclose(m0(x, slot=slot, pos=pos), m1(x, slot=slot, pos=pos), atol=1e-6)
    n += 1
    # 3. a bias on one finding moves that finding only, and its attention goes where the bias points
    f = TW.LABELS.index("Lateral OA")
    s0, b0 = int(slot[0, 0]), int((pos[0, 0] * 8).long().clamp_max(7))       # a (slot, bin) the batch has
    with torch.no_grad():
        m1.loc[f, s0, b0] = 30.0
        out = m1(x, slot=slot, pos=pos)
        base = m0(x, slot=slot, pos=pos)
    other = [i for i in range(12) if i != f]
    assert torch.allclose(out[:, other], base[:, other], atol=1e-6) and (out[:, f] - base[:, f]).abs().max() > 1e-4
    n += 1
    # 4. bins mirror: depth p and 1 - p land in bins b and B-1-b (mirror_windows reads a sagittal slot backwards)
    p = torch.rand(10000) * 0.98 + 0.01
    B = 8
    b1 = (p * B).long().clamp_max(B - 1)
    b2 = ((1 - p) * B).long().clamp_max(B - 1)
    away = ((p * B) - (p * B).round()).abs() > 1e-4                         # not on a bin edge
    assert torch.equal(b2[away], (B - 1 - b1)[away])
    n += 1
    # 5. the loss: the weak four's gradient is focus x the others' for equal errors
    logits = torch.zeros(1, 12, requires_grad=True)
    y = torch.ones(1, 12)
    (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * torch.from_numpy(TW.focus_weights(3.0))).mean().backward()
    g = logits.grad[0].abs()
    assert torch.allclose(g[[TW.LABELS.index(k) for k in TW.WEAK4]], 3 * g[TW.LABELS.index("ACL")])
    n += 1
    # 6. the submission loader rebuilds a W4 checkpoint from its last_meta.json, outputs identical
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        torch.save(m1.state_dict(), d / "last.pt")
        (d / "last_meta.json").write_text(json.dumps({
            "layout": "stage_b span [0.02, 0.98] img 384 canon True", "res": 64, "img": 384, "timm": "convnext_atto",
            "pool": "avg", "pos_emb": True, "n_slots": 7, "spatial": True, "loc_slots": 7, "loc_bins": 8,
            "slot_sizes": [26, 20, 16, 10, 14, 6, 4]}))
        meta = IW.load_meta(d / "last.pt")
        assert meta["loc_bins"] == 8 and meta["loc_slots"] == 7
        back = IW.build_model(d / "last.pt", "cpu", meta)
        with torch.no_grad():
            assert torch.allclose(back(x, slot=slot, pos=pos), out, atol=1e-6)
        # an old checkpoint (no loc keys in its meta) still loads as before
        (d / "last_meta.json").write_text(json.dumps({**json.loads((d / "last_meta.json").read_text()),
                                                      "loc_slots": 0, "loc_bins": 0}))
        torch.save(m0.state_dict(), d / "last.pt")
        old = IW.build_model(d / "last.pt", "cpu", IW.load_meta(d / "last.pt"))
        assert not hasattr(old, "loc")
    n += 2
    print(f"w4_fixture ok: {n} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
