"""Test the mirror: geometry against the real DICOM path, the target swap, the two-pass prediction, the trainer.

    python notebooks/mirror_fixture.py --out <dir>            # all checks (~2 min on a laptop CPU)
    python notebooks/mirror_fixture.py --out <dir> --no-train # skip the end-to-end trainer run

1. GEOMETRY. Two synthetic knees go through `build_slots64.py --canon`: one as it is, one whose anatomy is reflected
   medial <-> lateral (same side, same scanner geometry). `train_windows.mirror_windows` of the first must be the
   second, window for window, in every plane, at the mirrored depth. Controls: the two differ without the mirror,
   and a mirror that flips a sagittal slice in-plane (instead of reading the slot backwards) fails.
2. TARGETS. The medial / lateral pairs swap, MCL becomes unknown, the other seven do not move.
3. PREDICTION. On a small random model: the two-pass mean of a mirrored knee is the two-pass mean of the knee with
   its medial / lateral columns swapped; the trainer's batched `predict` equals the submission path's arithmetic.
4. DATASET. `Windows(mirror=0)` is unchanged; `mirror=1` is `mirror_windows` + `mirror_targets` of it; `mirror=0.5`
   mirrors about half of the draws.
5. TRAINER. The whole recipe's flags on a tiny conv encoder and random stacks: it trains, writes the two-pass
   predictions beside the plain pass, the averaged weights, and a meta that tells the submission path to mirror.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJ / "fusion"), str(PROJ / "pipeline"), str(PROJ / "notebooks")]
import canon_fixture as CF  # noqa: E402
import train_windows as TW  # noqa: E402

L = TW.LABELS
STUDIES = {"R": ("R", -150.0, "R", CF.STD), "R_mirrored": ("R", -150.0, "R", CF.STD, True),
           "L": ("L", 80.0, "L", CF.STD), "L_mirrored": ("L", 80.0, None, CF.REV, True)}


def geometry(out: Path) -> int:
    root = out / "input"
    truth = CF.write(root, STUDIES, mm=0.7)
    CF.build(root, out / "canon", canon=True, spec="c7")
    corpus = TW.Corpus(out / "canon", "stage_b")
    sag = TW.slot_is_sagittal(corpus)
    planes = [m["plane"] for m in corpus.slot_meta]
    ds = TW.Windows(corpus, list(corpus.ids), np.zeros((len(corpus.ids), 12), np.float32), k=None)
    got = {truth[u]["name"]: ds[i] for i, u in enumerate(corpus.ids)}
    n = 0

    def worst(a, b):
        """Largest pixel difference between window sets a and b, matched on (slot, depth); per plane."""
        wa, sa, pa = a
        wb, sb, pb = b
        assert wa.shape == wb.shape and len(wa) > 40, (wa.shape, wb.shape)
        by = {(int(s), round(float(p), 5)): k for k, (s, p) in enumerate(zip(sb, pb))}
        assert len(by) == len(wb), "two windows share a slot and depth"
        d = {}
        for k, (s, p) in enumerate(zip(sa, pa)):
            j = by[(int(s), round(float(p), 5))]            # KeyError = a depth the other knee does not have
            d[planes[int(s)]] = max(d.get(planes[int(s)], 0), int((wa[k].int() - wb[j].int()).abs().max()))
        return d

    for side in ("R", "L"):
        w, s, p, _ = got[side]
        wm, pm = TW.mirror_windows(w, s, p, sag)
        d = worst((wm, s, pm), got[f"{side}_mirrored"][:3])
        assert set(d) == {"Sagittal", "Coronal", "Axial"} and max(d.values()) <= 1, f"{side}: mirror != mirrored knee {d}"
        n += 3
        # the check can fail: without the mirror the two knees differ in every plane
        d0 = worst((w, s, p), got[f"{side}_mirrored"][:3])
        assert min(d0.values()) > 50, f"{side}: the unmirrored knee already matches {d0} -- the check is blind"
        # a plausible wrong mirror: flip the sagittal slice left-right instead of reading the slot backwards
        wrong = torch.where(sag[s][:, None, None, None], w.flip(-1), w.flip(-1))
        dw = worst((wrong, s, p), got[f"{side}_mirrored"][:3])
        assert dw["Sagittal"] > 50 and dw["Coronal"] <= 1, f"{side}: an in-plane sagittal flip passed {dw}"
        # and the depth matters: the right pixels at the unmirrored depth are the wrong windows
        dd = worst((wm, s, p), got[f"{side}_mirrored"][:3])
        assert dd["Sagittal"] > 50, f"{side}: sagittal depth not mirrored, and nothing noticed {dd}"
        n += 3
        # twice is the identity
        w2, p2 = TW.mirror_windows(wm, s, pm, sag)
        assert torch.equal(w2, w) and torch.allclose(p2, p, atol=1e-6)
        n += 1
    assert worst(got["R"][:3], got["L"][:3]) and max(worst(got["R"][:3], got["L"][:3]).values()) <= 1   # canon holds here too
    print(f"1 GEOMETRY PASSED ({n + 1} checks): mirror_windows(knee) == the knee with mirrored anatomy built from "
          "DICOMs, sagittal / coronal / axial, right and left, at the mirrored depth; an in-plane sagittal flip and an "
          "unmirrored depth both fail")
    return n + 1


def targets() -> int:
    y = np.arange(12, dtype=np.float32) / 20 + 0.1
    m = TW.mirror_targets(y)
    i = {k: L.index(k) for k in L}
    assert m[i["Medial Meniscus"]] == y[i["Lateral Meniscus"]] and m[i["Lateral Meniscus"]] == y[i["Medial Meniscus"]]
    assert m[i["Medial OA"]] == y[i["Lateral OA"]] and m[i["Lateral OA"]] == y[i["Medial OA"]]
    assert np.isnan(m[i["MCL"]]) and not np.isnan(y).any(), "MCL must be unknown on the mirrored knee, the input untouched"
    same = ["ACL", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
    assert all(m[i[k]] == y[i[k]] for k in same) and len(same) + 5 == 12
    back = TW.mirror_targets(m)
    keep = [k for k in range(12) if k != i["MCL"]]
    assert np.array_equal(back[keep], y[keep])
    yn = y.copy()
    yn[i["Lateral OA"]] = np.nan                            # a masked cell travels with its finding
    assert np.isnan(TW.mirror_targets(yn)[i["Medial OA"]])
    p, q = np.random.default_rng(0).random((2, 5, 12)).astype(np.float32)
    a = TW.mirror_average(p, q)
    assert np.array_equal(a[:, i["MCL"]], p[:, i["MCL"]])
    assert np.allclose(a[:, i["Medial OA"]], 0.5 * (p[:, i["Medial OA"]] + q[:, i["Lateral OA"]]))
    assert np.allclose(a[:, i["ACL"]], 0.5 * (p[:, i["ACL"]] + q[:, i["ACL"]]))
    assert np.allclose(TW.mirror_average(p, p[:, TW.MIRROR_PERM]), p)      # a mirror-consistent model: unchanged
    print("2 TARGETS PASSED (12 checks): Medial <-> Lateral Meniscus and OA swap, MCL unknown, seven unmoved; the "
          "two-pass mean reads the mirrored pass back into the plain columns, MCL from the plain pass")
    return 12


def prediction(out: Path) -> int:
    import timm
    torch.manual_seed(0)
    corpus = TW.Corpus(out / "canon", "stage_b")
    sag = TW.slot_is_sagittal(corpus)
    enc = timm.create_model("convnext_atto", pretrained=False, num_classes=0, global_pool="avg")
    model = TW.WindowMIL(enc, 64, n_slots=len(corpus.slot_sizes), spatial=True).eval()
    with torch.no_grad():                                   # the zero-initialised parts would hide a slot / depth slip
        for t in (model.slot_emb.weight, model.depth_proj.weight, model.depth_proj.bias, model.sq, model.w):
            t.normal_(0, 0.5)
    uids = list(corpus.ids)
    ds = TW.Windows(corpus, uids, np.zeros((len(uids), 12), np.float32), k=None)
    os.environ["TW_WORKERS"] = "0"
    raw = TW.predict(model, ds, "cpu", False, 2, sag)       # the trainer's path: batches of 2, padded and masked
    assert raw.shape == (len(uids), 2, 12)
    tta = TW.mirror_average(raw[:, 0], raw[:, 1])
    keep = [k for k in range(12) if k != TW.MIRROR_UNKNOWN]
    n = 0
    with torch.no_grad():
        for i, u in enumerate(uids):
            w, s, p, _ = ds[i]
            x, ones = w[None], torch.ones((1, len(w)), dtype=torch.bool)
            f = lambda xx, pp: torch.sigmoid(model(xx, ones, slot=s[None], pos=pp[None]).float())      # noqa: E731
            xm, pm = TW.mirror_windows(x, s[None], p[None], sag)
            one = TW.mirror_average(f(x, p), f(xm, pm[0]))[0].numpy()        # infer_windows.py's arithmetic
            assert np.abs(one - tta[i]).max() < 1e-5, f"trainer predict != submission arithmetic: {np.abs(one - tta[i]).max()}"
            # the mirrored knee as input: its two-pass mean is this knee's with the medial / lateral columns swapped
            xmm, pmm = TW.mirror_windows(xm, s[None], pm, sag)
            flipped = TW.mirror_average(f(xm, pm[0]), f(xmm, pmm[0]))[0].numpy()
            assert np.abs(flipped[TW.MIRROR_PERM][keep] - one[keep]).max() < 1e-5
            assert np.abs(raw[i, 0] - raw[i, 1][TW.MIRROR_PERM])[keep].max() > 1e-3, "the mirrored pass changed nothing"
            n += 3
    print(f"3 PREDICTION PASSED ({n} checks): the trainer's batched two-pass predict equals the submission path's "
          "arithmetic (< 1e-5); mirroring the input swaps the medial / lateral columns of the mean and nothing else")
    return n


def dataset(out: Path) -> int:
    corpus = TW.Corpus(out / "canon", "stage_b")
    sag = TW.slot_is_sagittal(corpus)
    uids = list(corpus.ids)
    y = np.random.default_rng(1).random((len(uids), 12)).astype(np.float32)
    for i in range(len(uids)):
        w0, s0, p0, y0 = TW.Windows(corpus, uids, y, k=24, seed=7)[i]
        w1, s1, p1, y1 = TW.Windows(corpus, uids, y, k=24, seed=7, mirror=1.0)[i]
        wm, pm = TW.mirror_windows(w0, s0, p0, sag)
        assert torch.equal(w1, wm) and torch.equal(s1, s0) and torch.allclose(p1, pm)
        assert torch.equal(torch.nan_to_num(y1, nan=-1), torch.nan_to_num(torch.from_numpy(TW.mirror_targets(y[i])), nan=-1))
        assert torch.equal(y0, torch.from_numpy(y[i])) and w0.shape == (24, 3, 384, 384)
    half = TW.Windows(corpus, uids, y, k=2, seed=3, mirror=0.5)
    share = np.mean([bool(torch.isnan(half[k % len(uids)][3][TW.MIRROR_UNKNOWN])) for k in range(400)])
    assert 0.4 < share < 0.6, f"mirror=0.5 mirrored {share:.2f} of 400 draws"
    assert not np.isnan(y).any(), "the dataset wrote into the targets it was given"
    print(f"4 DATASET PASSED ({3 * len(uids) + 2} checks): mirror=0 is the old sample, mirror=1 is its mirror with "
          f"swapped targets, mirror=0.5 mirrored {share:.2f} of 400 draws; the target table is not written to")
    return 3 * len(uids) + 2


def trainer(out: Path) -> int:
    run = out / "run"
    cmd = [sys.executable, str(PROJ / "fusion" / "train_windows.py"), "--arch", "cnxatto", "--synthetic", "24",
           "--out", str(run), "--epochs", "4", "--limit-train", "8", "--bs", "2", "--micro", "1", "--eval-bs", "2",
           "--res", "64", "--k", "24", "--spatial-attn", "--pos-emb", "--conf-weight", "none", "--allow-missing-labels",
           "--mirror", "0.5", "--mirror-tta", "--save-weights"]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "TW_WORKERS": "0"})
    if r.returncode:
        raise SystemExit("trainer failed:\n" + (r.stdout + r.stderr)[-3000:])
    s = json.loads((run / "summary.json").read_text())
    meta = json.loads((run / "avg3" / "last_meta.json").read_text())
    assert s["epochs_done"] == 4 and s["k"] == 24 and s["spatial_attn"] and s["mirror"] == 0.5 and s["mirror_tta"]
    assert all(np.isfinite(h["loss"]) for h in s["hist"])
    assert meta["mirror_tta"] is True and meta["spatial"] is True and meta["mirror"] == 0.5
    n, mcl = 4, L.index("MCL")
    for d in (run, run / "avg3"):
        a, p = pd.read_csv(d / "oof_all.csv"), pd.read_csv(d / "oof_plain.csv")
        assert list(a.StudyInstanceUID) == list(p.StudyInstanceUID) and len(a) == 8
        assert np.allclose(a["MCL"], p["MCL"]), "MCL must be the plain pass"
        assert (a[L].to_numpy() != p[L].to_numpy())[:, [k for k in range(12) if k != mcl]].any(), "no second pass in oof_all"
        assert (d / "last.pt").exists()
        n += 4
    for bad, why in ((["--mirror", "0.5"], "without --allow-missing-labels"), ):
        r2 = subprocess.run([sys.executable, str(PROJ / "fusion" / "train_windows.py"), "--arch", "cnxatto",
                             "--synthetic", "6", "--out", str(out / "bad"), "--epochs", "1"] + bad,
                            capture_output=True, text=True)
        assert r2.returncode and "allow-missing-labels" in (r2.stdout + r2.stderr), f"--mirror {why} was accepted"
        n += 1
    print(f"5 TRAINER PASSED ({n} checks): k 24 + spatial attention + slot/depth embedding + mirror 0.5 + two-pass "
          "predictions ran 4 epochs; oof_all = the two-pass mean (MCL = plain), oof_plain beside it, avg3 weights and a "
          "meta with mirror_tta; --mirror without masked targets is refused")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-train", action="store_true")
    a = ap.parse_args()
    n = geometry(a.out) + targets() + prediction(a.out) + dataset(a.out)
    if not a.no_train:
        n += trainer(a.out)
    print(f"MIRROR FIXTURE PASSED: {n} checks")


if __name__ == "__main__":
    main()
