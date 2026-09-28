"""Score studies straight from DICOMs with a `train_windows.py` checkpoint -- the submission path.

    python fusion/infer_windows.py --root <competition dir> --split test --weights last.pt \
        --out preds.csv                                   # every study in test.csv
    python fusion/infer_windows.py --root <competition dir> --split train --weights last.pt \
        --uids check.csv --out check.csv --check-against runs_ortho/oof_all.csv     # the gate

WHY IT IS BUILT FROM THE TRAINER'S OWN PARTS. The model scored 0.8298 on the 691 through
`train_windows.Windows` over a Stage A cache that `pipeline/build_slots64.build_study` wrote. This
file imports those same functions -- `build_study` for the stack, `window_centres` for which
windows, `WindowMIL` for the model -- so the only new code is the loop. `--check-against` then
proves it: studies the OOF already scored are rebuilt here from their DICOMs, in THIS environment,
and must reproduce the OOF probabilities. That one comparison covers the decode, the crop, the
series choice, the window set, the weights, the normalisation and the RoPE periods at once; any
of them wrong runs cleanly and scores wrongly (§9h), which is why it is a gate and not a log line.

A study that fails to build becomes an all-zero stack scored at the stack's middle -- what
`build_slots64._one` wrote into the cache for the same failure. More than --max-error-frac of
them stops the run: that is a systematic fault, and its output would only look like a model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))
import build_slots64 as B  # noqa: E402
import slots64 as S  # noqa: E402
from train_windows import LABELS, WindowMIL, thin_windows, window_centres, window_meta  # noqa: E402

SLOT_SIZES = tuple(s.n_slices for s in S.SLOTS)            # Stage A's; a checkpoint's last_meta.json overrides
# Ortho's own RoPE periods: timm's geometric series rounded to bf16 (read off the checkpoint
# 2026-09-24). `rope.periods` is a non-persistent buffer, so last.pt does not carry them.
ORTHO_PERIODS = [1.0, 1.335938, 1.78125, 2.375, 3.15625, 4.21875, 5.625, 7.5,
                 10.0, 13.3125, 17.75, 23.75, 31.625, 42.25, 56.25, 75.0]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def series_by_study(root: Path, split: str) -> dict[str, list[dict]]:
    """`build_slots64.main`'s series dicts, field for field."""
    ser = pd.read_csv(root / f"{split}_series.csv")
    return {u: [{"SeriesInstanceUID": str(r.SeriesInstanceUID), "plane": r.Anatomical_Plane,
                 "fluid": int(r.Fluid_Sensitive) if pd.notna(r.Fluid_Sensitive) else 0}
                for r in g.itertuples()] for u, g in ser.groupby(ser.StudyInstanceUID.astype(str))}


def load_meta(weights: Path) -> dict:
    """How the checkpoint was trained, read from the `last_meta.json` train_windows writes beside last.pt
    (since 09-24). Step 5's run predates it: absent, this is Stage A -- 64 x 336 as stored, no slot
    embedding, Ortho's own periods -- exactly what Submissions A and B ran and passed their gate on."""
    p = Path(weights).parent / "last_meta.json"
    m = json.loads(p.read_text()) if p.exists() else {}
    layout = str(m.get("layout", "stage_a"))
    spec = "c7" if layout.startswith("stage_b") else "c6"
    out = {"source": str(p) if p.exists() else "none (Stage A defaults)", "spec": spec,
           "canon": "canon True" in layout, "res": int(m.get("res", 336)), "img": int(m.get("img", 336)),
           "timm": m.get("timm", "vit_large_patch16_dinov3"), "pool": m.get("pool", "token"),
           "pos_emb": bool(m.get("pos_emb", False)), "n_slots": int(m.get("n_slots", 0)),
           "rope_periods": m.get("rope_periods"), "slot_sizes": tuple(m.get("slot_sizes") or SLOT_SIZES)}
    Sp = B.SPECS[spec]
    if tuple(s.n_slices for s in Sp.SLOTS) != out["slot_sizes"] or Sp.IMG != out["img"]:
        raise SystemExit(f"{p}: slots {out['slot_sizes']} img {out['img']} are not spec {spec}'s "
                         f"({tuple(s.n_slices for s in Sp.SLOTS)} x {Sp.IMG})")
    return out


class DicomStudies(torch.utils.data.Dataset):
    """One study -> (uid, uint8 windows [K,3,img,img], slot [K], depth [K], build error or '', build seconds)."""

    def __init__(self, img_dir: Path, uids: list[str], by_study: dict[str, list[dict]], n_windows: int = 0,
                 spec: str = "c6", canon: bool = False):
        self.img_dir, self.uids, self.by_study, self.n_windows = img_dir, uids, by_study, n_windows
        self.spec, self.canon = spec, canon
        self.slot_sizes = tuple(s.n_slices for s in B.SPECS[spec].SLOTS)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        uid, t = self.uids[i], time.time()
        Sp = B.SPECS[self.spec]
        try:
            vol, mask, _ = B.build_study(self.img_dir / uid, [dict(s) for s in self.by_study.get(uid, [])],
                                         spec=self.spec, canon=self.canon)
            err = ""
        except Exception as e:                              # noqa: BLE001 -- as build_slots64._one
            vol = np.zeros((Sp.N_SLICES, Sp.IMG, Sp.IMG), np.uint8)
            mask = np.zeros(Sp.N_SLICES, np.uint8)
            err = f"{type(e).__name__}: {e}"
        cs = window_centres(mask, self.slot_sizes) or [len(mask) // 2]     # Windows.__getitem__'s rule
        if self.n_windows:                                  # window_curve.thin's rule (identical on Stage A)
            cs = thin_windows(cs, self.slot_sizes, self.n_windows)
        w = np.stack([vol[c - 1:c + 2] for c in cs])
        slot, pos = window_meta(cs, self.slot_sizes)
        return (uid, torch.from_numpy(np.ascontiguousarray(w)), torch.from_numpy(slot), torch.from_numpy(pos),
                err, time.time() - t)


def build_model(weights: Path, dev: str, meta: dict | None = None) -> WindowMIL:
    import timm
    meta = meta or load_meta(weights)
    # As train_windows.build_encoder builds it: only the ViTs take img_size (their RoPE / patch grid is sized by it);
    # a CNN rejects it, and CoAtNet is built at its own 384.
    kw = dict(pretrained=False, num_classes=0, global_pool=meta["pool"])
    if meta["pool"] == "token":
        kw["img_size"] = meta["res"]
    enc = timm.create_model(meta["timm"], **kw)
    m = WindowMIL(enc, meta["res"], n_slots=meta["n_slots"] if meta["pos_emb"] else 0)
    m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    rope = getattr(m.encoder, "rope", None)
    if rope is None:                                        # CoAtNet, CNNs: no RoPE, nothing to restore
        if meta["rope_periods"]:
            raise SystemExit(f"{weights}: trained with RoPE periods, but {meta['timm']} has no RoPE -- wrong encoder")
        return m.to(dev).eval()
    # ⛔ RoPE periods are a NON-persistent buffer: last.pt does not carry them (§9h).
    want = meta["rope_periods"] or ORTHO_PERIODS
    with torch.no_grad():
        if meta["rope_periods"]:
            rope.periods.copy_(torch.tensor(want, dtype=rope.periods.dtype))
        else:                                               # Stage A: Ortho's are timm's rounded to bf16
            rope.periods.copy_(rope.periods.bfloat16().float())
    got = rope.periods.float().tolist()
    if len(got) != len(want) or max(abs(a - b) for a, b in zip(got, want)) > 1e-5:
        raise SystemExit(f"RoPE periods {got[:4]}... are not the trained ones {want[:4]}... -- the §9h trap")
    return m.to(dev).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="competition dir ({split}_series.csv)")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--uids", type=Path, help="csv with StudyInstanceUID; default: all of test.csv")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2, help="DICOM decode processes feeding the GPU")
    ap.add_argument("--n-windows", type=int, default=0,
                    help="thin to ~n windows per study (window_curve.thin); 0 = every window, as scored")
    ap.add_argument("--max-error-frac", type=float, default=0.02)
    ap.add_argument("--check-against", type=Path, help="oof_all.csv: the gate (see docstring)")
    ap.add_argument("--max-abs-diff", type=float, default=0.03)
    ap.add_argument("--mean-abs-diff", type=float, default=0.005)
    a = ap.parse_args()
    t0 = time.time()

    if a.uids:
        uids = pd.read_csv(a.uids).StudyInstanceUID.astype(str).drop_duplicates().tolist()
    else:
        uids = pd.read_csv(a.root / "test.csv").StudyInstanceUID.astype(str).drop_duplicates().tolist()
    uids = uids[a.shard::a.n_shards]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"{len(uids)} {a.split} studies (shard {a.shard}/{a.n_shards}) | device {dev}"
        + (f" {torch.cuda.get_device_name(0)}" if dev == "cuda" else "")
        + f" | torch {torch.__version__} | slots {SLOT_SIZES}")
    meta = load_meta(a.weights)
    model = build_model(a.weights, dev, meta)
    log(f"model loaded strict from {a.weights} ({time.time() - t0:.0f} s) | spec {meta['spec']} canon {meta['canon']} "
        f"res {meta['res']} slots {meta['slot_sizes']} pos_emb {meta['pos_emb']} | meta {meta['source']} | "
        + ("RoPE periods = the trained ones" if getattr(model.encoder, "rope", None) is not None else "no RoPE"))
    if a.n_windows and a.check_against:
        raise SystemExit("--check-against compares with the OOF, which read every window: drop --n-windows")
    ds = DicomStudies(a.root / f"{a.split}_series", uids, series_by_study(a.root, a.split), a.n_windows,
                      spec=meta["spec"], canon=meta["canon"])
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers,
                                     prefetch_factor=4 if a.workers else None)
    rows, errs, build_s, infer_s, nwin = [], {}, [], [], []
    with torch.no_grad():
        for k, (uid, w, sl, ps, err, sec) in enumerate(dl):
            t = time.time()
            x = w[None].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                p = torch.sigmoid(model(x, torch.ones(x.shape[:2], dtype=torch.bool, device=dev),
                                        slot=sl[None].to(dev), pos=ps[None].to(dev)).float())
            rows.append([uid] + p[0].cpu().numpy().tolist())
            infer_s.append(time.time() - t)
            build_s.append(sec)
            nwin.append(int(w.shape[0]))
            if err:
                errs[uid] = err
                log(f"  build error {uid}: {err}")
            if k % 100 == 0 or k == len(ds) - 1:
                log(f"  {k + 1}/{len(ds)}  build {np.mean(build_s):.2f} s/study/worker  "
                    f"infer {np.mean(infer_s):.2f} s/study  windows {np.median(nwin):.0f}  "
                    f"elapsed {(time.time() - t0) / 60:.1f} min  errors {len(errs)}")
    if len(errs) > a.max_error_frac * max(len(uids), 1) and len(errs) > 1:
        raise SystemExit(f"{len(errs)} of {len(uids)} studies failed to build -- systematic, refusing")

    out = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS)
    if out[LABELS].isna().any().any() or not np.isfinite(out[LABELS].to_numpy()).all():
        raise SystemExit("non-finite predictions")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    timing = {"n": len(uids), "errors": errs, "wall_min": (time.time() - t0) / 60,
              "build_sec_per_study_per_worker": float(np.mean(build_s)) if build_s else None,
              "infer_sec_per_study": float(np.mean(infer_s)) if infer_s else None,
              "windows_median": float(np.median(nwin)) if nwin else None, "workers": a.workers,
              "n_windows": a.n_windows or "all"}
    a.out.with_suffix(".timing.json").write_text(json.dumps(timing, indent=1))
    log(f"wrote {a.out}: {len(out)} studies in {timing['wall_min']:.1f} min")

    if a.check_against:
        ref = pd.read_csv(a.check_against).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        ref = ref.reindex(out.StudyInstanceUID)
        if ref[LABELS].isna().any().any():
            raise SystemExit("check: some rebuilt studies are not in the reference OOF")
        d = np.abs(out[LABELS].to_numpy() - ref[LABELS].to_numpy())
        verdict = {"n": len(out), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                   "per_study_max": dict(zip(out.StudyInstanceUID, d.max(1).round(5).tolist())),
                   "limits": {"max": a.max_abs_diff, "mean": a.mean_abs_diff}}
        a.out.with_suffix(".check.json").write_text(json.dumps(verdict, indent=1))
        log(f"CHECK vs {a.check_against.name}: {len(out)} studies rebuilt from DICOMs, "
            f"max |dp| {verdict['max_abs_diff']:.5f}, mean {verdict['mean_abs_diff']:.5f} "
            f"(limits {a.max_abs_diff} / {a.mean_abs_diff})")
        if verdict["max_abs_diff"] > a.max_abs_diff or verdict["mean_abs_diff"] > a.mean_abs_diff:
            raise SystemExit("CHECK FAILED: this path does not reproduce the model that was scored")
        log("CHECK PASSED: the DICOM path reproduces the scored OOF")


if __name__ == "__main__":
    main()
