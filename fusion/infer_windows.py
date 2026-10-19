r"""Score studies straight from DICOMs with a `train_windows.py` checkpoint: the submission path.

    python fusion/infer_windows.py --root <competition dir> --split test --weights last.pt --out preds.csv
    python fusion/infer_windows.py --root <competition dir> --split train --weights last.pt --uids check.csv \
        --out check.csv --check-against runs/oof_all.csv    # parity gate against the scored OOF
    python fusion/infer_windows.py --root <competition dir> --weights fold0/last.pt fold1/last.pt --out preds.csv
        # several checkpoints of one slice layout: each study is built once, the probabilities averaged
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
from train_windows import (LABELS, WindowMIL, mirror_average, mirror_windows, thin_windows,  # noqa: E402
                           window_centres, window_meta)



def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def series_by_study(root: Path, split: str) -> dict[str, list[dict]]:
    """`build_slots64.main`'s series dicts, field for field."""
    ser = pd.read_csv(root / f"{split}_series.csv")
    return {u: [{"SeriesInstanceUID": str(r.SeriesInstanceUID), "plane": r.Anatomical_Plane,
                 "fluid": int(r.Fluid_Sensitive) if pd.notna(r.Fluid_Sensitive) else 0}
                for r in g.itertuples()] for u, g in ser.groupby(ser.StudyInstanceUID.astype(str))}


def load_meta(weights: Path) -> dict:
    """How the checkpoint was trained, from the `last_meta.json` beside last.pt."""
    p = Path(weights).parent / "last_meta.json"
    if not p.exists():
        raise SystemExit(f"{p}: missing -- every checkpoint is served with the meta it was trained with")
    m = json.loads(p.read_text())
    layout = str(m["layout"])
    spec = "k4" if layout.startswith("stage_k") else "c7" if layout.startswith("stage_b") else "c6"
    out = {"source": str(p), "spec": spec, "canon": "canon True" in layout, "res": int(m["res"]),
           "img": int(m["img"]), "timm": m["timm"], "pool": m["pool"], "pos_emb": bool(m.get("pos_emb", False)),
           "n_slots": int(m.get("n_slots", 0)), "spatial": bool(m.get("spatial", False)),
           "mirror_tta": bool(m.get("mirror_tta", False)), "slot_sizes": tuple(m["slot_sizes"])}
    if m.get("loc_bins") or m.get("rope_periods"):
        raise SystemExit(f"{p}: a location bias or RoPE encoder -- not served by this reader")
    if spec == "k4":                                        # K: pipeline/kcache.py's region crops
        import kcache as K
        if out["slot_sizes"] != (K.PER,) * len(K.REGIONS) or out["img"] != K.IMG:
            raise SystemExit(f"{p}: slots {out['slot_sizes']} img {out['img']} are not K's")
        return out
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
        except Exception as e:                              # noqa: BLE001
            vol = np.zeros((Sp.N_SLICES, Sp.IMG, Sp.IMG), np.uint8)
            mask = np.zeros(Sp.N_SLICES, np.uint8)
            err = f"{type(e).__name__}: {e}"
        cs = window_centres(mask, self.slot_sizes) or [len(mask) // 2]     # Windows.__getitem__'s rule
        if self.n_windows:                                  # train_windows.thin_windows's rule
            cs = thin_windows(cs, self.slot_sizes, self.n_windows)
        w = np.stack([vol[c - 1:c + 2] for c in cs])
        slot, pos = window_meta(cs, self.slot_sizes)
        return (uid, torch.from_numpy(np.ascontiguousarray(w)), torch.from_numpy(slot), torch.from_numpy(pos),
                err, time.time() - t)


def _identity(b):
    return b


class KAnchorInputs(torch.utils.data.Dataset):
    """K, pass 1, in a decode worker: one study -> (uid, the segmenter's input or None, error or '', seconds)."""

    def __init__(self, img_dir: Path, uids: list[str], by_study: dict[str, list[dict]]):
        self.img_dir, self.uids, self.by_study = img_dir, uids, by_study

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        import kcache as K
        uid, t = self.uids[i], time.time()
        try:
            inp, err = K.study_anchor_inputs(self.img_dir / uid, [dict(s) for s in self.by_study.get(uid, [])]), ""
        except Exception as e:                              # noqa: BLE001
            inp, err = None, f"{type(e).__name__}: {e}"
        return uid, inp, err, time.time() - t


class KStudies(torch.utils.data.Dataset):
    """K, pass 2, in a decode worker: the regions cut around the anchors pass 1 found -> DicomStudies' tuple."""

    def __init__(self, img_dir: Path, uids: list[str], by_study: dict[str, list[dict]], anchors: dict):
        import kcache as K
        self.img_dir, self.uids, self.by_study, self.anchors = img_dir, uids, by_study, anchors
        self.slot_sizes = (K.PER,) * len(K.REGIONS)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        import kcache as K
        uid, t = self.uids[i], time.time()
        n = sum(self.slot_sizes)
        try:
            J, P = self.anchors.get(uid, (None, None))
            if J is None:                                   # no sagittal series: the cache build's empty row too
                vol, mask, err = np.zeros((n, K.IMG, K.IMG), np.uint8), np.zeros(n, np.uint8), "no anchor"
            else:
                vol, mask, _ = K.build_study(self.img_dir / uid, [dict(s) for s in self.by_study.get(uid, [])],
                                             override=(J, P))
                err = ""
        except Exception as e:                              # noqa: BLE001
            vol, mask, err = np.zeros((n, K.IMG, K.IMG), np.uint8), np.zeros(n, np.uint8), f"{type(e).__name__}: {e}"
        cs = window_centres(mask, self.slot_sizes) or [len(mask) // 2]     # Windows.__getitem__'s rule
        w = np.stack([vol[c - 1:c + 2] for c in cs])
        slot, pos = window_meta(cs, self.slot_sizes)
        return (uid, torch.from_numpy(np.ascontiguousarray(w)), torch.from_numpy(slot), torch.from_numpy(pos),
                err, time.time() - t)


def k_anchors(root: Path, split: str, uids: list[str], by_study: dict, seg_weights: str, seg_vendor: str,
              dev: str, workers: int) -> dict:
    """K, pass 1: the segmenter on the GPU, fed by decode workers -> {uid: (J, P) or (None, None)}."""
    import kcache as K
    seg = K.load_segmenter(seg_weights, seg_vendor, dev)
    dl = torch.utils.data.DataLoader(KAnchorInputs(root / f"{split}_series", uids, by_study), batch_size=None,
                                     shuffle=False, num_workers=workers, collate_fn=_identity,
                                     prefetch_factor=4 if workers else None)
    out, how, t0 = {}, {}, time.time()
    for k, (uid, inp, err, _) in enumerate(dl):
        if inp is None:
            out[uid], h = (None, None), f"none: {err or 'no sagittal series'}"[:60]
        else:
            J, P, h = K.anchors_from(seg, inp, dev)
            out[uid] = (J, P)
        how[h] = how.get(h, 0) + 1
        if k % 200 == 0 or k == len(uids) - 1:
            log(f"  K anchors {k + 1}/{len(uids)}  {(time.time() - t0) / 60:.1f} min")
    log(f"K anchors: {how}")
    del seg
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def build_model(weights: Path, dev: str, meta: dict | None = None) -> WindowMIL:
    import timm
    meta = meta or load_meta(weights)
    enc = timm.create_model(meta["timm"].split(".")[0], pretrained=False, num_classes=0, global_pool=meta["pool"])
    m = WindowMIL(enc, meta["res"], n_slots=meta["n_slots"] if meta["pos_emb"] else 0, spatial=meta["spatial"])
    m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    m = m.to(dev).eval()
    if dev == "cuda" and meta["timm"].startswith(("convnext", "resnet")):
        m.encoder.to(memory_format=torch.channels_last)
        m.channels_last = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="competition dir ({split}_series.csv)")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--uids", type=Path, help="csv with StudyInstanceUID; default: all of test.csv")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--weights", type=Path, nargs="+", required=True,
                    help="one checkpoint, or several of the same slice layout (their probabilities are averaged)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2, help="DICOM decode processes feeding the GPU")
    ap.add_argument("--n-windows", type=int, default=0,
                    help="thin to ~n windows per study (train_windows.thin_windows); 0 = every window, as scored")
    ap.add_argument("--max-error-frac", type=float, default=0.02)
    ap.add_argument("--seg-weights", help="K checkpoints: the SKM-TEA segmenter (heliosli's anatomy.pt)")
    ap.add_argument("--seg-vendor", help="K checkpoints: its vendored MONAI 0.6 (cloud_code/training/vendor_runtime/monai060)")
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
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True
    log(f"{len(uids)} {a.split} studies (shard {a.shard}/{a.n_shards}) | device {dev}"
        + (f" {torch.cuda.get_device_name(0)}" if dev == "cuda" else "")
        + f" | torch {torch.__version__}")
    metas = [load_meta(w) for w in a.weights]
    meta = metas[0]
    same = ("spec", "canon", "slot_sizes", "img")
    if any(tuple(m[k] for k in same) != tuple(meta[k] for k in same) for m in metas):
        raise SystemExit(f"--weights: the checkpoints were trained on different slice layouts {[[m[k] for k in same] for m in metas]}")
    if a.check_against and len(a.weights) > 1:
        raise SystemExit("--check-against reproduces ONE model's OOF: pass one checkpoint")
    models = []
    for w, m in zip(a.weights, metas):
        models.append(build_model(w, dev, m))
        log(f"model loaded strict from {w} ({time.time() - t0:.0f} s) | {m['timm']} | spec {m['spec']} canon {m['canon']} "
            f"res {m['res']} slots {m['slot_sizes']} pos_emb {m['pos_emb']} | meta {m['source']}")
    log(f"{len(models)} checkpoint(s): probabilities averaged with equal weight")
    if any(m["mirror_tta"] for m in metas):
        if not meta["canon"]:
            raise SystemExit("a checkpoint asks for mirror averaging, but its cache is not canonical")
        log(f"mirror averaging (plain + mirrored pass, MCL plain only) for {sum(m['mirror_tta'] for m in metas)} "
            f"of {len(metas)} checkpoint(s), as each was scored")
    if meta["spec"] == "k4":
        import kcache as K
        sagittal = torch.tensor([r[1] == "Sagittal" for r in K.REGIONS], device=dev)
    else:
        sagittal = torch.tensor([s_.plane == "Sagittal" for s_ in B.SPECS[meta["spec"]].SLOTS], device=dev)
    if a.n_windows and a.check_against:
        raise SystemExit("--check-against compares with the OOF, which read every window: drop --n-windows")
    if meta["spec"] == "k4":
        if not (a.seg_weights and a.seg_vendor):
            raise SystemExit("a K checkpoint needs --seg-weights and --seg-vendor (the segmenter that finds the knee)")
        if a.n_windows:
            raise SystemExit("K reads every window (24); drop --n-windows")
        by = series_by_study(a.root, a.split)
        anchors = k_anchors(a.root, a.split, uids, by, a.seg_weights, a.seg_vendor, dev, a.workers)
        ds = KStudies(a.root / f"{a.split}_series", uids, by, anchors)
    else:
        ds = DicomStudies(a.root / f"{a.split}_series", uids, series_by_study(a.root, a.split), a.n_windows,
                          spec=meta["spec"], canon=meta["canon"])
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers,
                                     prefetch_factor=4 if a.workers else None, pin_memory=dev == "cuda")
    uids_out, preds, errs, build_s, nwin = [], [], {}, [], []
    t_loop = time.time()
    with torch.inference_mode():
        for k, (uid, w, sl, ps, err, sec) in enumerate(dl):
            x = w[None].to(dev, non_blocking=True)
            sl_d, ps_d = sl[None].to(dev, non_blocking=True), ps[None].to(dev, non_blocking=True)
            ones = torch.ones(x.shape[:2], dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                ps_ = [torch.sigmoid(m(x, ones, slot=sl_d, pos=ps_d).float()) for m in models]
                if any(mt["mirror_tta"] for mt in metas):   # `train_windows.predict`'s two passes, per checkpoint
                    xm, pm = mirror_windows(x, sl_d, ps_d, sagittal)
                    ps_ = [mirror_average(q, torch.sigmoid(m(xm, ones, slot=sl_d, pos=pm).float()))
                           if mt["mirror_tta"] else q for q, m, mt in zip(ps_, models, metas)]
                p = torch.stack(ps_).mean(0)
            uids_out.append(uid)
            preds.append(p[0].float())
            build_s.append(sec)
            nwin.append(int(w.shape[0]))
            if err:
                errs[uid] = err
                log(f"  build error {uid}: {err}")
            if k % 100 == 0 or k == len(ds) - 1:
                log(f"  {k + 1}/{len(ds)}  build {np.mean(build_s):.2f} s/study/worker  "
                    f"{(time.time() - t_loop) / (k + 1):.2f} s/study  windows {np.median(nwin):.0f}  "
                    f"elapsed {(time.time() - t0) / 60:.1f} min  errors {len(errs)}")
    probs = torch.stack(preds).cpu().numpy() if preds else np.zeros((0, len(LABELS)), np.float32)
    rows = [[u] + r.tolist() for u, r in zip(uids_out, probs)]
    if len(errs) > a.max_error_frac * max(len(uids), 1) and len(errs) > 1:
        raise SystemExit(f"{len(errs)} of {len(uids)} studies failed to build -- systematic, refusing")

    out = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS)
    if out[LABELS].isna().any().any() or not np.isfinite(out[LABELS].to_numpy()).all():
        raise SystemExit("non-finite predictions")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    timing = {"n": len(uids), "errors": errs, "wall_min": (time.time() - t0) / 60,
              "build_sec_per_study_per_worker": float(np.mean(build_s)) if build_s else None,
              "infer_sec_per_study": (time.time() - t_loop) / max(len(uids_out), 1),
              "windows_median": float(np.median(nwin)) if nwin else None, "workers": a.workers,
              "checkpoints": [str(w) for w in a.weights],
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
