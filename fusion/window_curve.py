r"""Score one checkpoint at fewer windows per study and report its GPU cost (accuracy against runtime).

    python fusion/window_curve.py --corpus <cache> --weights runs_ortho/last.pt \
        --val runs_ortho/val_studies.csv --out curve --n 54 27 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from infer_windows import build_model  # noqa: E402
from train_windows import LABELS, Corpus, window_centres  # noqa: E402

FULL = 54


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def thin(mask: np.ndarray, slot_sizes, n: int) -> list[int]:
    """Evenly spaced within each slot, each slot keeping its share of `n` out of FULL."""
    cs = window_centres(mask, slot_sizes)
    if not cs:
        return [len(mask) // 2]
    if n >= FULL:
        return cs
    out, o = [], 0
    for size in slot_sizes:
        mine = [c for c in cs if o <= c < o + size]
        if mine:
            k = max(1, int(round(len(mine) * n / FULL)))
            idx = np.unique(np.round(np.linspace(0, len(mine) - 1, k)).astype(int))
            out += [mine[i] for i in idx]
        o += size
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True, help="csv with StudyInstanceUID (the 691)")
    ap.add_argument("--reference", type=Path, help="oof_all.csv the n=54 pass must reproduce")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = Corpus(a.corpus, "stage_a")
    uids = pd.read_csv(a.val).StudyInstanceUID.astype(str).drop_duplicates().tolist()
    uids = [u for u in uids if u in corpus.row]
    if a.limit:
        uids = uids[: a.limit]
    model = build_model(a.weights, dev)
    log(f"{len(uids)} studies | n {a.n} | device {dev}"
        + (f" {torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))

    # warm-up so the first study's timing is not cuDNN/allocator start-up
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        vol, mask = corpus.study(uids[0])
        model(torch.from_numpy(np.stack([vol[c - 1:c + 2] for c in thin(mask, corpus.slot_sizes, FULL)]))[None].to(dev))
    summary = {}
    for n in a.n:
        rows, fwd, step, nwin = [], [], [], []
        for u in uids:
            t0 = time.time()
            vol, mask = corpus.study(u)
            w = np.stack([vol[c - 1:c + 2] for c in thin(mask, corpus.slot_sizes, n)])
            x = torch.from_numpy(np.ascontiguousarray(w))[None].to(dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                p = torch.sigmoid(model(x).float())[0].cpu().numpy()
            if dev == "cuda":
                torch.cuda.synchronize()
            fwd.append(time.time() - t1)
            step.append(time.time() - t0)
            nwin.append(len(w))
            rows.append([u] + p.tolist())
        d = a.out / f"n{n}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS).to_csv(d / "oof_all.csv", index=False)
        summary[n] = {"studies": len(uids), "windows_mean": float(np.mean(nwin)),
                      "windows_median": float(np.median(nwin)),
                      "forward_sec_per_study": float(np.mean(fwd)), "step_sec_per_study": float(np.mean(step)),
                      "device": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
                      "hidden_1322_forward_min_this_device": float(np.mean(fwd)) * 1322 / 60}
        log(f"n={n}: {summary[n]['windows_mean']:.1f} windows/study, forward {summary[n]['forward_sec_per_study']:.3f} s, "
            f"step {summary[n]['step_sec_per_study']:.3f} s -> 1,322 studies {summary[n]['hidden_1322_forward_min_this_device']:.1f} min on one {summary[n]['device']}")
        if n >= FULL and a.reference:
            ref = pd.read_csv(a.reference).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
            got = pd.read_csv(d / "oof_all.csv").set_index("StudyInstanceUID")
            diff = np.abs(got[LABELS].to_numpy() - ref.reindex(got.index)[LABELS].to_numpy())
            summary[n]["reproduces_reference_max_abs_diff"] = float(np.nanmax(diff))
            log(f"n={n} vs the scored OOF: max |dp| {np.nanmax(diff):.5f}, mean {np.nanmean(diff):.5f}")
            if np.nanmax(diff) > 0.03 or np.nanmean(diff) > 0.005:
                raise SystemExit("the full-window pass does not reproduce the scored OOF -- the curve would be off another path")
    (a.out / f"summary_{'_'.join(map(str, a.n))}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
