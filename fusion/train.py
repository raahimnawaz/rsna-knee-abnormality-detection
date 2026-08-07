"""Train the fusion head on cached DINOv2 features. Runs on the M5; MPS, CUDA or CPU.

    python fusion/train.py --synthetic          # no cache needed; proves the loop runs
    python fusion/train.py --features data/features
    python fusion/train.py --features data/features --labels data/public_weak_labels.csv

That third form is the experiment the whole label track has been waiting for (PLAN.md 7.2).
Train twice on IDENTICAL folds -- once on our pseudo-labels, once on nekkon's public weak
labels -- and compare model AUC on the gold studies. The week-2 A/B compared label sources
against a report-derived reference; this compares what actually gets submitted. Label-AUC and
model-AUC are not monotonically related, and noise-robust training over 4,349 studies can
absorb a 0.1 label-AUC gap entirely, so the moat is unproven until this run exists.

GOLD IS SCORED OUT-OF-FOLD AND POOLED, never per fold. 58 gold studies over 5 folds is 8-16
each, and MCL lands at zero positives in two of them -- a per-fold gold AUC is noise. Each gold
study is predicted by the fold that held it out, then all 58 are scored at once.

Gold is never trained on. It is the only honest measurement of the whole pipeline and spending
it on 58 extra training rows would be a bad trade.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "fusion"))
sys.path.insert(0, str(PROJ / "extractor"))
from dataset import LABELS, FeatureStore, StudyDataset, collate     # noqa: E402
from model import FusionHead, soft_bce                              # noqa: E402
from metrics import auc                                             # noqa: E402

D = PROJ / "data"


def device() -> str:
    """MPS first, then CUDA, then CPU.

    The fusion head is 3.7M parameters over cached vectors, so this is not a compute-bound job
    on any of them -- the binding constraint is RAM for the ~2.4 GB cache, not the accelerator.
    A 16 GB M5 peaks at 2.18 GB with the cache, model and optimizer live, measured.

    Everything here is fp32 deliberately. No autocast: a 3.7M-parameter model gains nothing from
    mixed precision, MPS fp16 reductions are uneven, and a GTX 980 Ti (Maxwell, sm_52) has no
    tensor cores, so fp16 would buy nothing there either.
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(dev: str) -> str:
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        return f"cuda ({p.name}, {p.total_memory / 1e9:.0f} GB, sm_{p.major}{p.minor})"
    return dev


def load_targets(path: Path | None, ids: list[str]) -> pd.DataFrame:
    """Soft targets per study. Defaults to the rule extractor's pseudo-labels."""
    p = path or (D / "pseudo_labels.csv")
    if not p.exists():
        sys.exit(f"{p} not found -- run extractor/run_extract.py first")
    df = pd.read_csv(p).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    missing = [c for c in LABELS if c not in df.columns]
    if missing:
        sys.exit(f"{p.name} is missing {missing}")
    return df.reindex(ids)[LABELS].fillna(0.5)


@torch.no_grad()
def predict(model, loader, dev) -> tuple[list[str], np.ndarray]:
    model.eval()
    uids, out = [], []
    for b in loader:
        logits = model(b["feats"].to(dev), b["slice_mask"].to(dev),
                       b["series_mask"].to(dev), b["series_type"].to(dev))
        out.append(torch.sigmoid(logits).float().cpu().numpy())
        uids += b["uid"]
    return uids, np.concatenate(out)


def run_fold(fold, folds, store, targets, args, dev):
    tr_ids = folds[(folds.fold != fold) & (~folds.is_gold)].StudyInstanceUID.tolist()
    va_ids = folds[folds.fold == fold].StudyInstanceUID.tolist()
    tr_ids = [u for u in tr_ids if u in store.data]
    va_ids = [u for u in va_ids if u in store.data]

    tr_ds = StudyDataset(store, tr_ids, targets, train=True,
                         series_dropout=args.series_dropout,
                         feature_noise=args.feature_noise, seed=args.seed + fold)
    va_ds = StudyDataset(store, va_ids, targets, train=False)
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                                        collate_fn=collate, drop_last=True)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=args.batch, collate_fn=collate)

    model = FusionHead(d=args.d, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.epochs * len(tr_dl), 1), pct_start=0.25)

    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for b in tr_dl:
            opt.zero_grad(set_to_none=True)
            logits = model(b["feats"].to(dev), b["slice_mask"].to(dev),
                           b["series_mask"].to(dev), b["series_type"].to(dev))
            loss = soft_bce(logits, b["y"].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b["uid"])
            n += len(b["uid"])
        if args.verbose:
            print(f"    fold {fold} ep {ep + 1}/{args.epochs}  loss {tot / max(n, 1):.4f}")

    uids, preds = predict(model, va_dl, dev)
    return model, uids, preds, len(tr_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default=str(D / "features"))
    ap.add_argument("--labels", default=None,
                    help="soft-target CSV; default data/pseudo_labels.csv. Point at "
                         "data/public_weak_labels.csv for the PLAN 7.2 A/B")
    ap.add_argument("--synthetic", action="store_true",
                    help="random features -- proves the loop runs before the cache exists")
    ap.add_argument("--folds", default=str(D / "folds.csv"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--series-dropout", type=float, default=0.25)
    ap.add_argument("--feature-noise", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out", default=str(PROJ / "fusion" / "runs"))
    ap.add_argument("--limit", type=int, default=0,
                    help="train on the first N studies only -- fast iteration on the M5 "
                         "while the full cache downloads")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = device()

    if not Path(args.folds).exists():
        sys.exit(f"{args.folds} not found -- run: python fusion/folds.py")
    folds = pd.read_csv(args.folds)

    if args.synthetic:
        # Keep EVERY gold study, then fill up with non-gold. Sampling blind leaves ~3 gold in
        # 300 and the pooled-OOF evaluation never runs -- which is the half of this script most
        # worth proving before the real cache lands.
        g = folds[folds.is_gold]
        rest = folds[~folds.is_gold].sample(n=min(300, (~folds.is_gold).sum()),
                                            random_state=args.seed)
        folds = pd.concat([g, rest]).sort_index().reset_index(drop=True)
        store = FeatureStore(None, folds.StudyInstanceUID.tolist(), synthetic=True,
                             seed=args.seed)
        print(f"SYNTHETIC features for {len(store)} studies ({len(g)} gold) -- shapes only, "
              f"AUCs are chance by construction")
    else:
        fdir = Path(args.features)
        if not fdir.exists():
            sys.exit(f"{fdir} not found.\nThe cache is built on Kaggle: run "
                     f"notebooks/kaggle_02_dinov2_cache.py, publish /kaggle/working/features as "
                     f"a Dataset, download it here. Until then: --synthetic")
        if args.limit:
            g = folds[folds.is_gold]
            rest = folds[~folds.is_gold].head(args.limit)
            folds = pd.concat([g, rest]).drop_duplicates("StudyInstanceUID").reset_index(drop=True)
            print(f"--limit {args.limit}: {len(folds)} studies ({len(g)} gold kept)")
        store = FeatureStore(fdir, folds.StudyInstanceUID.tolist())

    targets = load_targets(Path(args.labels) if args.labels else None,
                           folds.StudyInstanceUID.tolist())
    targets.index = folds.StudyInstanceUID

    print(f"device {describe_device(dev)} · {len(store):,} studies with features · "
          f"labels {Path(args.labels).name if args.labels else 'pseudo_labels.csv'}")

    t0 = time.time()
    oof: dict[str, np.ndarray] = {}
    for f in sorted(folds.fold.unique()):
        model, uids, preds, n_tr = run_fold(f, folds, store, targets, args, dev)
        oof.update(dict(zip(uids, preds)))
        print(f"  fold {f}: trained on {n_tr:,}, predicted {len(uids):,}")
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "d": args.d,
                    "labels": LABELS, "fold": int(f)}, outdir / f"fold{f}.pt")

    # ---- pooled OOF evaluation on gold ------------------------------------------------
    tr = pd.read_csv(D / "train.csv")
    gold = tr.dropna(subset=LABELS)
    gold = gold[gold.StudyInstanceUID.isin(oof)]
    print(f"\n{'=' * 62}\nOOF on {len(gold)} gold studies (pooled across folds)\n{'=' * 62}")
    if len(gold) < 10:
        print("  too few gold studies covered to score")
        return

    P = np.stack([oof[u] for u in gold.StudyInstanceUID])
    aucs = {}
    print(f"{'label':<18}{'n_pos':>6}{'AUC':>9}")
    for i, lab in enumerate(LABELS):
        y = gold[lab].to_numpy(dtype=float)
        if min(y.sum(), (1 - y).sum()) < 2:
            print(f"{lab:<18}{int(y.sum()):>6}{'--':>9}")
            continue
        a = auc(y, P[:, i])
        aucs[lab] = a
        print(f"{lab:<18}{int(y.sum()):>6}{a:>9.3f}")
    macro = float(np.mean(list(aucs.values())))
    print(f"{'-' * 33}\n{'MACRO':<18}{'':>6}{macro:>9.3f}")
    print(f"\n{time.time() - t0:.0f}s on {dev}")
    print("n=58 with 9-35 positives per label: the +-0.038 bootstrap CI from IMPROVEMENTS.md 0 "
          "applies here too.\nDo not read a macro difference under ~0.04 as real.")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    summary = {"macro_auc": macro, "per_label": aucs, "n_gold": int(len(gold)),
               "labels_source": args.labels or "pseudo_labels.csv",
               "synthetic": bool(args.synthetic), "epochs": args.epochs, "d": args.d}
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(P, columns=LABELS).assign(
        StudyInstanceUID=gold.StudyInstanceUID.values).to_csv(
        Path(args.out) / "oof_gold.csv", index=False)


if __name__ == "__main__":
    main()
