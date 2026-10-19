"""Train the v3 recipe on a GPU that is not Kaggle's, and bring the weights back where a submission kernel can gate them.

    # this laptop, once: the fold of every study, from the Kaggle runs' own lists (goes into the private targets dataset)
    python kaggle/offkaggle.py folds --out data/student/folds_site_stage_b.csv

    # the GPU machine, once: this repo, ~/.kaggle credentials, data/train.csv, the targets dataset, then the cache (~62 GB)
    python kaggle/offkaggle.py pull --dest <cache>
    # one fold (two T4s take ~4.9 h; `--dry` prints the command and the lists' sizes without training)
    python kaggle/offkaggle.py train --fold 0 --corpus <cache> --labels data/student/targets_teacher_v3.csv \
        --folds data/student/folds_site_stage_b.csv --out <runs>
    # the folds that are done -> one flat private dataset each (<slug>-f<k>), and the submission builder's arguments
    python kaggle/offkaggle.py pack --runs <runs> --slug rsna-knee-v3 --out <dir>

The recipe is `build_backbone_ab_kernel.py`'s (SMALL_PACKAGE + V3_TRAIN_ARGS, V3_EPOCHS), imported, not copied; the
lists are the Kaggle folds' (held out by site, gold in none). A fold trained here and one trained there differ in the
GPU count behind the batch of 8 and in nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJ / "kaggle"), str(PROJ / "fusion")]
import build_backbone_ab_kernel as K  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# the five teacher-label folds' run records: fold k's lists as its Kaggle kernel wrote them
RECORDS = {0: "runs_cnxs_teach1_f0", 1: "runs_cnxs_teach1_f12", 2: "runs_cnxs_teach1_f12",
           3: "runs_cnxs_teach1_f34", 4: "runs_cnxs_teach1_f34"}


def _ids(p: Path) -> list[str]:
    return pd.read_csv(p).StudyInstanceUID.astype(str).tolist()


def folds(a) -> None:
    """Fold k's held-out studies are the rows of its tracked oof_all.csv; where the kernel's own lists are on disk
    (they are not in git) they must agree."""
    val = {}
    for k, rec in RECORDS.items():
        d = PROJ / "fusion" / rec / "runs_convnext_small" / f"fold{k}"
        val[k] = _ids(d / "oof_all.csv")
        if (d / "val_studies.csv").exists() and _ids(d / "val_studies.csv") != val[k]:
            raise SystemExit(f"fold {k}: oof_all.csv is not the kernel's val list")
    every = [u for k in val for u in val[k]]
    if len(set(every)) != len(every):
        raise SystemExit("a study is held out in two folds")
    for k, rec in RECORDS.items():                          # each train list is exactly the other four folds
        t = PROJ / "fusion" / rec / "runs_convnext_small" / f"fold{k}" / "train_studies.csv"
        if t.exists() and set(_ids(t)) != set(every) - set(val[k]):
            raise SystemExit(f"fold {k}: the train list is not the other four folds")
    out = pd.DataFrame([(u, k) for k in val for u in val[k]], columns=["StudyInstanceUID", "fold"])
    out = out.sort_values("StudyInstanceUID").reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} studies, held out per fold {out.fold.value_counts().sort_index().tolist()}, "
          f"sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()[:16]}")


def lists(folds_csv: Path, fold: int, labels: Path, gold: Path | None) -> tuple[list[str], list[str]]:
    f = pd.read_csv(folds_csv)
    if f.StudyInstanceUID.duplicated().any() or set(f.fold) != set(range(5)):
        raise SystemExit(f"{folds_csv}: duplicate studies or not five folds")
    t = pd.read_csv(labels).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    has = set(t.index[t[LABELS].notna().any(axis=1)])
    tr = sorted(u for u in f.loc[f.fold != fold, "StudyInstanceUID"].astype(str) if u in has)
    va = sorted(f.loc[f.fold == fold, "StudyInstanceUID"].astype(str))
    if gold:
        g = set(pd.read_csv(gold).dropna(subset=LABELS).StudyInstanceUID.astype(str))
        if g & (set(tr) | set(va)) or g & set(t.index):
            raise SystemExit("a graded study is in a list or in the targets")
    if set(tr) & set(va) or not tr or not va:
        raise SystemExit("train and val overlap, or one is empty")
    return tr, va


def train(a) -> None:
    import torch
    tr, va = lists(a.folds, a.fold, a.labels, a.gold)
    run = a.out / f"fold{a.fold}"
    run.mkdir(parents=True, exist_ok=True)
    pd.Series(tr, name="StudyInstanceUID").to_csv(run / "train_studies.csv", index=False)
    pd.Series(va, name="StudyInstanceUID").to_csv(run / "val_studies.csv", index=False)
    n_gpu = torch.cuda.device_count()
    world = n_gpu if n_gpu in (2, 4, 8) else 1              # the batch of 8 must split evenly
    launch = ([sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--node_rank", "0", "--master_addr",
               "127.0.0.1", "--master_port", "29511", "--nproc_per_node", str(world)] if world > 1 else [sys.executable, "-u"])
    cmd = launch + [str(PROJ / "fusion" / "train_windows.py"), "--arch", a.arch, "--corpus", str(a.corpus),
                    "--layout", "stage_b", "--train", str(run / "train_studies.csv"), "--val", str(run / "val_studies.csv"),
                    "--labels", str(a.labels), "--out", str(run), "--weights-dir", str(a.out), "--epochs", str(a.epochs)]
    cmd += K.SMALL_PACKAGE + ["--time-budget-h", str(a.time_budget_h)] + K.V3_TRAIN_ARGS + a.extra.split()
    cmd += (["--gold", str(a.gold)] if a.gold else []) + ["--save-weights"]
    head = subprocess.run(["git", "-C", str(PROJ), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(PROJ), "status", "--porcelain", "--", "fusion", "kaggle", "pipeline"],
                           capture_output=True, text=True).stdout.strip()
    prov = {"built_from": head + ("+uncommitted" if dirty else ""), "fold": a.fold, "n_train": len(tr), "n_val": len(va),
            "labels_sha256": hashlib.sha256(a.labels.read_bytes()).hexdigest(),
            "folds_sha256": hashlib.sha256(a.folds.read_bytes()).hexdigest(), "gpus": n_gpu, "world": world,
            "gpu": torch.cuda.get_device_name(0) if n_gpu else "none", "cmd": cmd}
    (run / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"fold {a.fold}: train {len(tr)} | val {len(va)} | {n_gpu} GPU(s), {world} process(es) | from {prov['built_from']}")
    print(" ".join(cmd))
    if dirty:
        print("  ⚠️ fusion/, kaggle/ or pipeline/ differ from HEAD -- commit first so the run is reproducible")
    if a.dry:
        return
    rc = subprocess.call(cmd)
    if rc:
        raise SystemExit(f"train_windows exited {rc}")
    s = json.loads((run / "summary.json").read_text())
    bad = [k for k, want in (("n_train", len(tr)), ("n_val", len(va)), ("epochs_done", a.epochs)) if s[k] != want]
    if bad or s["stopped_early"]:
        raise SystemExit(f"fold {a.fold} is not the run that was asked for: {bad} {s['stopped_early']} "
                         f"(studies missing from the cache, or the schedule cut short)")
    print(f"fold {a.fold} COMPLETE: {s['epochs_done']} epochs, {s['wall_min']:.0f} min, gold-58 "
          f"{s['gold_macro_report_only']} (report only) -> {run}")


def pull(a) -> None:
    kaggle = str(Path(sys.executable).parent / "kaggle")
    for sh in K.STAGE_B_SHARDS:
        d = a.dest / sh.split("/")[1]
        have = lambda: any(d.rglob("vols.npy")) and any(d.rglob("manifest.json"))     # noqa: E731  (they land in stage_b/)
        if have() and not a.force:
            print(f"{sh}: already there")
            continue
        for attempt in range(1, a.retries + 1):
            try:
                rc = subprocess.run([kaggle, "kernels", "output", sh, "-p", str(d)], timeout=a.timeout_min * 60).returncode
            except subprocess.TimeoutExpired:
                rc = -1
            if rc == 0 and have():
                break
            print(f"{sh}: attempt {attempt} failed (rc {rc})")
        else:
            raise SystemExit(f"{sh}: not downloaded after {a.retries} attempts")
    import train_windows as TW
    c = TW.Corpus(a.dest, "stage_b")                        # the trainer's own checks: 8 of 8 shards, one spec, shapes
    print(f"cache complete: {len(c.ids)} studies, {c.layout}")


def pack(a) -> None:
    """One FLAT private dataset per fold (<slug>-f<k>): the submission kernel looks for a folder with last.pt and
    oof_all.csv side by side, and a flat dataset is that folder with no archive in between."""
    done = sorted(p.parent.parent if a.which else p.parent
                  for p in a.runs.glob(f"fold*/{a.which}/last.pt" if a.which else "fold*/last.pt"))
    if not done:
        raise SystemExit(f"no fold*/{a.which or ''}/last.pt under {a.runs}")
    if a.out.exists():
        shutil.rmtree(a.out)
    shas, slugs = [], []
    for fd in done:
        s = json.loads((fd / "summary.json").read_text())
        if s["stopped_early"] or s["epochs_done"] != s["epochs_cap"]:
            raise SystemExit(f"{fd.name}: {s['epochs_done']} of {s['epochs_cap']} epochs -- not packed")
        slugs.append(f"{a.slug}-f{fd.name.removeprefix('fold')}")
        src, dst = (fd / a.which if a.which else fd), a.out / slugs[-1]
        dst.mkdir(parents=True)
        for f in ("last.pt", "last_meta.json", "oof_all.csv", "oof_plain.csv", "oof_gold.csv"):
            if (src / f).exists():
                shutil.copy(src / f, dst / f)
        for f in ("summary.json", "provenance.json"):
            if (fd / f).exists():
                shutil.copy(fd / f, dst / f)
        if not {"last.pt", "last_meta.json", "oof_all.csv"} <= {p.name for p in dst.iterdir()}:
            raise SystemExit(f"{fd.name}: the gate needs last.pt, last_meta.json and oof_all.csv side by side")
        shas.append(hashlib.sha256((dst / "last.pt").read_bytes()).hexdigest())
        (dst / "dataset-metadata.json").write_text(json.dumps(
            {"id": f"{K.OWNER}/{slugs[-1]}", "title": slugs[-1], "licenses": [{"name": "other"}]}))
        print(f"{fd.name}: {a.which or 'last'} sha256 {shas[-1]} -> {dst}")
    (a.out / "weights_sha.json").write_text(json.dumps(dict(zip(slugs, shas)), indent=1))
    kg, py = ".venv/bin/kaggle", ".venv/bin/python"
    print("\nupload, each private (the default):\n" + "\n".join(f"  {kg} datasets create -p {a.out / g}" for g in slugs)
          + f"\nthen:  {py} kaggle/build_submit_kernel.py --ours-only \\\n"
          f"           --weights-dataset {','.join(f'{K.OWNER}/{g}' for g in slugs)} \\\n"
          f"           --weights-sha {','.join(shas)} --tag <short> --desc \"<what this is>\" --out <dir>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("folds")
    f.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--fold", type=int, choices=range(5), required=True)
    t.add_argument("--corpus", type=Path, required=True, help="the Stage B cache (pull's --dest)")
    t.add_argument("--labels", type=Path, required=True)
    t.add_argument("--folds", type=Path, required=True)
    t.add_argument("--gold", type=Path, default=PROJ / "data" / "train.csv", help="scored, report only; '' to skip")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--arch", default="convnext_small")
    t.add_argument("--epochs", type=int, default=K.V3_EPOCHS)
    t.add_argument("--time-budget-h", type=float, default=48.0)
    t.add_argument("--extra", default="", help="more train_windows flags, one string; a later flag overrides the recipe's")
    t.add_argument("--dry", action="store_true")
    p = sub.add_parser("pull")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--timeout-min", type=int, default=90, help="per shard and attempt (~7.8 GB each)")
    p.add_argument("--force", action="store_true")
    k = sub.add_parser("pack")
    k.add_argument("--runs", type=Path, required=True)
    k.add_argument("--slug", required=True)
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--which", default="avg3", help="avg3 = the averaged weights (v3 submits these); '' = last.pt")
    a = ap.parse_args()
    if a.cmd == "train" and str(a.gold) in ("", "."):
        a.gold = None
    {"folds": folds, "train": train, "pull": pull, "pack": pack}[a.cmd](a)


if __name__ == "__main__":
    main()
