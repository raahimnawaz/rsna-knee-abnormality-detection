"""Offline test of train_windows' gold fine-tune (--gold-train, --gold-repeats, --soup-init) on the synthetic corpus.

    python tests/gold_finetune_fixture.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "fusion"))
import train_windows as T  # noqa: E402

TW = [sys.executable, str(PROJ / "fusion" / "train_windows.py"), "--arch", "tiny", "--synthetic", "24", "--epochs", "1",
      "--k", "4", "--bs", "4", "--micro", "2", "--save-weights", "--seed", "7", "--bb-lr", "3e-3", "--head-lr", "3e-2"]


def run(args: list[str], ok: bool = True) -> str:
    r = subprocess.run(TW + args, capture_output=True, text=True, env={"TW_WORKERS": "0", "PATH": "/usr/bin:/bin"})
    if (r.returncode == 0) != ok:
        raise AssertionError(f"exit {r.returncode}, wanted {'0' if ok else 'non-zero'}:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.stdout + r.stderr


def main() -> None:
    n = 0
    tr, ytr = ["a", "b"], np.zeros((2, 12), np.float32)
    t2, y2 = T.with_gold(tr, ytr, ["g1", "g2"], np.ones((2, 12), np.float32) * [[1], [0.5]], 3)
    assert t2 == ["a", "b", "g1", "g1", "g1", "g2", "g2", "g2"] and y2.shape == (8, 12)
    assert (y2[2:5] == 1).all() and (y2[5:] == 0.5).all() and (y2[:2] == 0).all()
    n += 1
    a = {"w": torch.zeros(3), "n": torch.tensor([5])}
    b = {"w": torch.ones(3) * 4, "n": torch.tensor([9])}
    s = T.soup(a, b, 0.25)
    assert torch.allclose(s["w"], torch.full((3,), 3.0)) and s["n"].item() == 9
    n += 1
    print("ok  with_gold appends each graded study `repeats` times with its grades; soup = w init + (1 - w) trained")

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        gold = pd.DataFrame({"StudyInstanceUID": ["syn0020", "syn0021", "syn0022", "syn0023", "elsewhere"]}
                            | {c: [1.0, 0.0, 1.0, 0.0, 1.0] for c in T.LABELS})
        gold.to_csv(t / "train.csv", index=False)
        run(["--out", str(t / "base")])
        init = t / "base" / "last.pt"
        out = run(["--out", str(t / "ft0"), "--init", str(init), "--gold-train", str(t / "train.csv"), "--gold-repeats", "3"])
        assert "gold-train: 4 graded studies at their grades (1 not in the corpus), 3 draws each -> 28 draws" in out, out[-800:]
        out = run(["--out", str(t / "ft25"), "--init", str(init), "--gold-train", str(t / "train.csv"), "--gold-repeats", "3",
                   "--soup-init", "0.25"])
        assert "soup: 0.25 x the --init weights" in out
        n += 1
        A, F0, F25 = (torch.load(p, map_location="cpu") for p in (init, t / "ft0" / "last.pt", t / "ft25" / "last.pt"))
        fl = [k for k in A if A[k].is_floating_point()]
        a, f0, f25 = (torch.cat([d[k].double().flatten() for k in fl]) for d in (A, F0, F25))
        big = (f0 - a).abs() > 0.02
        frac = float(((f25 - f0)[big] / (a - f0)[big]).median())
        dev = float((f25 - (0.25 * a + 0.75 * f0)).abs().max())
        assert int(big.sum()) > 100 and abs(frac - 0.25) < 0.02 and dev < 2e-3, (int(big.sum()), frac, dev)
        n += 1
        print(f"ok  end to end: {int(big.sum())} weights moved > 0.02 in the fine-tune; the saved soup sits {frac:.3f} "
              f"of the way back to init (0.25 wanted), max deviation from 0.25 init + 0.75 unsouped {dev:.1e} (fp16)")
        sm = json.loads((t / "ft25" / "summary.json").read_text())
        assert sm["gold_train"] == 4 and sm["gold_repeats"] == 3 and sm["soup_init"] == 0.25 and sm["n_train"] == 28
        assert (t / "ft25" / "oof_all.csv").exists() and (t / "ft25" / "last_meta.json").exists()
        n += 1
        print("ok  summary carries gold_train 4, repeats 3, soup 0.25; oof_all.csv + last_meta.json written for the gate")
        run(["--out", str(t / "x"), "--gold-train", str(t / "train.csv")], ok=False)
        run(["--out", str(t / "y"), "--init", str(init), "--gold-train", str(t / "train.csv"), "--gold", str(t / "train.csv")],
            ok=False)
        n += 1
        print("ok  refused: --gold-train without --init; --gold with --gold-train")
    print(f"GOLD FINETUNE FIXTURE: {n} checks passed")


if __name__ == "__main__":
    main()
