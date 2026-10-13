"""Offline test of build_public_blend_kernel's two cells on synthetic predictions: no Kaggle, no model.

    python notebooks/public_blend_fixture.py

Checks the mix against the formula, that an unbuilt study keeps the public rank, that the scored rerun keeps the
public file when our stage fails, is skipped or runs late, and that the commit run stops instead.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_public_blend_kernel as P  # noqa: E402
import build_submit_kernel as B  # noqa: E402

L = B.LABELS


def cells(tmp: Path, ours_src: str) -> tuple[str, str]:
    stage = (P.STAGE.replace("__MARGIN_MIN__", "25").replace("__MIN_LEFT_MIN__", "60")
             .replace("__OURS_SRC__", repr(ours_src)))
    blend = P.BLEND.replace("__W_OURS__", repr(P.W_OURS)).replace("__BUILT_FROM__", "fixture")
    return tuple(c.replace("/kaggle/working", str(tmp / "working")).replace("/tmp/ortho_sub", str(tmp / "ortho_sub"))
                 for c in (stage, blend))


def world(tmp: Path, n: int, seed: int = 0, agree: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    ids = [f"1.2.{i}" for i in range(n)]
    (tmp / "working").mkdir(parents=True)
    (tmp / "ortho_sub").mkdir()
    (tmp / "data").mkdir()
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(tmp / "data" / "test.csv", index=False)
    pub = pd.DataFrame(rng.random((n, 12)).astype(np.float32), columns=L)
    pub.insert(0, "StudyInstanceUID", ids)
    pub.to_csv(tmp / "working" / "submission.csv", index=False)
    ours = pd.DataFrame(agree * pub[L].to_numpy(float) + (1 - agree) * rng.random((n, 12)), columns=L)
    ours.insert(0, "StudyInstanceUID", ids)
    ours.sample(frac=1.0, random_state=1).to_csv(tmp / "ortho_sub" / "ortho_preds.csv", index=False)   # another order
    bad = ids[3:5]
    (tmp / "ortho_sub" / "preds_0.timing.json").write_text(json.dumps({"errors": {u: "x" for u in bad}}))
    return pub, ours, bad


def run(tmp: Path, ours_src: str, elapsed_h: float = 0.5) -> dict:
    g = {"NB_T0": time.time() - elapsed_h * 3600, "DATA": tmp / "data"}
    stage, blend = cells(tmp, ours_src)
    exec(compile(stage, "stage", "exec"), g)
    exec(compile(blend, "blend", "exec"), g)
    return g


OK_SRC = "_O_ARM, _O_MIN = 'fixture x5', 1.0"
n_pass = 0

# 1. the mix is the formula; unbuilt studies keep the public rank; the public file is kept beside it
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    pub, ours, bad = world(tmp, 400)
    g = run(tmp, OK_SRC)
    assert g["OURS_OK"]
    out = pd.read_csv(tmp / "working" / "submission.csv", dtype={"StudyInstanceUID": str})
    assert out.StudyInstanceUID.tolist() == pub.StudyInstanceUID.tolist() and out.columns.tolist() == pub.columns.tolist()
    ok = ~pub.StudyInstanceUID.isin(bad).to_numpy()
    for c in L:
        rp = pub[c].rank(pct=True).to_numpy(float)
        ro = ours[c][ok].rank(pct=True).to_numpy(float)
        want = rp.copy()
        want[ok] = (1 - P.W_OURS[c]) * rp[ok] + P.W_OURS[c] * ro
        assert np.allclose(out[c].to_numpy(), want, atol=1e-12), c
        assert np.allclose(out[c].to_numpy()[~ok], rp[~ok], atol=1e-12), c
    kept = pd.read_csv(tmp / "working" / "public_blend_sub.csv")
    assert np.allclose(kept[L].to_numpy(), pub[L].to_numpy())
    assert json.loads((tmp / "working" / "public_plus_ours_receipt.json").read_text())["ours_unbuilt"] == 2
    n_pass += 1

# 2. the scored rerun (>= 100 studies): a failed stage, a late start, a broken prediction file and predictions that
#    do not line up with the public ones (the misalignment guard) all keep the public file
for what, src, hours, spoil, agree in (("stage fails", "raise RuntimeError('boom')", 0.5, False, 0.7),
                                       ("late", OK_SRC, 8.2, False, 0.7),
                                       ("predictions missing", OK_SRC, 0.5, True, 0.7),
                                       ("misaligned", OK_SRC, 0.5, False, 0.0)):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        pub, _, _ = world(tmp, 400, agree=agree)
        before = (tmp / "working" / "submission.csv").read_bytes()
        if spoil:
            (tmp / "ortho_sub" / "ortho_preds.csv").unlink()
        g = run(tmp, src, hours)
        assert (tmp / "working" / "submission.csv").read_bytes() == before, what
        assert g["OURS_OK"] is (what in ("predictions missing", "misaligned")), what
        n_pass += 1

# 3. the commit run (3 studies): the same faults stop the notebook
for src, spoil in (("raise RuntimeError('boom')", False), (OK_SRC, True)):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        world(tmp, 3)
        if spoil:
            (tmp / "ortho_sub" / "ortho_preds.csv").unlink()
        try:
            run(tmp, src)
        except (RuntimeError, FileNotFoundError):
            n_pass += 1
        else:
            raise AssertionError("the commit run did not stop")

# 4. the deadline patch: _owait kills a subprocess that runs past _O_DEADLINE
patched = P.with_deadline(B.OURS)
fn = patched[patched.index("def _owait(procs, what):"):patched.index("# ---- 1. the gate")]
with tempfile.TemporaryDirectory() as d:
    g = {"_O": Path(d), "_osp": subprocess, "_otime": time, "_O_DEADLINE": time.time() - 100}
    exec(compile(fn, "owait", "exec"), g)
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    (Path(d) / "t.log").write_text("")
    t0 = time.time()
    try:
        g["_owait"]({"t": (p, open(Path(d) / "t.log"))}, "test inference")
    except RuntimeError as e:
        assert "deadline" in str(e) and p.wait(timeout=10) != 0 and time.time() - t0 < 60
        n_pass += 1
    else:
        raise AssertionError("no timeout")

print(f"public_blend_fixture: {n_pass} checks passed")
