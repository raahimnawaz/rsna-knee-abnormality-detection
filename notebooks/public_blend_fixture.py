"""Offline test of build_public_blend_kernel's cells (ours, and KDC's two) on synthetic predictions: no Kaggle, no model.

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


def cells(tmp: Path, ours_src: str, read: bool = False) -> tuple[str, str]:
    stage = (P.STAGE.replace("__MARGIN_MIN__", "25").replace("__MIN_LEFT_MIN__", "60").replace("__READ__", repr(read))
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


def run(tmp: Path, ours_src: str, elapsed_h: float = 0.5, read: bool = False) -> dict:
    g = {"NB_T0": time.time() - elapsed_h * 3600, "DATA": tmp / "data"}
    stage, blend = cells(tmp, ours_src, read)
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

# 3b. a read (--read), scored rerun: the same four faults raise instead of keeping the public file, and a clean run
#     still writes the mix
for what, src, hours, spoil, agree in (("stage fails", "raise RuntimeError('boom')", 0.5, False, 0.7),
                                       ("late", OK_SRC, 8.2, False, 0.7),
                                       ("predictions missing", OK_SRC, 0.5, True, 0.7),
                                       ("misaligned", OK_SRC, 0.5, False, 0.0)):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        world(tmp, 400, agree=agree)
        if spoil:
            (tmp / "ortho_sub" / "ortho_preds.csv").unlink()
        try:
            run(tmp, src, hours, read=True)
        except (RuntimeError, FileNotFoundError):
            n_pass += 1
        else:
            raise AssertionError(f"read mode kept the public file: {what}")
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    pub, _, _ = world(tmp, 400)
    g = run(tmp, OK_SRC, read=True)
    assert g["OURS_OK"] and (tmp / "working" / "public_plus_ours_receipt.json").is_file()
    n_pass += 1

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

# 5. KDC (10-18), the two cells after ours: on its findings only, the same rank formula on top of the file our blend
#    wrote; every other column byte-identical; a study KDC could not build keeps its rank
K_MIX = {"Lateral OA": 0.3}
K_SRC = "_O_ARM, _O_MIN = 'kdc fixture x3', 1.0"


def k_world(tmp: Path, pub: pd.DataFrame, agree: float = 0.7, seed: int = 5) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    (tmp / "k_sub").mkdir()
    k = pd.DataFrame(agree * pub[L].to_numpy(float) + (1 - agree) * rng.random((len(pub), 12)), columns=L)
    k.insert(0, "StudyInstanceUID", pub.StudyInstanceUID)
    k.sample(frac=1.0, random_state=2).to_csv(tmp / "k_sub" / "ortho_preds.csv", index=False)
    kbad = pub.StudyInstanceUID.tolist()[7:10]
    (tmp / "k_sub" / "preds_1.timing.json").write_text(json.dumps({"errors": {u: "x" for u in kbad}}))
    return k, kbad


def exec_k(g: dict, tmp: Path, k_src: str) -> dict:
    for name, c in (("stage_k", P.STAGE_K), ("blend_k", P.BLEND_K)):
        c = (c.replace("__MARGIN_MIN__", "25").replace("__MIN_LEFT_MIN__", "60").replace("__K_SRC__", repr(k_src))
             .replace("__K_MIX__", repr(K_MIX)).replace("__BUILT_FROM__", "fixture")
             .replace("/kaggle/working", str(tmp / "working")).replace("/tmp/k_sub", str(tmp / "k_sub")))
        exec(compile(c, name, "exec"), g)
    return g


with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    pub, _, _ = world(tmp, 400)
    k, kbad = k_world(tmp, pub)
    g = run(tmp, OK_SRC)
    mid = pd.read_csv(tmp / "working" / "submission.csv", dtype={"StudyInstanceUID": str})
    exec_k(g, tmp, K_SRC)
    assert g["K_OK"]
    out = pd.read_csv(tmp / "working" / "submission.csv", dtype={"StudyInstanceUID": str})
    assert out.StudyInstanceUID.tolist() == mid.StudyInstanceUID.tolist() and out.columns.tolist() == mid.columns.tolist()
    kok = ~mid.StudyInstanceUID.isin(kbad).to_numpy()
    for c in L:
        if c not in K_MIX:
            assert np.array_equal(out[c].to_numpy(), mid[c].to_numpy()), c
            continue
        rp = mid[c].rank(pct=True).to_numpy(float)
        rk = k.set_index("StudyInstanceUID").reindex(mid.StudyInstanceUID)[c].to_numpy(float)[kok]
        want = rp.copy()
        want[kok] = (1 - K_MIX[c]) * rp[kok] + K_MIX[c] * pd.Series(rk).rank(pct=True).to_numpy()
        assert np.allclose(out[c].to_numpy(), want, atol=1e-12), c
        assert np.allclose(out[c].to_numpy()[~kok], rp[~kok], atol=1e-12), c
    assert np.allclose(pd.read_csv(tmp / "working" / "before_kdc_sub.csv")[L].to_numpy(), mid[L].to_numpy())
    assert json.loads((tmp / "working" / "plus_kdc_receipt.json").read_text())["kdc_unbuilt"] == 3
    n_pass += 1

# 6. KDC in the scored rerun: a failed stage, a late start, a missing file and a misaligned file all keep the file our
#    blend wrote, byte for byte
for what, src, k_hours, spoil, agree in (("stage fails", "raise RuntimeError('boom')", None, False, 0.7),
                                         ("late", K_SRC, 8.2, False, 0.7),
                                         ("predictions missing", K_SRC, None, True, 0.7),
                                         ("misaligned", K_SRC, None, False, 0.0)):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        pub, _, _ = world(tmp, 400)
        k_world(tmp, pub, agree=agree)
        if spoil:
            (tmp / "k_sub" / "ortho_preds.csv").unlink()
        g = run(tmp, OK_SRC)
        before = (tmp / "working" / "submission.csv").read_bytes()
        if k_hours is not None:
            g["NB_T0"] = time.time() - k_hours * 3600
        exec_k(g, tmp, src)
        assert (tmp / "working" / "submission.csv").read_bytes() == before, what
        assert g["K_OK"] is (what in ("predictions missing", "misaligned")), what
        n_pass += 1

# 7. KDC in the commit run (3 studies): a failed stage or a missing file stops the notebook
for src, spoil in (("raise RuntimeError('boom')", False), (K_SRC, True)):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        pub, _, _ = world(tmp, 3)
        k_world(tmp, pub)
        if spoil:
            (tmp / "k_sub" / "ortho_preds.csv").unlink()
        try:
            exec_k(run(tmp, OK_SRC), tmp, src)
        except (RuntimeError, FileNotFoundError):
            n_pass += 1
        else:
            raise AssertionError("the commit run did not stop on a KDC fault")

print(f"public_blend_fixture: {n_pass} checks passed")
