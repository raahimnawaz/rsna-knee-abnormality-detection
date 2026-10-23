"""Build the Kaggle CPU kernel that reads every training series' header (pipeline/header_prior.py) for the de-bias stage.

    python kaggle/build_header_table_kernel.py --out /tmp/ht
    .venv/bin/kaggle kernels push -p /tmp/ht

Its output (train_series_headers.csv, report_labels.csv) is mounted by the finals as a kernel source. It runs on the
finals' Docker image, so the reader and the model see the same pydicom and scikit-learn as the scored run.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OWNER = "raahimnawaz"
SLUG = "rsna-knee-header-table"
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
TARGETS = PROJ / "data" / "student" / "targets_consensus_v2.csv"

SCRIPT = r'''
import base64, hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
import pandas as pd
T0 = time.time()
SRC = base64.b64decode(__SRC__).decode()
assert hashlib.sha256(SRC.encode()).hexdigest() == "__SRC_SHA__", "header_prior.py: embedded source does not match"
hp = types.ModuleType("header_prior")
exec(compile(SRC, "header_prior.py", "exec"), hp.__dict__)
W = Path("/kaggle/working")
(W / "report_labels.csv").write_bytes(base64.b64decode(__LABELS__))
assert hashlib.sha256((W / "report_labels.csv").read_bytes()).hexdigest() == "__LABELS_SHA__"
ROOT = None
for d, dirs, fs in os.walk("/kaggle/input"):
    dirs[:] = [x for x in dirs if x not in ("train_series", "test_series", "train_images", "test_images")]
    if {"train_series.csv", "test_series.csv", "test.csv"} <= set(fs):
        ROOT = Path(d)
        break
assert ROOT is not None, "competition root not found"
import pydicom, sklearn
print("HEADER_ENV", json.dumps({"pydicom": pydicom.__version__, "sklearn": sklearn.__version__,
                                "pandas": pd.__version__, "numpy": np.__version__}), flush=True)
tr = hp.read_split(ROOT, "train", workers=32)
tr.to_csv(W / "train_series_headers.csv", index=False)
err = tr.error.notna()
print(f"TRAIN_HEADERS {len(tr)} series, {tr.StudyInstanceUID.nunique()} studies, {int(err.sum())} unread "
      f"({time.time() - T0:.0f} s); errors: {tr.error[err].value_counts().head(5).to_dict()}", flush=True)
Y = pd.read_csv(W / "report_labels.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
X, voc = hp.study_features(tr)
print("HEADER_VOCAB", json.dumps({k: len(v) for k, v in voc.items()}), flush=True)
print("HEADER_CV", json.dumps(hp.cv_auroc(X.reindex(Y.index), Y)), flush=True)
te = hp.read_split(ROOT, "test")
Xte, _ = hp.study_features(te, voc)
ids = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID
prior = hp.fit_predict(X.reindex(Y.index), Y, Xte.reindex(ids))
sub = pd.DataFrame({"StudyInstanceUID": ids} | {c: np.linspace(0.1, 0.9, len(ids)) for c in hp.LABELS})
out = hp.debias(sub, prior, hp.DEBIAS_W)
assert np.isfinite(out[hp.LABELS].to_numpy()).all() and list(out.StudyInstanceUID) == list(ids)
print(f"TEST_PATH {len(te)} series of {len(ids)} studies read, {int(te.error.notna().sum())} unread; prior and "
      f"de-bias ran: {prior.round(3).to_dict(orient='index')}", flush=True)
print("HEADER_TABLE_DONE", json.dumps({f: hashlib.sha256((W / f).read_bytes()).hexdigest()
                                       for f in ("train_series_headers.csv", "report_labels.csv")}),
      f"{time.time() - T0:.0f} s", flush=True)
'''


def report_labels() -> bytes:
    """targets_consensus_v2 at >= 0.5, unread cells kept empty: the report reading the prior is fitted on."""
    y = pd.read_csv(TARGETS, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    buf = io.StringIO()
    (y >= 0.5).astype(float).where(y.notna()).reset_index().to_csv(buf, index=False, float_format="%.0f")
    return buf.getvalue().encode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    src = (PROJ / "pipeline" / "header_prior.py").read_bytes()
    lab = report_labels()
    script = (SCRIPT.replace("__SRC__", repr(base64.b64encode(src).decode()))
              .replace("__SRC_SHA__", hashlib.sha256(src).hexdigest())
              .replace("__LABELS__", repr(base64.b64encode(lab).decode()))
              .replace("__LABELS_SHA__", hashlib.sha256(lab).hexdigest()))
    compile(script, SLUG, "exec")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{SLUG}.py").write_text(script, encoding="utf-8")
    (a.out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "keywords": [], "dataset_sources": [], "kernel_sources": [], "model_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"], "docker_image": DOCKER_IMAGE}, indent=2))
    print(f"built {a.out / (SLUG + '.py')} ({len(script):,} bytes) | header_prior.py {hashlib.sha256(src).hexdigest()[:16]}"
          f" | report_labels.csv {hashlib.sha256(lab).hexdigest()[:16]} ({len(lab):,} bytes)")


if __name__ == "__main__":
    main()
