"""A free CPU mount test of a built final: its mounts, asset cell, Raptor's lookups, our weights, and its de-bias cell.

    python kaggle/build_finals_mounttest.py --final <built final dir> --out <dir>   # then kaggle kernels push -p <dir>
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

TEMPLATE = r'''import base64, json, os, stat, time, hashlib
from pathlib import Path
P = json.loads(base64.b64decode("__PAYLOAD__").decode())
NB_T0 = time.time()
print("MOUNTTEST", P["final"], "| /kaggle/input:", sorted(os.listdir("/kaggle/input")), flush=True)
_bin = Path("/tmp/mt_bin"); _bin.mkdir(exist_ok=True)
(_bin / "nvidia-smi").write_text("#!/bin/sh\necho 'no GPU (mount test)'\n")
(_bin / "nvidia-smi").chmod(0o755)
os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

exec(compile(P["asset"], "final_asset_cell", "exec"), globals())

for _k, _block in enumerate(P["raptor"]):
    _ns = {"hashlib": hashlib, "Path": Path}
    exec(compile(_block, "raptor_lookup_%d" % _k, "exec"), _ns)
    print("RAPTOR_OK", _k, str(_ns["_weights"]), flush=True)

_found = {}
for _root, _dirs, _files in os.walk("/kaggle/input"):
    _dirs[:] = [d for d in _dirs if d != "competitions"]
    if "last.pt" in _files:
        _p = Path(_root) / "last.pt"
        _h = hashlib.sha256(_p.read_bytes()).hexdigest()
        if _h in P["shas"]:
            _found[_h] = str(_p)
_missing = [h[:16] for h in P["shas"] if h not in _found]
assert not _missing, ("our members not found", _missing)
print("OURS_OK", len(_found), "of", len(P["shas"]), flush=True)
__KDC____DEBIAS__print("MOUNTTEST_DONE %.1f s" % (time.time() - NB_T0), flush=True)
'''

KDC = r'''
_kfound, _seg_w, _seg_v = {}, None, None
for _root, _dirs, _files in os.walk("/kaggle/input"):
    _dirs[:] = [d for d in _dirs if d != "competitions"]
    if "last.pt" in _files:
        _p = Path(_root) / "last.pt"
        _h = hashlib.sha256(_p.read_bytes()).hexdigest()
        if _h in P["k_shas"]:
            _kfound[_h] = str(_p)
    if _seg_w is None and "anatomy.pt" in _files:
        _seg_w = os.path.join(_root, "anatomy.pt")
    if _seg_v is None and _root.endswith(os.path.join("vendor_runtime", "monai060")):
        _seg_v = _root
_kmissing = [h[:16] for h in P["k_shas"] if h not in _kfound]
assert not _kmissing, ("KDC's members not found", _kmissing)
assert _seg_w and _seg_v, ("KDC's segmenter not mounted", _seg_w, _seg_v)
print("KDC_OK", len(_kfound), "of", len(P["k_shas"]), "| segmenter", _seg_w, "|", _seg_v, flush=True)
'''

DEBIAS = r'''
import numpy as _mnp, pandas as _mpd
for _root, _dirs, _files in os.walk("/kaggle/input"):
    _dirs[:] = [d for d in _dirs if d not in ("train_series", "test_series", "train_images", "test_images")]
    if {"test.csv", "test_series.csv", "train_series.csv", "sample_submission.csv"} <= set(_files):
        DATA = Path(_root)
        break
_cols = list(_mpd.read_csv(DATA / "sample_submission.csv", nrows=1).columns)
_ids = _mpd.read_csv(DATA / "test.csv", dtype=str).StudyInstanceUID
_mpd.DataFrame({"StudyInstanceUID": _ids} | {c: _mnp.linspace(0.1, 0.9, len(_ids)) for c in _cols[1:]}).to_csv(
    "/kaggle/working/submission.csv", index=False)
_P_COMMIT, _P_READ = True, False
exec(compile(P["debias"], "final_debias_cell", "exec"), globals())
assert H_OK, "the de-bias cell did not finish"
print("DEBIAS_OK", flush=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--final", type=Path, required=True, help="a built final's kernel folder (kernel-metadata.json + notebook)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta = json.loads((a.final / "kernel-metadata.json").read_text())
    nb = json.loads(next(a.final.glob("*.ipynb")).read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    asset = [s for s in code if "CODE_SLUG = " in s and "ASSETS_READY" in s]
    if len(asset) != 1:
        raise SystemExit(f"{len(asset)} asset cells (CODE_SLUG + ASSETS_READY) in the final: expected 1")
    raptor = []
    for s in code:
        if "_weight_name = '" in s:
            i = s.index("_weight_name = '")
            j = s.index("Checkpoint", i)
            raptor.append(s[i:s.index("\n", j) + 1])
    if not raptor and any("raptor" in d for d in meta["dataset_sources"]):
        raise SystemExit("a Raptor dataset is mounted but no Raptor lookup was found")
    groups = [re.findall(r"[0-9a-f]{64}", g) for g in re.findall(r"_O_SHAS = \[(.*?)\]", "\n".join(code), re.S)]
    if not groups or not groups[0]:
        raise SystemExit("no _O_SHAS (our members' sha256) in the final")
    if len(groups) > 2 or (len(groups) == 2 and not groups[1]):
        raise SystemExit(f"{len(groups)} groups of members in the final: expected ours, and KDC's at most")
    shas, k_shas = groups[0], (groups[1] if len(groups) == 2 else [])
    debias = [s for s in code if "H_STAGE ok=" in s]
    if len(debias) > 1:
        raise SystemExit(f"{len(debias)} de-bias cells in the final: expected 1 at most")
    final_slug = meta["id"].split("/")[1]
    slug = "rsna-knee-mounttest-" + final_slug.removeprefix("rsna-knee-sub-")
    if len(slug) > 50:
        raise SystemExit(f"slug {slug} over 50 characters")
    body = {"final": final_slug, "asset": asset[0], "raptor": raptor, "shas": shas}
    if k_shas:
        body["k_shas"] = k_shas
    if debias:
        body["debias"] = debias[0]
    payload = base64.b64encode(json.dumps(body).encode()).decode()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{slug}.py").write_text(TEMPLATE.replace("__PAYLOAD__", payload).replace("__KDC__", KDC.lstrip("\n") if k_shas else "")
                                           .replace("__DEBIAS__", DEBIAS.lstrip("\n") if debias else ""))
    m = {"id": f"{meta['id'].split('/')[0]}/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
         "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
         "keywords": [], "dataset_sources": meta["dataset_sources"], "kernel_sources": meta["kernel_sources"],
         "competition_sources": meta["competition_sources"], "model_sources": meta.get("model_sources", [])}
    if meta.get("docker_image"):
        m["docker_image"] = meta["docker_image"]
    (a.out / "kernel-metadata.json").write_text(json.dumps(m, indent=2))
    print(f"built {a.out / (slug + '.py')} | asset cell {len(asset[0]):,} chars | Raptor lookups {len(raptor)} | "
          f"our members {len(shas)}" + (f" + KDC's {len(k_shas)}" if k_shas else "") + (" + de-bias cell" if debias else "") + f" | image {meta.get('docker_image', 'default')[:60]}")


if __name__ == "__main__":
    main()
