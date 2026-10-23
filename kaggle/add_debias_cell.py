"""Append the de-bias cell to a finals notebook already built, e.g. the scored KDC final as pulled from Kaggle.

    .venv/bin/kaggle kernels pull raahimnawaz/rsna-knee-sub-main958-r6r7-t1-kdc -p /tmp/kdc -m
    python kaggle/add_debias_cell.py /tmp/kdc /tmp/out --tag main958-kdc-debias \\
        --debias-kernel raahimnawaz/rsna-knee-header-table --debias-shas <table sha256>,<labels sha256>

Every cell of the input stays as it is (outputs cleared); the de-bias cell comes last and its table kernel is mounted.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import build_public_blend_kernel as P
import build_submit_kernel as B


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--debias-kernel", required=True)
    ap.add_argument("--debias-shas", required=True)
    ap.add_argument("--margin-min", type=int, default=25)
    a = ap.parse_args()
    shas = dict(zip(("train_series_headers.csv", "report_labels.csv"), a.debias_shas.split(",")))
    if len(shas) != 2 or not all(re.fullmatch(r"[0-9a-f]{64}", v) for v in shas.values()):
        raise SystemExit("--debias-shas: two full sha256 values, the table's then the labels'")
    meta = json.loads((a.src / "kernel-metadata.json").read_text())
    nb = json.loads((a.src / meta["code_file"]).read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for name in ("NB_T0 = time.time()", "_P_COMMIT = ", "_P_READ = ", "DATA = find_mount(COMPETITION"):
        if not any(name in s for s in code):
            raise SystemExit(f"{a.src}: no cell defines {name!r}, which the de-bias cell reads")
    if any(re.search(r"\bH_OK\b|\b_H_W\b", s) for s in code):
        raise SystemExit(f"{a.src}: already carries a de-bias cell")
    if not re.search(r"\[\+KDC\] ARMS|\[public\+ours\] ARMS", code[-1]):
        raise SystemExit(f"{a.src}: its last code cell is not one of our blend cells")
    built_from = f"{B._git_head()}, appended to {meta['id']}"
    cell, intro = P.debias_cell(a.debias_kernel, shas, a.margin_min, built_from)
    if nb["cells"][0]["cell_type"] != "markdown":
        raise SystemExit(f"{a.src}: its first cell is not the intro")
    cells = [dict(c, outputs=[], execution_count=None) if c["cell_type"] == "code" else c for c in nb["cells"]]
    cells[0] = dict(cells[0], source=("".join(cells[0]["source"]) + intro).splitlines(keepends=True))
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": cell.strip("\n").splitlines(keepends=True)})
    slug = f"rsna-knee-sub-{a.tag}"
    if len(slug) > 50 or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", a.tag):
        raise SystemExit(f"slug {slug}: Kaggle refuses more than 50 characters; lowercase, digits, '-'")
    text = json.dumps(dict(nb, cells=cells), indent=1)
    if len(text.encode()) >= P.MAX_SOURCE:
        raise SystemExit(f"the notebook is {len(text.encode()):,} bytes: Kaggle refuses a source of 1 MB or more")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{slug}.ipynb").write_text(text, encoding="utf-8")
    out_meta = {k: v for k, v in meta.items() if k != "id_no"} | {
        "id": f"{B.OWNER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
        "kernel_sources": meta["kernel_sources"] + [a.debias_kernel]}
    (a.out / "kernel-metadata.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (slug + '.ipynb')}: {meta['id']}'s {len(nb['cells'])} cells unchanged + the de-bias cell "
          f"({len(text.encode()):,} bytes) | w {P.debias_w()} | from {built_from}")


if __name__ == "__main__":
    main()
