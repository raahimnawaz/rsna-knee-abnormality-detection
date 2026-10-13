r"""Build a submission kernel that reads the public R7 (heliosli's four ConvNeXt-small folds) ALONE on the leaderboard.

    kaggle kernels pull varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956 -p <pub> -m
    python notebooks/build_r7_read_kernel.py --public-dir <pub> --out <dir>             # R7 as published
    python notebooks/build_r7_read_kernel.py --public-dir <pub> --flip --out <dir>      # R7 with its mirror pass on
    python -m kaggle kernels push -p <dir>

The public notebook's cells run unchanged (the same notebook, by sha256, that our blend kernel runs), so R7's stage
is exactly the one inside the 0.956 recipe. One cell follows: it publishes R7's own predictions (`ours_sub.csv`, the
file the public blend cell reads) as the submission in place of the blend.

--flip changes ONE line of the public R7 cell: the inference worker's `--flip-tta` (every fold model also scores the
mirrored windows, MCL from the plain pass) is passed whatever the manifest says. R7 trained with mirror 0.5 and its
manifest ships `flip_tta: False`.

A read must not be mistaken for something else: if R7 left more than 2% of the studies unscored (they would be
mean-filled), or --flip did not reach the worker, the cell raises and there is no score.

Read in the commit log before submitting: `OURS_STAGE_COMPLETE {"status": "PASS", …}` · `NOTEBOOK_COMPLETE` ·
`R7_READ …` · 0 tracebacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

OWNER = "raahimnawaz"
PUBLIC = "varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956"
FLIP_OFF = "if manifest.get('flip_tta'):"
FLIP_ON = "if True:   # [our read] the mirror pass ON; the manifest ships flip_tta False"

TAIL = r'''
# ============================================================================ #
# Our read: R7's four folds ALONE__FLIP_NOTE__. Everything above is the public
# notebook; this cell publishes R7's own predictions (ours_sub.csv, the file
# the blend cell read) in place of the blend. It raises rather than publish
# something that is not that.
# ============================================================================ #
import json as _r7json
_r7_flip = __FLIP__
if ('--flip-tta' in WORKER_ARGS) != _r7_flip:
    raise RuntimeError(f"--flip-tta in the worker's arguments is {'--flip-tta' in WORKER_ARGS}, this read wants {_r7_flip}")
_r7_unscored = int(our_fail.sum())
if _r7_unscored > int(0.02 * n):          # the commit run's three studies: any one unscored stops it
    raise RuntimeError(f"R7 left {_r7_unscored} of {n} studies unscored (mean-filled): not a read of R7")
_r7_sub = ours.reset_index()[cols]
_r7_v = _r7_sub[labels].to_numpy(np.float64)
assert _r7_sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(_r7_v).all()
assert ((_r7_v >= 0) & (_r7_v <= 1)).all()
_r7_distinct = {c: int(_r7_sub[c].nunique()) for c in labels}
(WORK / 'public_blend_sub.csv').write_text((WORK / 'submission.csv').read_text())
_r7_sub.to_csv(WORK / 'submission.csv', index=False)
(WORK / 'r7_read_receipt.json').write_text(_r7json.dumps({
    "what": "R7 four folds alone" + (" + mirror pass" if _r7_flip else ""), "flip_tta": _r7_flip, "rows": n,
    "unscored_mean_filled": _r7_unscored, "distinct_values": _r7_distinct, "worker_args": WORKER_ARGS,
    "seconds_total": round(time.time() - NB_T0, 1), "built_from": "__BUILT_FROM__"}, indent=1))
print(f"R7_READ flip_tta={_r7_flip} rows={n} unscored={_r7_unscored} min_distinct={min(_r7_distinct.values())} "
      f"seconds_total={time.time() - NB_T0:.0f}", flush=True)
display(_r7_sub.head())
'''

INTRO = """# RSNA Knee -- a read of the public R7 alone__FLIP_TITLE__

Cells 1-__N_PUB__ are `__PUBLIC__` as published (notebook sha256 `__PUB_SHA__`)__FLIP_INTRO__. The last cell is ours:
it submits R7's own predictions, heliosli's four ConvNeXt-small folds, in place of the blend. R7's score alone is
published nowhere; this reads it. Built from `__BUILT_FROM__`.
"""


def main() -> None:
    import build_submit_kernel as B
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--public-dir", type=Path, required=True, help=f"`kaggle kernels pull {PUBLIC} -m`")
    ap.add_argument("--flip", action="store_true", help="pass --flip-tta to R7's inference worker")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    pmeta = json.loads((a.public_dir / "kernel-metadata.json").read_text())
    if pmeta["id"] != PUBLIC:
        raise SystemExit(f"{a.public_dir} holds {pmeta['id']}, not {PUBLIC}")
    nb_path = a.public_dir / pmeta["code_file"]
    pub_sha = hashlib.sha256(nb_path.read_bytes()).hexdigest()
    pub = json.loads(nb_path.read_text())
    cells = pub["cells"]
    src = ["".join(c["source"]) for c in cells]
    code_ix = [i for i, c in enumerate(cells) if c["cell_type"] == "code"]
    # the seams this build relies on -- refuse an upstream version that moved any of them
    last = src[code_ix[-1]]
    need_last = ("sub.to_csv(WORK / 'submission.csv', index=False)", "print('NOTEBOOK_COMPLETE'",
                 "ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels]", "our_fail = np.array(")
    if not all(s in last for s in need_last):
        raise SystemExit("public notebook's last code cell is not the blend cell this build reads -- re-read it")
    if sum("shutil.copyfile(pred / 'submission.csv', WORK / 'ours_sub.csv')" in src[i] for i in code_ix) != 1:
        raise SystemExit("public notebook: R7's stage no longer writes ours_sub.csv in one cell -- re-read it")
    hits = [i for i in code_ix if FLIP_OFF in src[i]]
    if len(hits) != 1 or src[hits[0]].count(FLIP_OFF) != 1 or "WORKER_ARGS += ['--flip-tta']" not in src[hits[0]]:
        raise SystemExit("public notebook: the flip_tta line is not where it was -- re-read it")
    if any("_r7_" in src[i] for i in code_ix):
        raise SystemExit("public notebook already uses one of our names -- re-read it")

    built_from = f"{B._git_head()}, public notebook sha256 {pub_sha[:16]}"
    tail = (TAIL.replace("__FLIP__", str(a.flip)).replace("__BUILT_FROM__", built_from)
            .replace("__FLIP_NOTE__", ", with its mirror pass on" if a.flip else ""))
    compile(tail, "r7_tail", "exec")
    body = [dict(c, outputs=[], execution_count=None) if c["cell_type"] == "code" else c for c in cells]
    if a.flip:
        body[hits[0]] = dict(body[hits[0]], source=src[hits[0]].replace(FLIP_OFF, FLIP_ON).splitlines(keepends=True))
        if "".join(body[hits[0]]["source"]).replace(FLIP_ON, FLIP_OFF) != src[hits[0]]:
            raise SystemExit("the flip patch changed more than its one line")
    intro = (INTRO.replace("__N_PUB__", str(len(cells))).replace("__PUBLIC__", PUBLIC).replace("__PUB_SHA__", pub_sha[:16])
             .replace("__BUILT_FROM__", built_from)
             .replace("__FLIP_TITLE__", ", with its mirror pass on" if a.flip else "")
             .replace("__FLIP_INTRO__", ", with ONE line changed: R7's inference worker is given `--flip-tta` (the "
                      "mirror pass its trainer's mirror augmentation pairs with; its manifest ships it off)" if a.flip
                      else ", unchanged"))
    new = ([{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)}] + body
           + [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
               "source": tail.strip("\n").splitlines(keepends=True)}])
    slug = "rsna-knee-sub-r7-alone" + ("-flip" if a.flip else "")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{slug}.ipynb").write_text(json.dumps(dict(pub, cells=new), indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": pmeta["dataset_sources"], "kernel_sources": [], "model_sources": [],
            "docker_image": pmeta["docker_image"], "machine_shape": "NvidiaTeslaT4"}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (slug + '.ipynb')}: {len(cells)} public cells "
          + ("with the one flip line changed" if a.flip else "unchanged") + f", + intro and the R7 cell | from {built_from}")
    if "uncommitted" in built_from:
        print("  ⚠️ files differ from HEAD -- commit first so the run is reproducible")
    print(f"push:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
