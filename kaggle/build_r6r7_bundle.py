r"""Build one inference bundle holding R6 and R7, for the public R6 worker's own two-group path (one decode a study).

    python kaggle/build_r6r7_bundle.py --r6 <R6 version-1 files> --r7 <R7 bundle files> --out <dir>
    kaggle datasets create -p <dir>        # private: raahimnawaz/rsna-knee-r6r7-dual

R6 (heliosli, bundle fe055406) and R7 (heliosli, bundle 8c7b666e) share the worker, trainer (md5 8622009e), views v4, size
384 / crop 352, slices, physical step, knee anchors and the state layout (their state specs differ only in prior numbers).
The worker (`compact_infer.py --models2 --arm2 --backbone2 --w2`) runs group 2 on the windows it decoded for group 1 and
mixes the two per finding in logit space: z = (1 - w2) z_R6 + w2 z_R7.

- **w2 = 0.5** on ten findings; **0 on Effusion and Baker's**, which R6 refines with stackers fitted on its own output.
- R6's Baker's state stacker reads group 1 only, so Baker's stays R6's. R6's Effusion fluid stacker refuses a two-group run
  unless it declares two lines whose weights equal the deployed (1 - w2, w2) = (1, 0): this build adds R7 as a second line
  at weight 0, which leaves R6's Effusion exactly as it was.
- Every source file is checked against its own manifest's sha256 table first; the new manifest lists the sha256 of every
  file in the bundle, which the notebook checks again before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
R6_ID, R7_ID = "fe055406fad6a12f", "8c7b666e27dc0e8f"
ARM2 = "R7lp"                        # group 2's file prefix: the worker globs <arm2>_s*_f*.pt
ZERO = ("Effusion", "Baker's")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def checked(d: Path, want_id: str) -> dict:
    m = json.loads((d / "manifest.json").read_text())
    if m["bundle_id"] != want_id:
        raise SystemExit(f"{d}: bundle {m['bundle_id']}, expected {want_id}")
    bad = [n for n, h in m["sha256"].items() if sha(d / n) != h]
    if bad:
        raise SystemExit(f"{d}: sha256 differs from its manifest for {bad}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r6", type=Path, required=True)
    ap.add_argument("--r7", type=Path, required=True)
    ap.add_argument("--w2", type=float, default=0.5)
    ap.add_argument("--slug", default="rsna-knee-r6r7-dual")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    m6, m7 = checked(a.r6, R6_ID), checked(a.r7, R7_ID)
    same = ("backbone", "size", "crop", "slices", "phys_mm", "views", "trainer_md5", "anchor_weights_sha256", "freeze_blocks")
    diff = [k for k in same if m6.get(k) != m7.get(k)]
    if diff:
        raise SystemExit(f"R6 and R7 differ on {diff}: one decode cannot serve both")
    s6, s7 = (json.loads((d / "state_spec.json").read_text()) for d in (a.r6, a.r7))
    if any(s6[k] != s7[k] for k in ("binaries", "categories", "labels")):
        raise SystemExit("R6 and R7 state layouts differ: R7's fold models cannot load under R6's state spec")

    a.out.mkdir(parents=True, exist_ok=False)
    for n in m6["sha256"]:
        shutil.copy2(a.r6 / n, a.out / n)
    models2 = []
    for k, n in enumerate(m7["models"]):
        new = f"{ARM2}_s0_f{k}.pt"
        shutil.copy2(a.r7 / n, a.out / new)
        models2.append(new)

    st = json.loads((a.r6 / m6["fluid_stacker"]).read_text())
    if st["feature"]["weights"] != [1.0] or len(st["feature"]["lines"]) != 1:
        raise SystemExit("R6's fluid stacker is not the one-line stacker this build expects")
    st["feature"]["lines"].append({"run": f"R7 bundle {R7_ID}", "note": "weight 0 on Effusion: R6's Effusion unchanged"})
    st["feature"]["weights"] = [1.0, 0.0]
    (a.out / m6["fluid_stacker"]).write_text(json.dumps(st, indent=1))

    w2 = [0.0 if c in ZERO else a.w2 for c in LABELS]
    man = dict(m6, models2=models2, arm2=ARM2, backbone2=m7["backbone"], w2=w2, flip_tta=False,
               source_bundles={"group1": {"bundle_id": R6_ID, "tag": m6.get("tag")}, "group2": {"bundle_id": R7_ID, "tag": m7.get("tag")}},
               note=f"R6 + R7 in one decode: z = (1 - w2) z_R6 + w2 z_R7; w2 {a.w2} except Effusion and Baker's (0)")
    man["sha256"] = {n: sha(a.out / n) for n in list(m6["sha256"]) + models2}
    man["bundle_id"] = hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()[:16]
    (a.out / "manifest.json").write_text(json.dumps(man, indent=1))
    (a.out / "dataset-metadata.json").write_text(json.dumps({
        "title": a.slug, "id": f"raahimnawaz/{a.slug}", "licenses": [{"name": "other"}],
        "description": (f"Private. heliosli's R6 (bundle {R6_ID}, version 1 of heliosli/rsna-knee-r6-frozen-fe055406) and R7 "
                        f"(bundle {R7_ID}, heliosli/rsna-knee-compact-r7cnxfsss) as one two-group bundle for heliosli's own "
                        f"worker; built by kaggle/build_r6r7_bundle.py. Credit: heliosli.")}, indent=1))
    print(f"bundle {man['bundle_id']} in {a.out}: R6 x{len(m6['models'])} + R7 x{len(models2)} (prefix {ARM2}), "
          f"w2 {dict(zip(LABELS, w2))}")


if __name__ == "__main__":
    main()
