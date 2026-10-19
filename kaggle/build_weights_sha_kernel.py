r"""Build a CPU kernel that prints the sha256 of every checkpoint in the attached training kernels.

    python kaggle/build_weights_sha_kernel.py --kernels <owner>/<training kernel> [<owner>/<another> ...] --out /tmp/sha
    python -m kaggle kernels push -p /tmp/sha
    python -m kaggle kernels output raahimnawaz/rsna-knee-weights-sha -p /tmp/sha_out     # weights_sha.json

The submission builder pins weights by sha256. This reads them where the files are, so a checkpoint
never has to be downloaded to be submitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "raahimnawaz"
SLUG = "rsna-knee-weights-sha"

SCRIPT = r'''
import hashlib, json, os
out = {}
for d, _, fs in os.walk("/kaggle/input"):
    for f in fs:
        if f == "last.pt":
            p = os.path.join(d, f)
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            out[p] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
            print(p, out[p], flush=True)
json.dump(out, open("/kaggle/working/weights_sha.json", "w"), indent=1)
print("n =", len(out))
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernels", nargs="+", required=True, help="training kernels whose outputs hold last.pt files")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{SLUG}.py").write_text(SCRIPT.lstrip("\n"), encoding="utf-8")
    meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "keywords": [], "dataset_sources": [], "competition_sources": [],
            "model_sources": [], "kernel_sources": a.kernels}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {a.out / (SLUG + '.py')} for {len(a.kernels)} kernel(s)\npush:  python -m kaggle kernels push -p {a.out}")


if __name__ == "__main__":
    main()
