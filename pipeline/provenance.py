"""SHA-256 record of the external inputs that cannot be regenerated, so a changed upstream fails loudly."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
DIR = PROJ / "provenance"
RECORD = DIR / "inputs.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path, source: str, accept_new: bool = False, keep_copy: bool = False) -> str:
    """Hash `path` and hold it to the record for its file name. Returns the hash."""
    path = Path(path)
    rec = json.loads(RECORD.read_text()) if RECORD.exists() else {"files": {}}
    h = sha256(path)
    old = rec["files"].get(path.name)
    if old is not None and old["sha256"] != h and not accept_new:
        sys.exit(f"⛔ {path.name} is not the file this project measured on.\n"
                 f"   recorded {old['sha256'][:16]}  ({old['recorded']}, {old['source']})\n"
                 f"   got      {h[:16]}  ({path})\n"
                 f"   Every number measured on the recorded file was measured on a different input.\n"
                 f"   Restore it (provenance/ docstring), or re-run with --accept-new to move on.")
    if old is None or old["sha256"] != h:
        rec["files"][path.name] = {"sha256": h, "source": source,
                                   "recorded": _dt.date.today().isoformat()}
        DIR.mkdir(exist_ok=True)
        RECORD.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        verb = "RECORDED" if old is None else "MOVED (--accept-new)"
        print(f"  provenance: {verb} {path.name} {h[:12]} -- commit provenance/")
    else:
        print(f"  provenance: {path.name} matches the record ({h[:12]})")
    if keep_copy:
        dst = DIR / path.name
        if not dst.exists() or sha256(dst) != h:
            shutil.copyfile(path, dst)
            print(f"  provenance: stored a copy at {dst.relative_to(PROJ)} -- commit it")
    return h
