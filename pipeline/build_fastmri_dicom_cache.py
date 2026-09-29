"""NYU fastMRI clinical knee DICOMs (unlabelled, all planes) -> our Stage A layout, streamed out of the .tar.xz.

    .venv/bin/python pipeline/build_fastmri_dicom_cache.py \
        --tar data/external/fastmri/dicom/knee_DICOMs_batch2.tar.xz data/external/fastmri/dicom/knee_DICOMs_batch1.tar.xz \
        --out data/external/fastmri/dicom_stage_a --workers 4
    .venv/bin/python pipeline/build_fastmri_dicom_cache.py --tar ... --out /tmp/x --limit 15 --headers   # a look first

WHY (RUNBOOK, owner 09-28 "the NYU 10k image thing"): more KNEES, pseudo-labelled later by our teachers (Noisy Student). These
exams carry no labels; this file only puts their images in the exact byte layout `train_windows.Corpus` reads.

HOW. The archives are ~108 GB of xz; unpacked they would not fit beside everything else, so each is read as ONE stream and every
study (patient/study_x/) is extracted to a scratch dir only while it is built, then deleted. A tar lists a study's files together,
so a study is complete when the next one starts. Per series, ONE header supplies what the competition's `train_series.csv` gives
for its own studies:
    plane  from ImageOrientationPatient: the axis the slice normal points along (x Sagittal, y Coronal, z Axial)
    fluid  fat-suppressed, the competition's Fluid_Sensitive (identical to its Fat_Suppression on all 24,371 series):
           "FS" / "FAT SAT" / "FATSAT" / "SPAIR" / "SPIR" / "STIR" / "TIRM" in SeriesDescription or ProtocolName, or "FS"
           in ScanOptions
Dropped: localizers / scouts (ImageType LOCALIZER, or those words in the description), DERIVED images (reformats), series
under 8 files, and series whose slices do not share one orientation. Then `build_slots64.build_study` UNCHANGED (spec c6 = Stage
A, stored orientation), so the bytes follow the same rules as the competition's own cache.

OUTPUT: parts `p000/`, `p001/`, ... of `--chunk` studies each: vols.npy (memmap, (n, 64, 336, 336) uint8), masks.npy, ids.npy,
slots.csv, series.csv (every series seen, with its plane / fluid / reason if dropped), manifest.json (Stage A's spec). Each part's
manifest gets its FINAL shard count when the run ends -- including a truncated archive (batch1 stopped at 77 of 137 GB when the
app relaunched 09-28 11:40), whose last, incomplete study is discarded. Study ids are `nyu_<patient>_<study>` -- never a
competition StudyInstanceUID.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_slots64 as B  # noqa: E402

FS_TOKENS = ("FS", "FAT SAT", "FATSAT", "FAT-SAT", "SPAIR", "SPIR", "STIR", "TIRM", "FS_", "_FS")
LOC_WORDS = ("LOC", "SURVEY", "CAL")                                  # whole words
LOC_SUBSTR = ("LOCALI", "SCOUT", "3 PLANE", "TRI PLANE", "TRIPLANE", "3PLANE")  # anywhere, after words()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def words(s: str) -> str:
    return " " + "".join(ch if ch.isalnum() else " " for ch in s.upper()) + " "


def series_info(d: Path) -> dict:
    files = sorted(f for f in d.iterdir() if not f.name.startswith("."))
    info = {"dir": d.name, "n_files": len(files), "plane": None, "fluid": 0, "drop": "", "desc": "", "uid": ""}
    if len(files) < 8:
        info["drop"] = "under 8 files"
        return info
    normals = set()
    for f in (files[0], files[len(files) // 2], files[-1]):
        h = pydicom.dcmread(f, stop_before_pixels=True)
        iop = [float(v) for v in getattr(h, "ImageOrientationPatient", [])]
        if len(iop) != 6:
            info["drop"] = "no orientation"
            return info
        n = np.cross(iop[:3], iop[3:])
        normals.add(int(np.argmax(np.abs(n))))
    desc = f"{getattr(h, 'SeriesDescription', '')} | {getattr(h, 'ProtocolName', '')}"
    it = [str(x).upper() for x in getattr(h, "ImageType", [])]
    info.update(desc=desc, uid=str(getattr(h, "SeriesInstanceUID", "")),
                model=str(getattr(h, "ManufacturerModelName", "")), software=str(getattr(h, "SoftwareVersions", "")),
                field_t=str(getattr(h, "MagneticFieldStrength", "")), study_date=str(getattr(h, "StudyDate", "")),
                tr=str(getattr(h, "RepetitionTime", "")), te=str(getattr(h, "EchoTime", "")),
                laterality=str(getattr(h, "Laterality", "")))  # de-identified: no scanner model survives (09-28)
    w = words(desc)
    if "LOCALIZER" in it or any(f" {t} " in w for t in LOC_WORDS) or any(t in w for t in LOC_SUBSTR):
        info["drop"] = "localizer"
    elif "DERIVED" in it:
        info["drop"] = "derived"
    elif len(normals) != 1:
        info["drop"] = "mixed orientation"
    info["plane"] = ("Sagittal", "Coronal", "Axial")[normals.pop()] if len(normals) == 1 else None
    opts = words(str(getattr(h, "ScanOptions", "")))                  # multi-valued ("PFP\\FS") or one string
    info["fluid"] = int(any(f" {t.strip('_')} " in w for t in FS_TOKENS) or " FS " in opts)
    info["scan_options"] = opts.strip()
    return info


def build_one(args):
    uid, study_dir = args
    t = time.time()
    study_dir = Path(study_dir)
    rows, series = [], []
    try:
        for d in sorted(p for p in study_dir.iterdir() if p.is_dir()):
            s = series_info(d)
            rows.append(s)
            if s["drop"] or not s["uid"]:
                continue
            (study_dir / s["uid"]).exists() or d.rename(study_dir / s["uid"])     # build_study wants study/<SeriesUID>/
            series.append({"SeriesInstanceUID": s["uid"], "plane": s["plane"], "fluid": s["fluid"]})
        vol, mask, meta = B.build_study(study_dir, series, B.SPAN, "c6", False)
        err = None if mask.any() else "no usable slot"
    except Exception as e:                                                        # noqa: BLE001 -- one study never kills a part
        vol, mask, meta, err = (np.zeros((B.SPECS["c6"].N_SLICES, B.SPECS["c6"].IMG, B.SPECS["c6"].IMG), np.uint8),
                                np.zeros(B.SPECS["c6"].N_SLICES, np.uint8), [], f"{type(e).__name__}: {e}")
    shutil.rmtree(study_dir, ignore_errors=True)
    return uid, vol, mask, meta, rows, err, time.time() - t


def studies_from(tar_path: Path, scratch: Path, limit: int):
    """Yield (uid, dir) per COMPLETE study, extracting one study at a time from a single xz stream."""
    cur, cur_dir, n = None, None, 0
    try:
        with tarfile.open(tar_path, mode="r|xz") as tf:
            for m in tf:
                parts = m.name.split("/")
                if len(parts) < 4 or not m.isfile():                   # batch/patient/study/series/file
                    continue
                key = (parts[1], parts[2])
                if key != cur:
                    if cur is not None:
                        yield f"nyu_{cur[0].split('_')[1] if '_' in cur[0] else cur[0]}_{cur[1]}", cur_dir
                        n += 1
                        if limit and n >= limit:
                            return
                    cur = key
                    cur_dir = scratch / f"{len(parts[1])}_{abs(hash(key)) % 10**12}"
                    shutil.rmtree(cur_dir, ignore_errors=True)
                dest = cur_dir / parts[3] / parts[-1]
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
    except (EOFError, tarfile.ReadError, OSError) as e:
        log(f"{tar_path.name}: stream ended early ({type(e).__name__}: {e}) -- the study in progress is discarded")
        if cur_dir is not None:
            shutil.rmtree(cur_dir, ignore_errors=True)
        return
    if cur is not None:
        yield f"nyu_{cur[0].split('_')[1] if '_' in cur[0] else cur[0]}_{cur[1]}", cur_dir


class PartWriter:
    def __init__(self, out: Path, chunk: int):
        self.out, self.chunk, self.k, self.buf = out, chunk, 0, []

    def add(self, item):
        self.buf.append(item)
        if len(self.buf) >= self.chunk:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        Sp = B.SPECS["c6"]
        d = self.out / f"p{self.k:03d}"
        d.mkdir(parents=True, exist_ok=True)
        vols = np.lib.format.open_memmap(d / "vols.npy", mode="w+", dtype=np.uint8,
                                         shape=(len(self.buf), Sp.N_SLICES, Sp.IMG, Sp.IMG))
        masks = np.zeros((len(self.buf), Sp.N_SLICES), np.uint8)
        meta, ser, errs = [], [], {}
        for i, (uid, vol, mask, m, rows, err, _) in enumerate(self.buf):
            vols[i], masks[i] = vol, mask
            meta += [{"StudyInstanceUID": uid, **x} for x in m]
            ser += [{"StudyInstanceUID": uid, **r} for r in rows]
            if err:
                errs[uid] = err
        vols.flush()
        np.save(d / "masks.npy", masks)
        np.save(d / "ids.npy", np.asarray([b[0] for b in self.buf], dtype=object), allow_pickle=True)
        pd.DataFrame(meta).to_csv(d / "slots.csv", index=False)
        pd.DataFrame(ser).to_csv(d / "series.csv", index=False)
        man = {**Sp.manifest(tuple(B.SPAN)), "canon": False, "split": "nyu_fastmri_clinical", "shard": self.k,
               "n_shards": None, "n_studies": len(self.buf), "errors": errs,
               "slot_fill": {s.name: float(masks[:, x:y].any(1).mean()) for (x, y), s in zip(Sp.slot_offsets(), Sp.SLOTS)}}
        (d / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
        log(f"part p{self.k:03d}: {len(self.buf)} studies, {len(errs)} errors, fill "
            + json.dumps({k: round(v, 2) for k, v in man["slot_fill"].items()}))
        self.k += 1
        self.buf = []

    def finalize(self):
        self.flush()
        for d in sorted(self.out.glob("p[0-9][0-9][0-9]")):
            j = json.loads((d / "manifest.json").read_text())
            j["n_shards"] = self.k
            (d / "manifest.json").write_text(json.dumps(j, indent=2, default=str))
        log(f"finalized {self.k} parts (n_shards written to every manifest)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tar", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, default=None, help="per-study extraction dir (default: <out>/_scratch)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="studies per archive (a look first)")
    ap.add_argument("--headers", action="store_true", help="print every series' description / plane / fluid / drop")
    a = ap.parse_args()
    if any(a.out.glob("p[0-9][0-9][0-9]")):
        raise SystemExit(f"{a.out} already holds parts -- use a new --out (parts are numbered from 0 per run)")
    scratch = a.scratch or a.out / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    w = PartWriter(a.out, a.chunk)
    t0, n, secs = time.time(), 0, []
    ex = ProcessPoolExecutor(max_workers=a.workers)
    try:
        for tar in a.tar:
            log(f"{tar.name}: streaming")
            pending = []
            for uid, d in studies_from(tar, scratch, a.limit):
                pending.append(ex.submit(build_one, (uid, str(d))))
                while len(pending) > 2 * a.workers:                   # bounded: never more studies on disk than this
                    r = pending.pop(0).result()
                    w.add(r); n += 1; secs.append(r[-1])
                    if a.headers:
                        for s in r[4]:
                            print(f"  {r[0][:40]:40s} {s['n_files']:4d} {str(s['plane']):9s} fs={s['fluid']} "
                                  f"{s['drop'] or 'kept':16s} {s['desc'][:60]}", flush=True)
                    if n % 100 == 0:
                        log(f"  {n} studies  {np.mean(secs):.2f} s/study/worker  {(time.time() - t0) / 60:.1f} min")
            for f in pending:
                r = f.result()
                w.add(r); n += 1; secs.append(r[-1])
                if a.headers:
                    for s in r[4]:
                        print(f"  {r[0][:40]:40s} {s['n_files']:4d} {str(s['plane']):9s} fs={s['fluid']} "
                              f"{s['drop'] or 'kept':16s} {s['desc'][:60]}", flush=True)
            log(f"{tar.name}: done, {n} studies so far, {(time.time() - t0) / 60:.1f} min")
    finally:
        ex.shutdown()
        w.finalize()
        shutil.rmtree(scratch, ignore_errors=True)
    log(f"done: {n} studies in {(time.time() - t0) / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
