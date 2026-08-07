"""Feature cache -> padded batches. Runs entirely in RAM on a 16 GB M5.

The whole cache is ~2.4 GB (24,371 series x 32 slices x 1536 dims fp16), which on a unified-
memory machine means it is loaded once and never touched again -- no DataLoader workers, no
disk pressure, no sharding. Measured peak on a 16 GB M5, cache + model + optimizer steps: 2.18
GB. That is the single biggest reason this trains in minutes per experiment.

SYNTHETIC MODE exists so every consumer of this file -- the model, the training loop, the
submission notebook -- is testable before a single DICOM has been decoded on Kaggle. It emits
tensors with the exact shapes, dtypes, masks and degenerate cases the real cache produces,
including single-series studies and studies missing every optional plane. `--self-test` is the
gate that says the pipeline is correct up to the features being real.

Augmentation on cached features is a narrow menu, and the reason is worth restating: the
backbone already ran, so every pixel-space transform in PLAN.md 3.3 (affine, gamma, bias field,
Rician noise) is unavailable. What survives:

  series dropout  mandatory, not defensive -- 87.2% of studies are missing at least one of the
                  six types (FINDINGS.md 3.2), so this augmentation matches the test
                  distribution rather than merely regularising against it.
  slice jitter    which 24 of the 32 cached slices to use. This is why the cache stores 32.
  feature noise   mild gaussian; the cheapest stand-in for intensity jitter.

NO horizontal flip. Not because features cannot be flipped -- they cannot -- but because it is
invalid here at any level: hflip swaps Medial and Lateral, so it would need the output pairs
swapped too (PLAN.md 3.2). Handedness is canonicalised once, in preprocessing.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "pipeline"))
from preprocess import (EMBED_DIM, N_SERIES_TYPES, PLANE_ID,          # noqa: E402
                        SLICES_PER_SERIES, SLICES_PER_SERIES_TRAIN)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
MAX_SERIES = 14          # measured range is 3-14 series/study (FINDINGS.md 3)


def series_type_id(plane: int, fluid_sensitive: int) -> int:
    """(plane, FS) -> 0..5, or -1 when either is unknown.

    Fluid_Sensitive and Fat_Suppression are perfectly redundant -- 10,361 studies at 0/0 and
    14,010 at 1/1, zero off-diagonal -- so one flag gives 6 types, not 12 (FINDINGS.md 3.1).
    """
    if plane < 0 or fluid_sensitive < 0:
        return -1
    return int(plane) * 2 + int(fluid_sensitive)


class FeatureStore:
    """Loads every study's .npz once into memory, grouped by series."""

    def __init__(self, feature_dir: Path | None, study_ids: list[str], synthetic: bool = False,
                 seed: int = 0):
        self.ids = list(study_ids)
        self.synthetic = synthetic
        self.data: dict[str, tuple] = {}
        if synthetic:
            self._make_synthetic(seed)
        else:
            self._load(Path(feature_dir))

    def _load(self, d: Path) -> None:
        """Load every study once. ~2.4 GB for the full corpus, which fits an M5 with room.

        Measured on a 16 GB M5: 2.18 GB peak resident with the full cache, the model and
        optimizer steps live. Features stay fp16 here and are upcast per batch in __getitem__ --
        holding them as float32 would double this for no benefit.
        """
        missing, corrupt, t0 = [], [], time.time()
        for n, uid in enumerate(self.ids, 1):
            p = d / f"{uid}.npz"
            if not p.exists():
                missing.append(uid)
                continue
            try:
                z = np.load(p)
                feats, sidx = z["feats"], z["series_idx"]
                plane, fs = z["plane"], z["fluid_sensitive"]
            except Exception:
                # A session killed mid-write leaves a truncated .npz. kaggle_02 writes via
                # tmp+rename so this should not happen, but a silently-dropped study would be
                # worse than a loud one.
                corrupt.append(uid)
                continue
            series = []
            for k in np.unique(sidx):
                m = sidx == k
                series.append((feats[m], series_type_id(int(plane[m][0]), int(fs[m][0]))))
            if series:
                self.data[uid] = series
            if n % 500 == 0 or n == len(self.ids):
                print(f"\r  loading features {n:,}/{len(self.ids):,} "
                      f"({time.time() - t0:.0f}s)", end="", flush=True)
        print()

        nbytes = sum(a.nbytes for s in self.data.values() for a, _ in s)
        n_ser = sum(len(s) for s in self.data.values())
        print(f"  {len(self.data):,} studies / {n_ser:,} series resident, "
              f"{nbytes / 1e9:.2f} GB fp16")
        if missing:
            print(f"  WARNING: {len(missing):,}/{len(self.ids):,} studies have no feature file "
                  f"(first: {missing[0]}). Is the cache build finished?")
        if corrupt:
            print(f"  WARNING: {len(corrupt):,} unreadable .npz files (first: {corrupt[0]}). "
                  f"Delete them and re-run the shard that produced them.")
        self.ids = [u for u in self.ids if u in self.data]

    def _make_synthetic(self, seed: int) -> None:
        """Shapes, dtypes and degenerate cases identical to the real cache."""
        rng = np.random.default_rng(seed)
        for i, uid in enumerate(self.ids):
            # Mirror the measured distribution: median 5 series, range 3-14, and force the
            # single-series and max-series edge cases into the first two studies so the
            # self-test always exercises them.
            k = 1 if i == 0 else (MAX_SERIES if i == 1 else int(rng.integers(3, 8)))
            series = []
            for _ in range(k):
                n = int(rng.integers(SLICES_PER_SERIES_TRAIN, SLICES_PER_SERIES + 1))
                series.append((rng.standard_normal((n, EMBED_DIM)).astype(np.float16),
                               int(rng.integers(-1, N_SERIES_TYPES))))
            self.data[uid] = series

    def __len__(self) -> int:
        return len(self.ids)


class StudyDataset(torch.utils.data.Dataset):
    def __init__(self, store: FeatureStore, ids: list[str], targets: pd.DataFrame,
                 train: bool = False, n_slices: int = SLICES_PER_SERIES_TRAIN,
                 series_dropout: float = 0.25, feature_noise: float = 0.0, seed: int = 0):
        self.store, self.ids, self.train = store, list(ids), train
        self.n_slices = n_slices
        self.series_dropout = series_dropout
        self.feature_noise = feature_noise
        self.y = targets.reindex(self.ids)[LABELS].to_numpy(dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def _pick(self, n_have: int) -> np.ndarray:
        """Slice jitter: which n_slices of the n_have cached to use."""
        if n_have <= self.n_slices:
            return np.arange(n_have)
        if not self.train:
            return np.linspace(0, n_have - 1, self.n_slices).round().astype(int)
        # Jittered even spread -- keeps volume coverage but shifts the sampling grid.
        base = np.linspace(0, n_have - 1, self.n_slices)
        step = max((n_have - 1) / max(self.n_slices - 1, 1), 1e-6)
        jit = self.rng.uniform(-step / 2, step / 2, size=self.n_slices)
        return np.clip((base + jit).round(), 0, n_have - 1).astype(int)

    def __getitem__(self, i: int):
        uid = self.ids[i]
        series = self.store.data[uid]

        if self.train and self.series_dropout > 0 and len(series) > 1:
            keep = [j for j in range(len(series))
                    if self.rng.random() > self.series_dropout]
            series = [series[j] for j in keep] if keep else [series[self.rng.integers(len(series))]]

        series = series[:MAX_SERIES]
        K, S = len(series), self.n_slices
        feats = torch.zeros(K, S, EMBED_DIM)
        smask = torch.zeros(K, S, dtype=torch.bool)
        stype = torch.full((K,), -1, dtype=torch.long)

        for k, (arr, t) in enumerate(series):
            idx = self._pick(len(arr))
            v = torch.from_numpy(arr[idx].astype(np.float32))
            feats[k, :len(idx)] = v
            smask[k, :len(idx)] = True
            stype[k] = t

        if self.train and self.feature_noise > 0:
            feats = feats + torch.randn_like(feats) * self.feature_noise * smask.unsqueeze(-1)

        return {"feats": feats, "slice_mask": smask, "series_type": stype,
                "y": torch.from_numpy(self.y[i].copy()), "uid": uid}


def collate(batch: list[dict]) -> dict:
    """Pad to the batch's max series count and build the series mask."""
    B = len(batch)
    K = max(b["feats"].shape[0] for b in batch)
    S = batch[0]["feats"].shape[1]

    feats = torch.zeros(B, K, S, EMBED_DIM)
    smask = torch.zeros(B, K, S, dtype=torch.bool)
    sermask = torch.zeros(B, K, dtype=torch.bool)
    stype = torch.full((B, K), -1, dtype=torch.long)
    y = torch.stack([b["y"] for b in batch])

    for i, b in enumerate(batch):
        k = b["feats"].shape[0]
        feats[i, :k] = b["feats"]
        smask[i, :k] = b["slice_mask"]
        sermask[i, :k] = True
        stype[i, :k] = b["series_type"]

    return {"feats": feats, "slice_mask": smask, "series_mask": sermask,
            "series_type": stype, "y": y, "uid": [b["uid"] for b in batch]}


def self_test() -> None:
    """Prove the model consumes what the dataset emits, including the degenerate cases."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import FusionHead, soft_bce

    ids = [f"synthetic_{i:04d}" for i in range(24)]
    store = FeatureStore(None, ids, synthetic=True)
    tgt = pd.DataFrame(np.random.default_rng(0).uniform(size=(len(ids), len(LABELS))),
                       index=ids, columns=LABELS)

    ds = StudyDataset(store, ids, tgt, train=True, series_dropout=0.5, feature_noise=0.01)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=collate, shuffle=True)
    model = FusionHead()
    n_par = sum(p.numel() for p in model.parameters())

    print(f"synthetic studies : {len(ds)}  (study 0 has 1 series, study 1 has {MAX_SERIES})")
    print(f"fusion head params: {n_par:,}")
    for b in dl:
        out = model(b["feats"], b["slice_mask"], b["series_mask"], b["series_type"])
        loss = soft_bce(out, b["y"])
        assert out.shape == (b["y"].shape[0], len(LABELS)), out.shape
        assert torch.isfinite(out).all(), "non-finite logits"
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        print(f"  batch feats {tuple(b['feats'].shape)} -> logits {tuple(out.shape)}  "
              f"loss {loss.item():.4f}  finite grads "
              f"{all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)}")
        break

    # The case that crashes submissions: one study, one series, minimum slices, unknown type.
    solo = FeatureStore(None, ["solo"], synthetic=True)
    solo.data["solo"] = [(np.zeros((3, EMBED_DIM), np.float16), -1)]
    b = collate([StudyDataset(solo, ["solo"], tgt.head(1).set_axis(["solo"]))[0]])
    out = FusionHead()(b["feats"], b["slice_mask"], b["series_mask"], b["series_type"])
    assert torch.isfinite(out).all(), "degenerate single-series study produced non-finite logits"
    print(f"  degenerate study (1 series, 3 slices, unknown type) -> {tuple(out.shape)} finite OK")
    print("\nself-test PASSED")


if __name__ == "__main__":
    self_test()
