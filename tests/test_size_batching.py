"""Lots à budget de pixels et collation des recadrages (phase 5 E7)."""
import numpy as np
import pytest
import torch

import cell_shards as cs
from cell_shard_dataset import CellShardDataset
from shard_helpers import write_tiny_dir
from size_batching import (PixelBudgetSampler, batch_to_device, collate, load_index, padded_pixels)

BUDGET = 3 * 32 * 32  # trois recadrages de 32 x 32


@pytest.fixture
def index(tmp_path):
    return load_index(write_tiny_dir(tmp_path / "s", n_train=12, n_val=4))


def _batches(index, **kw):
    args = dict(split="train", px_budget=BUDGET, seed=0, epoch=0) | kw
    return list(PixelBudgetSampler(index, **args))


def _padded(index, batch):
    es = [index.entries[i] for i in batch]
    return len(es) * padded_pixels(max(e.h for e in es), max(e.w for e in es))


@pytest.mark.unit
def test_padded_pixels_rounds_up_to_multiple_of_32():
    assert padded_pixels(15, 33) == 32 * 64


@pytest.mark.unit
def test_every_batch_respects_budget(index):
    batches = _batches(index)
    assert all(_padded(index, b) <= BUDGET for b in batches)


@pytest.mark.unit
def test_each_record_exactly_once_per_epoch(index):
    ids = sorted(i for b in _batches(index) for i in b)
    train_ids = [i for i, e in enumerate(index.entries) if e.split == "train"]
    assert ids == train_ids


@pytest.mark.unit
def test_same_seed_and_epoch_reproduce_and_epochs_differ(index):
    assert _batches(index, epoch=3) == _batches(index, epoch=3)
    assert _batches(index, epoch=0) != _batches(index, epoch=1)


@pytest.mark.unit
def test_oversized_cell_forms_its_own_batch(index):
    batches = _batches(index, px_budget=32 * 32)  # un seul recadrage 32 x 32 par lot
    assert all(len(b) == 1 for b in batches)
    assert len(batches) == 12
    small = _batches(index, px_budget=10)  # plus petit que n'importe quel recadrage
    assert len(small) == 12


@pytest.mark.unit
def test_bins_filter_restricts_classes(index):
    ids = [i for b in _batches(index, bins=("200-1000",)) for i in b]
    assert ids and all(index.entries[i].bin == "200-1000" for i in ids)


@pytest.mark.unit
def test_sampler_len_and_invalid_budget(index):
    assert len(PixelBudgetSampler(index, "train", BUDGET, 0, 0)) == len(_batches(index))
    with pytest.raises(ValueError):
        PixelBudgetSampler(index, "train", 0, 0, 0)


@pytest.mark.unit
def test_max_per_bin_subsample_is_fixed_across_epochs_and_capped(index):
    a = sorted(i for b in _batches(index, max_per_bin=2, epoch=0) for i in b)
    b = sorted(i for b in _batches(index, max_per_bin=2, epoch=5) for i in b)
    assert a == b and len(a) == 4
    assert sorted(index.entries[i].bin for i in a) == ["200-1000", "200-1000", "50-200", "50-200"]


@pytest.mark.unit
def test_collate_shapes_and_valid_excludes_missing(index):
    ds = CellShardDataset(index)
    recs = [ds[0], ds[2]]  # 15 x 15 et 26 x 26
    batch = collate(recs)
    assert batch.x.shape == (2, 3, 32, 32)
    assert (batch.targets.Hm, batch.targets.Wm) == (26, 26)
    assert batch.crop_hw == ((15, 15), (26, 26))
    assert batch.areas.tolist() == [81.0, 400.0]
    v0 = batch.targets.valid[0]
    assert int(v0.sum()) == 15 * 15 - int(recs[0].invalid.sum()) == int(batch.targets.npix[0])
    assert not v0[:15, :15][torch.from_numpy(recs[0].invalid.copy())].any() and not v0[15:, :].any()
    assert torch.equal(batch.x[0, 2, :15, :15], (~torch.from_numpy(recs[0].invalid)).float())
    assert np.allclose(batch.targets.T[0, :15, :15].numpy(), cs.rate_mm_h(recs[0]))
    assert batch.meta[1]["bin"] == "200-1000"


@pytest.mark.unit
def test_collate_does_not_mutate_and_rejects_empty(index):
    rec = CellShardDataset(index)[0]
    before = rec.raw.copy()
    collate([rec])
    assert np.array_equal(rec.raw, before)
    with pytest.raises(ValueError):
        collate([])


@pytest.mark.unit
def test_batch_to_device_keeps_content(index):
    batch = collate([CellShardDataset(index)[0]])
    moved = batch_to_device(batch, torch.device("cpu"))
    assert moved is not batch and torch.equal(moved.x, batch.x)
    assert moved.crop_hw == batch.crop_hw
