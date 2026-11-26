"""Tests de ShardIndex et CellShardDataset : index par enregistrement, accès paresseux avec cache LRU."""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import cell_shards as cs
from cell_shard_dataset import CellShardDataset, ShardIndex

T = datetime(2016, 2, 3)


def _recs(n: int, split: str, side: int) -> list[cs.CellRecord]:
    out = []
    for i in range(n):
        R = np.zeros((40, 40), np.int16)
        R[5:5 + side, 5:5 + side] = 20 + i
        rec = cs.extract_frame_cells(R, T + timedelta(minutes=5 * i), split)[0]
        out.append(rec)
    return out


def _make_dir(tmp_path: Path) -> Path:
    cs.write_shard(tmp_path / "train_50-200_0000.shard", _recs(3, "train", 9))
    cs.write_shard(tmp_path / "train_50-200_0001.shard", _recs(2, "train", 10))
    cs.write_shard(tmp_path / "val_200-1000_0000.shard", _recs(2, "val", 20))
    (tmp_path / "index.csv").write_text("ignoré")
    return tmp_path


@pytest.mark.unit
def test_index_lists_every_record(tmp_path: Path):
    idx = ShardIndex.from_dir(_make_dir(tmp_path))
    assert len(idx) == 7
    frame = idx.to_frame()
    assert list(frame.columns) == ["shard", "position", "split", "bin", "area", "h", "w"]
    assert frame.groupby("split").size().to_dict() == {"train": 5, "val": 2}
    assert idx.entries[0].shard == "train_50-200_0000.shard" and idx.entries[0].position == 0


@pytest.mark.unit
def test_index_filter_by_split_and_bin(tmp_path: Path):
    idx = ShardIndex.from_dir(_make_dir(tmp_path))
    assert len(idx.filter(split="train")) == 5
    assert len(idx.filter(bin="200-1000")) == 2
    assert len(idx.filter(split="val", bin="50-200")) == 0


@pytest.mark.unit
def test_empty_directory_gives_empty_index(tmp_path: Path):
    assert len(ShardIndex.from_dir(tmp_path)) == 0


@pytest.mark.unit
def test_dataset_returns_records_matching_the_shards(tmp_path: Path):
    d = _make_dir(tmp_path)
    ds = CellShardDataset(ShardIndex.from_dir(d))
    assert len(ds) == 7
    rec = ds[3]  # 1er enregistrement du 2e shard
    assert rec.area == 100 and rec.raw.max() == 20
    assert ds[6].split == "val"
    with pytest.raises(IndexError):
        ds[7]


@pytest.mark.unit
def test_dataset_random_access_never_reads_whole_shards(tmp_path: Path, monkeypatch):
    # Arrange
    d = _make_dir(tmp_path)
    import cell_shard_dataset as mod
    monkeypatch.setattr(cs, "read_shard", lambda p: (_ for _ in ()).throw(AssertionError("shard entier lu")))
    opened = []
    real = cs.open_shard
    monkeypatch.setattr(mod, "open_shard", lambda p: (opened.append(p.name), real(p))[1])
    ds = CellShardDataset(ShardIndex.from_dir(d), cache_size=1)
    # Act
    recs = [ds[i] for i in (6, 0, 4, 3, 5)]
    # Assert : les enregistrements sont corrects et les poignées mappées sont rouvertes selon le LRU
    assert [r.area for r in recs] == [400, 81, 100, 100, 400]
    assert opened == ["val_200-1000_0000.shard", "train_50-200_0000.shard", "train_50-200_0001.shard",
                      "val_200-1000_0000.shard"]  # indices 4 puis 3 : même shard, une seule ouverture
