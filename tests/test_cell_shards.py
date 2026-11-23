"""Tests de cell_shards : recadrage des cellules, taux, masque invalide, aller-retour des shards."""
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import cell_shards as cs
from tracking_synth import disc_frame, rain_frame

T = datetime(2016, 2, 3, 12, 5)
SHAPE = (60, 80)


def _frame_two_cells() -> np.ndarray:
    R = disc_frame(SHAPE, (20, 20), 120)
    R2 = disc_frame(SHAPE, (40, 60), 400)
    return np.maximum(R, R2).astype(np.int16)


def _assert_same(a: cs.CellRecord, b: cs.CellRecord) -> None:
    assert (a.t, a.local_id, a.split, a.bin, a.area, a.y0, a.x0) == (b.t, b.local_id, b.split, b.bin, b.area, b.y0, b.x0)
    assert (a.cy, a.cx, a.border_flag, a.missing_flag) == (b.cy, b.cx, b.border_flag, b.missing_flag)
    assert a.raw.dtype == b.raw.dtype == np.int16
    np.testing.assert_array_equal(a.raw, b.raw)
    np.testing.assert_array_equal(a.cell_mask, b.cell_mask)
    np.testing.assert_array_equal(a.invalid, b.invalid)


@pytest.mark.unit
def test_bins_match_variable_source_fit():
    import variable_source_fit as vsf
    assert [(n, lo, hi) for n, lo, hi, _ in vsf.BINS] == [(n, lo, hi) for n, lo, hi in cs.BINS]
    assert vsf.MARGIN == cs.MARGIN


@pytest.mark.unit
@pytest.mark.parametrize("area,expected", [(49, None), (50, "50-200"), (199, "50-200"), (200, "200-1000"),
                                           (1000, "1000-5000"), (4999, "1000-5000"), (5000, ">=5000"), (10 ** 6, ">=5000")])
def test_bin_of_boundaries(area, expected):
    assert cs.bin_of(area) == expected


@pytest.mark.unit
def test_rate_equals_vsf_crop_cell():
    # Arrange
    import variable_source_fit as vsf
    from scipy import ndimage as ndi
    from cell_delimitation import segment
    R = _frame_two_cells()
    labels, _, _ = segment(R)
    objs = ndi.find_objects(labels)
    # Act
    recs = cs.extract_frame_cells(R, T, "train")
    # Assert
    assert len(recs) == 2
    for rec in recs:
        ref = vsf.crop_cell(R, labels, rec.local_id, objs[rec.local_id - 1], rec.area, "x", T)
        got = cs.rate_mm_h(rec)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, ref["rate"])
        assert (rec.y0, rec.x0) == (ref["y0"], ref["x0"])
        assert rec.cy == pytest.approx(ref["cy"]) and rec.cx == pytest.approx(ref["cx"])


@pytest.mark.unit
def test_rate_does_not_overflow_for_large_raw_values():
    R = np.zeros(SHAPE, dtype=np.int16)
    R[10:20, 10:20] = 5000
    rec = cs.extract_frame_cells(R, T, "train", min_area=50)[0]
    assert cs.rate_mm_h(rec).max() == pytest.approx(600.0)


@pytest.mark.unit
def test_record_fields_bin_and_split():
    rec = cs.extract_frame_cells(_frame_two_cells(), T, "val")[0]
    assert (rec.split, rec.t, rec.bin, rec.area) == ("val", T, "50-200", 120)
    assert rec.raw.dtype == np.int16 and rec.cell_mask.dtype == bool and rec.invalid.dtype == bool
    assert rec.raw.shape == rec.cell_mask.shape == rec.invalid.shape
    assert not rec.raw[~rec.cell_mask].any()  # zéro hors masque
    with pytest.raises(FrozenInstanceError):
        rec.area = 3  # type: ignore[misc]


@pytest.mark.unit
def test_min_area_filters_small_cells():
    R = rain_frame(SHAPE, [(30, 30, 2, 2, 60)])
    assert cs.extract_frame_cells(R, T, "train", min_area=50) == []


@pytest.mark.unit
def test_invalid_mask_marks_missing_pixels_inside_crop():
    # Arrange : cellule 20x20, trou -1 dans la marge du recadrage (hors cellule)
    R = np.zeros(SHAPE, dtype=np.int16)
    R[20:40, 20:40] = 50
    R[19, 19] = -1  # voisin diagonal de la cellule, dans la marge
    # Act
    rec = cs.extract_frame_cells(R, T, "train")[0]
    # Assert
    assert rec.invalid.sum() == 1 and rec.invalid[19 - rec.y0, 19 - rec.x0]
    assert not (rec.invalid & rec.cell_mask).any()
    assert rec.missing_flag


@pytest.mark.unit
def test_border_cell_is_clipped_at_frame_edge():
    R = np.zeros(SHAPE, dtype=np.int16)
    R[0:10, 0:10] = 50
    rec = cs.extract_frame_cells(R, T, "train")[0]
    assert (rec.y0, rec.x0) == (0, 0)
    assert rec.raw.shape == (13, 13)
    assert rec.border_flag
    R2 = np.zeros(SHAPE, dtype=np.int16)
    R2[55:60, 70:80] = 50
    rec2 = cs.extract_frame_cells(R2, T, "train", min_area=50)[0]
    assert rec2.y0 + rec2.raw.shape[0] == SHAPE[0] and rec2.x0 + rec2.raw.shape[1] == SHAPE[1]


@pytest.mark.unit
def test_keep_predicate_skips_cells_and_input_is_not_mutated():
    R = _frame_two_cells()
    before = R.copy()
    recs = cs.extract_frame_cells(R, T, "train", keep=lambda b, lid: b == ">=5000")
    assert recs == []
    recs = cs.extract_frame_cells(R, T, "train", keep=lambda b, lid: b == "200-1000")
    assert [r.bin for r in recs] == ["200-1000"]
    np.testing.assert_array_equal(R, before)


@pytest.mark.unit
def test_shard_round_trip_is_lossless(tmp_path: Path):
    # Arrange : formes impaires, trou -1, cellule au bord
    R = _frame_two_cells()
    R[5, 5] = -1
    R[0:7, 70:79] = 33
    recs = cs.extract_frame_cells(R, datetime(2016, 2, 3, 12, 5, 0, 250), "train")
    assert len(recs) == 3
    path = tmp_path / "s.shard"
    # Act
    cs.write_shard(path, recs)
    back = cs.read_shard(path)
    # Assert
    assert len(back) == len(recs)
    for a, b in zip(recs, back):
        _assert_same(a, b)


@pytest.mark.unit
def test_empty_shard_round_trip(tmp_path: Path):
    path = tmp_path / "e.shard"
    cs.write_shard(path, [])
    assert cs.read_shard(path) == []
    assert cs.read_shard_meta(path)["h"].shape == (0,)


@pytest.mark.unit
def test_write_is_atomic_leaves_no_tmp_and_replaces(tmp_path: Path):
    recs = cs.extract_frame_cells(_frame_two_cells(), T, "train")
    path = tmp_path / "a.shard"
    cs.write_shard(path, recs)
    cs.write_shard(path, recs[:1])
    assert [p.name for p in tmp_path.iterdir()] == ["a.shard"]
    assert len(cs.read_shard(path)) == 1


@pytest.mark.unit
def test_failed_write_leaves_no_tmp_and_keeps_old_file(tmp_path: Path):
    recs = cs.extract_frame_cells(_frame_two_cells(), T, "train")
    path = tmp_path / "a.shard"
    cs.write_shard(path, recs)
    with pytest.raises(Exception):
        cs.write_shard(path, [object()])  # type: ignore[list-item]
    assert [p.name for p in tmp_path.iterdir()] == ["a.shard"]
    assert len(cs.read_shard(path)) == 2


@pytest.mark.unit
def test_read_shard_meta_has_sampler_columns(tmp_path: Path):
    recs = cs.extract_frame_cells(_frame_two_cells(), T, "train")
    cs.write_shard(tmp_path / "a.shard", recs)
    meta = cs.read_shard_meta(tmp_path / "a.shard")
    assert list(meta["area"]) == [r.area for r in recs]
    assert list(meta["h"]) == [r.raw.shape[0] for r in recs]
    assert list(meta["bin"]) == [r.bin for r in recs]


@pytest.mark.unit
def test_fast_cell_stats_equal_describe_on_random_frame():
    # Arrange : bruit pluvieux, bord de carte et zone manquante
    from cell_delimitation import describe, segment
    rng = np.random.default_rng(0)
    R = np.where(rng.random((70, 90)) < 0.4, 40, 0).astype(np.int16)
    R[10:20, 30:50] = -1
    R[0:5, 0:5] = 40
    labels, n, valid = segment(R)
    # Act
    got = cs.cell_stats(labels, n, valid)
    ref = describe(R, labels, n, valid, T)
    # Assert
    np.testing.assert_array_equal(got["area"], ref.area_px.to_numpy())
    np.testing.assert_allclose(got["cy"], ref.cy.to_numpy())
    np.testing.assert_allclose(got["cx"], ref.cx.to_numpy())
    np.testing.assert_array_equal(got["border"], ref.border_flag.to_numpy())
    np.testing.assert_array_equal(got["missing"], ref.missing_neighbor_flag.to_numpy())


@pytest.mark.unit
def test_min_area_below_smallest_bin_is_rejected():
    with pytest.raises(ValueError, match="min_area"):
        cs.extract_frame_cells(_frame_two_cells(), T, "train", min_area=10)


@pytest.mark.unit
def test_packed_record_round_trip_and_smaller_than_unpacked():
    rec = cs.extract_frame_cells(_frame_two_cells(), T, "train")[1]
    packed = cs.pack_record(rec)
    assert packed.nbytes < rec.raw.nbytes + rec.cell_mask.nbytes + rec.invalid.nbytes
    _assert_same(rec, cs.unpack_record(packed))


@pytest.mark.unit
def test_open_shard_gives_memory_mapped_random_access(tmp_path: Path, monkeypatch):
    # Arrange
    R = _frame_two_cells()
    R[5, 5] = -1
    recs = cs.extract_frame_cells(R, T, "train")
    cs.write_shard(tmp_path / "a.shard", recs)
    monkeypatch.setattr(cs, "read_shard", lambda p: (_ for _ in ()).throw(AssertionError("shard entier lu")))
    # Act
    handle = cs.open_shard(tmp_path / "a.shard")
    # Assert : un enregistrement s'obtient par tranche d'un tableau mappé, sans lire le shard
    assert isinstance(handle.raw, np.memmap)
    assert len(handle) == len(recs)
    for i, rec in enumerate(recs):
        _assert_same(rec, handle.record(i))
    with pytest.raises(IndexError):
        handle.record(len(recs))
