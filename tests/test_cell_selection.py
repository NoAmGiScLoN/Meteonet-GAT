"""Tests de cell_selection : sous-échantillonnage bottom-k déterministe, pas de trames par classe."""
import random
from datetime import datetime, timedelta

import numpy as np
import pytest

import cell_selection as sel
from cell_shards import CellRecord, extract_frame_cells

ORIGIN = datetime(2016, 2, 3)


def _rec(t: datetime, lid: int, split: str = "train", bin_: str = "50-200") -> CellRecord:
    m = np.ones((2, 2), bool)
    return CellRecord(t=t, local_id=lid, split=split, bin=bin_, area=60, y0=0, x0=0, cy=0.5, cx=0.5,
                      border_flag=False, missing_flag=False, raw=np.full((2, 2), 20, np.int16),
                      cell_mask=m, invalid=np.zeros((2, 2), bool))


def _stream(n: int = 300) -> list[CellRecord]:
    return [_rec(ORIGIN + timedelta(minutes=5 * (i // 3)), i % 3 + 1) for i in range(n)]


def _ids(recs):
    return [(r.t, r.local_id) for r in recs]


@pytest.mark.unit
def test_default_quotas_match_plan():
    q = sel.default_quotas()
    assert q[("train", "50-200")] == 80000 and q[("train", ">=5000")] == 20000
    assert sum(v for (s, _), v in q.items() if s == "val") == 12500
    assert all(v == 0 for (s, _), v in q.items() if s == "test")


@pytest.mark.unit
def test_default_frame_steps():
    assert sel.DEFAULT_FRAME_STEPS == {"50-200": 6, "200-1000": 6, "1000-5000": 3, ">=5000": 2}


@pytest.mark.unit
def test_cell_key_depends_on_salt_time_and_id():
    k = sel.cell_key("a", ORIGIN, 1)
    assert k == sel.cell_key("a", ORIGIN, 1)
    assert k != sel.cell_key("b", ORIGIN, 1) and k != sel.cell_key("a", ORIGIN, 2)
    assert k != sel.cell_key("a", ORIGIN + timedelta(minutes=5), 1)


@pytest.mark.unit
def test_bottom_k_respects_quota_and_is_order_independent():
    # Arrange
    stream = _stream()
    quotas = {("train", "50-200"): 17}
    a = sel.CellSelector(quotas, "s")
    for r in stream:
        a.offer(r)
    shuffled = list(stream)
    random.Random(1).shuffle(shuffled)
    b = sel.CellSelector(quotas, "s")
    for r in shuffled:
        b.offer(r)
    # Assert
    assert len(a.selected("train", "50-200")) == 17
    assert _ids(a.selected("train", "50-200")) == _ids(b.selected("train", "50-200"))


@pytest.mark.unit
def test_bottom_k_keeps_exactly_the_smallest_hashes():
    stream = _stream(60)
    s = sel.CellSelector({("train", "50-200"): 10}, "z")
    for r in stream:
        s.offer(r)
    expected = sorted(stream, key=lambda r: sel.cell_key("z", r.t, r.local_id))[:10]
    assert set(_ids(s.selected("train", "50-200"))) == set(_ids(expected))


@pytest.mark.unit
def test_selector_ignores_zero_quota_and_unknown_class_and_tracks_counts():
    s = sel.CellSelector({("train", "50-200"): 2, ("test", "50-200"): 0}, "s")
    assert not s.offer(_rec(ORIGIN, 1, split="test"))
    assert not s.would_accept("val", "50-200", ORIGIN, 1)
    for r in _stream(10):
        s.offer(r)
    assert s.counts() == {("train", "50-200"): 2}


@pytest.mark.unit
def test_would_accept_agrees_with_offer():
    s = sel.CellSelector({("train", "50-200"): 5}, "s")
    for r in _stream(50):
        predicted = s.would_accept(r.split, r.bin, r.t, r.local_id)
        assert s.offer(r) == predicted


@pytest.mark.unit
def test_selection_is_independent_of_candidate_filtering_by_would_accept():
    stream = _stream(100)
    full = sel.CellSelector({("train", "50-200"): 7}, "s")
    for r in stream:
        full.offer(r)
    lazy = sel.CellSelector({("train", "50-200"): 7}, "s")
    for r in stream:
        if lazy.would_accept(r.split, r.bin, r.t, r.local_id):
            lazy.offer(r)
    assert _ids(full.selected("train", "50-200")) == _ids(lazy.selected("train", "50-200"))


@pytest.mark.unit
def test_frame_index_uses_time_not_iteration_count():
    assert sel.frame_index(ORIGIN, ORIGIN) == 0
    assert sel.frame_index(ORIGIN + timedelta(minutes=30), ORIGIN) == 6
    assert sel.frame_index(ORIGIN + timedelta(days=1), ORIGIN) == 288  # un trou ne décale pas l'indice


@pytest.mark.unit
def test_frame_index_rejects_off_grid_time():
    with pytest.raises(ValueError):
        sel.frame_index(ORIGIN + timedelta(minutes=3), ORIGIN)


@pytest.mark.unit
@pytest.mark.parametrize("idx,expected", [
    (0, {"50-200", "200-1000", "1000-5000", ">=5000"}),
    (1, set()),
    (2, {">=5000"}),
    (3, {"1000-5000"}),
    (4, {">=5000"}),
    (6, {"50-200", "200-1000", "1000-5000", ">=5000"}),
])
def test_bins_due_per_frame_step(idx, expected):
    assert sel.bins_due(idx, sel.DEFAULT_FRAME_STEPS) == frozenset(expected)


@pytest.mark.unit
def test_frame_filter_via_keep_predicate_only_keeps_due_bins():
    R = np.zeros((60, 80), np.int16)
    R[5:15, 5:15] = 50          # 100 px -> 50-200
    R[30:50, 20:60] = 50        # 800 px -> 200-1000
    due = sel.bins_due(2, sel.DEFAULT_FRAME_STEPS)
    recs = extract_frame_cells(R, ORIGIN, "train", keep=lambda b, lid: b in due)
    assert recs == []
    due = sel.bins_due(6, sel.DEFAULT_FRAME_STEPS)
    recs = extract_frame_cells(R, ORIGIN, "train", keep=lambda b, lid: b in due)
    assert {r.bin for r in recs} == {"50-200", "200-1000"}


@pytest.mark.unit
def test_selector_keeps_records_packed_and_tracks_bytes():
    s = sel.CellSelector({("train", "50-200"): 3}, "s")
    assert s.nbytes() == 0
    for r in _stream(30):
        s.offer(r)
    assert 0 < s.nbytes() < 3 * 2000
    kept = s.selected("train", "50-200")
    assert all(r.cell_mask.dtype == bool and r.cell_mask.shape == (2, 2) for r in kept)


@pytest.mark.unit
def test_selector_bytes_are_released_on_eviction():
    s = sel.CellSelector({("train", "50-200"): 2}, "s")
    for r in _stream(40):
        s.offer(r)
    two = s.nbytes()
    s2 = sel.CellSelector({("train", "50-200"): 2}, "s")
    for r in _stream(2):
        s2.offer(r)
    assert two == s2.nbytes()


@pytest.mark.unit
def test_selector_state_round_trip_gives_same_final_selection():
    stream = _stream(120)
    quotas = {("train", "50-200"): 9}
    whole = sel.CellSelector(quotas, "s")
    for r in stream:
        whole.offer(r)
    first = sel.CellSelector(quotas, "s")
    for r in stream[:50]:
        first.offer(r)
    resumed = sel.CellSelector.from_state(quotas, "s", first.state())
    for r in stream[50:]:
        resumed.offer(r)
    assert _ids(resumed.selected("train", "50-200")) == _ids(whole.selected("train", "50-200"))
    assert resumed.nbytes() == whole.nbytes()
