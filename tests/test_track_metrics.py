"""Tests de track_metrics : métriques sur tables à la main, flicker, planche de revue, CLI."""
import json
import logging
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import track_metrics
import track_run
from track_metrics import (RunTables, compute_metrics, flicker_rates, flicker_table, load_run,
                           review_sheet)
from track_review import overlay, sample_events, write_review
from tracker_state import TrackerConfig, TrackerState
from tracking_synth import DEFAULT_SHAPE, rain_frame, write_npz_files

T0 = datetime(2016, 1, 1, 12, 0)
STEP = timedelta(minutes=5)


def _t(i: int) -> datetime:
    return T0 + STEP * i


def _det(i, tid, area, cy=10.0, cx=10.0, isum=1.0, v=(0.0, 0.0), reset=False, lid=1):
    return {"timestamp": _t(i), "local_id": lid, "area_px": area, "cy": cy, "cx": cx,
            "intensity_sum": isum, "track_id": tid, "v_y": v[0], "v_x": v[1],
            "velocity_reset": reset}


def _ev(i, tid, kind, area=100):
    return {"t": _t(i), "track_id": tid, "event": kind, "area": area}


def _tables(dets, evs) -> RunTables:
    return RunTables(detections=pd.DataFrame(dets), lineage=pd.DataFrame(),
                     events=pd.DataFrame(evs, columns=["t", "track_id", "event", "area"]))


def _run(frames, tid, area=60, isum=1.0):
    return [_det(i, tid, area, isum=isum) for i in frames]


@pytest.fixture
def hand_tables() -> RunTables:
    # cartes 0 et 14 : seulement des cellules non suivies (bornes de l'exécution)
    dets = []
    for i, (a, vx) in enumerate([(100, 1), (100, 1), (100, 1), (400, 2)]):  # piste 1 : 1..4
        dets.append(_det(i + 1, 1, a, isum=10, v=(0, vx), reset=(i == 3)))
    dets += [_det(i + 1, 2, 60, isum=5) for i in range(2)]  # piste 2 : 1..2
    dets += [_det(i + 1, 3, 300, isum=1, v=(1, 0)) for i in list(range(5)) + list(range(6, 13))]
    dets += [_det(0, 0, 80, isum=20, lid=2), _det(0, 0, 30, isum=100, lid=3),
             _det(14, 0, 30, isum=0.0, lid=2)]
    evs = [_ev(1, 1, "birth"), _ev(1, 2, "birth"), _ev(1, 3, "birth"),
           _ev(5, 1, "death"), _ev(3, 2, "coast", 60), _ev(4, 2, "death", 60),
           _ev(6, 3, "coast", 300)]
    return _tables(dets, evs)


def test_track_count_and_lengths(hand_tables):
    m = compute_metrics(hand_tables)
    assert m["n_tracks"] == 3 and m["n_censored"] == 0
    assert m["length"]["median"] == 4
    assert m["length"]["share_le_2"] == pytest.approx(1 / 3)
    assert m["length"]["share_ge_12"] == pytest.approx(1 / 3)
    assert m["length_all"] == m["length"]


def test_lengths_by_max_area_class(hand_tables):
    by = compute_metrics(hand_tables)["by_class"]
    assert by["50-200"]["n_tracks"] == 1 and by["50-200"]["median"] == 2
    assert by["200-1000"]["n_tracks"] == 2 and by["200-1000"]["median"] == 8
    assert by["1000-5000"]["n_tracks"] == 0
    assert math.isnan(by[">=5000"]["median"])


def test_events_per_1000_tracks(hand_tables):
    per = compute_metrics(hand_tables)["events_per_1000_tracks"]
    assert per["birth"] == pytest.approx(1000)
    assert per["death"] == pytest.approx(2000 / 3)
    assert per["coast"] == pytest.approx(2000 / 3)
    assert per["merge_end"] == 0


def test_rain_mass_share_in_long_tracks(hand_tables):
    # cellules >= 50 px : 40 + 10 + 12 + 20 = 82 ; pistes > 3 cartes : 40 + 12 = 52
    assert compute_metrics(hand_tables)["mass_share_long_tracks"] == pytest.approx(52 / 82)


def test_identity_switch_proxies_use_consecutive_pairs_only(hand_tables):
    m = compute_metrics(hand_tables)["switch"]
    # 3 + 1 + 10 paires à un pas ; la piste 3 a une paire à cheval sur un trou
    assert m["n_pairs"] == 14 and m["n_pairs_across_gap"] == 1
    assert m["share_area_ratio"] == pytest.approx(1 / 14)
    assert m["share_velocity_reset"] == pytest.approx(1 / 14)
    assert m["median_abs_dv"] == 0.0


def test_coast_success_rate(hand_tables):
    m = compute_metrics(hand_tables)["coast"]
    assert (m["n"], m["n_continued"]) == (2, 1)
    assert m["success_rate"] == pytest.approx(0.5)


def test_coast_followed_by_missing_frame_is_excluded():
    tables = _tables(_run(range(3), 5) + _run([5], 5), [_ev(3, 5, "coast")])
    m = compute_metrics(tables)["coast"]
    assert (m["n"], m["n_excluded"]) == (0, 1) and math.isnan(m["success_rate"])


def test_coast_chain_takes_outcome_of_next_record():
    cont = _tables(_run(range(3), 5) + _run([5], 5), [_ev(3, 5, "coast"), _ev(4, 5, "coast")])
    dead = _tables(_run(range(3), 5), [_ev(3, 5, "coast"), _ev(4, 5, "coast"),
                                       _ev(5, 5, "death")])
    assert compute_metrics(cont)["coast"]["n_continued"] == 2
    m = compute_metrics(dead)["coast"]
    assert (m["n"], m["n_continued"]) == (2, 0)


def test_coast_on_last_frame_of_run_is_excluded():
    m = compute_metrics(_tables(_run(range(3), 5), [_ev(3, 5, "coast")]))["coast"]
    assert (m["n"], m["n_excluded"]) == (0, 1)


# --- censure -----------------------------------------------------------------------------------
def test_censored_tracks_are_flagged_and_excluded_from_headline_lengths():
    dets = (_run(range(0, 5), 1) + _run(range(1, 4), 2) + _run(range(3, 7), 3)
            + _run(range(2, 4), 4) + _run(range(4, 6), 5))
    # 1 : 1re carte ; 3 : dernière carte ; 4 : coupée par un gap_break ; 5 : née à la carte
    # du gap_break
    evs = [_ev(4, 4, "gap_break")]
    m = compute_metrics(_tables(dets, evs))
    assert m["n_tracks"] == 5 and m["n_censored"] == 4
    assert m["length"]["n_tracks"] == 1 and m["length"]["median"] == 3
    assert m["length_all"]["n_tracks"] == 5
    assert m["length"]["share_le_2"] == 0.0
    assert m["length_all"]["share_le_2"] == pytest.approx(2 / 5)
    assert m["by_class"]["50-200"]["n_tracks"] == 1
    assert m["by_class_all"]["50-200"]["n_tracks"] == 5


def test_mass_share_drops_short_censored_tracks_but_keeps_long_ones():
    dets = (_run([0, 1], 1, isum=10)               # censurée (1re carte), 2 cartes : exclue
            + _run(range(1, 6), 2, isum=1)         # longue, non censurée
            + _run([2, 3], 3, isum=1)              # courte, non censurée : au dénominateur
            + _run(range(0, 4), 4, isum=100)       # censurée mais 4 cartes : certainement longue
            + [_det(7, 0, 30, lid=2)])
    share = compute_metrics(_tables(dets, []))["mass_share_long_tracks"]
    assert share == pytest.approx(405 / 407)


# --- flicker -----------------------------------------------------------------------------------
def _flicker_tables(dead, cands, steps=2) -> RunTables:
    """Pistes mortes `dead` = [(tid, cx_dernière_obs, v_x)] : dernière obs. à la carte 3,
    coasts jusqu'à la carte 3 + steps - 1, mort à 3 + steps. `cands` = (track_id, aire, cx)
    des détections de la carte de mort (track_id < 0 : née cette carte, id = -track_id)."""
    dets, evs, t_death = [], [], 3 + steps
    for tid, cx, vx in dead:
        dets += [_det(i, tid, 100, cx=cx - vx * (3 - i), v=(0, vx), lid=tid) for i in range(4)]
        evs += [_ev(3 + k, tid, "coast") for k in range(1, steps)] + [_ev(t_death, tid, "death")]
    for k, (tid, area, cx) in enumerate(cands):
        real = abs(tid)
        dets.append(_det(t_death, real, area, cy=10, cx=cx, lid=100 + k))
        if tid < 0:
            evs.append(_ev(t_death, real, "birth", area))
    return _tables(dets, evs)


def test_flicker_counts_birth_near_prediction():
    table = flicker_table(_flicker_tables([(1, 16, 2)], [(-2, 60, 22)]))
    assert len(table) == 1 and bool(table.recovered.iloc[0])
    assert flicker_rates(table)["overall"] == pytest.approx(1.0)


def test_flicker_ignores_birth_far_from_prediction():
    table = flicker_table(_flicker_tables([(1, 16, 2)], [(-2, 60, 70)]))
    rates = flicker_rates(table)
    assert not bool(table.recovered.iloc[0])
    assert rates["overall"] == 0.0 and rates["by_class"]["50-200"] == 0.0


def test_flicker_uses_two_steps_velocity_from_last_observation():
    # v = 30 px/pas, dernière obs. x = 10 : x2 -> 70 (compté) ; x1 ou position dérivée -> 40
    hit = flicker_table(_flicker_tables([(1, 10, 30)], [(-2, 60, 70)]))
    miss = flicker_table(_flicker_tables([(1, 10, 30)], [(-2, 60, 40)]))
    assert bool(hit.recovered.iloc[0]) and not bool(miss.recovered.iloc[0])


def test_flicker_elapsed_steps_follow_max_coast_two():
    tables = _flicker_tables([(1, 10, 30)], [(-2, 60, 100)], steps=3)
    wrong = _flicker_tables([(1, 10, 30)], [(-2, 60, 70)], steps=3)
    assert bool(flicker_table(tables).recovered.iloc[0])
    assert not bool(flicker_table(wrong).recovered.iloc[0])


@pytest.mark.parametrize("track_id,area,expected", [
    (0, 40, True),      # non suivie 35-49 px (sous le seuil de naissance)
    (-2, 40, True),     # née cette carte (suite d'une scission, hystérésis...)
    (0, 30, False),     # sous 35 px
    (9, 60, False),     # piste déjà existante, pas une naissance
])
def test_flicker_candidates_rules(track_id, area, expected):
    table = flicker_table(_flicker_tables([(1, 16, 2)], [(track_id, area, 21)]))
    assert bool(table.recovered.iloc[0]) is expected


def test_flicker_candidate_recovers_at_most_one_dead_track():
    # prédictions x = 20 et x = 26 ; une seule candidate en x = 25 : va à la piste 2 (plus proche)
    table = flicker_table(_flicker_tables([(1, 16, 2), (2, 22, 2)], [(-3, 60, 25)]))
    recovered = dict(zip(table.track_id, table.recovered))
    assert recovered == {1: False, 2: True}
    assert flicker_rates(table)["overall"] == pytest.approx(0.5)


def test_flicker_empty_when_no_death_after_coast(hand_tables):
    only = RunTables(hand_tables.detections, hand_tables.lineage,
                     hand_tables.events[hand_tables.events.event == "birth"])
    assert len(flicker_table(only)) == 0
    assert math.isnan(flicker_rates(flicker_table(only))["overall"])


# --- run synthétique sur disque ------------------------------------------------------------------
def _write_run(root, out, n=14):
    dates = [T0 + STEP * i for i in range(n)]
    maps = []
    for i in range(n):
        blobs = [] if i in (6, 7) else [(30, 10 + 2 * i, 6, 6, 60)]
        blobs.append((10, 50 + 5 * i, 5, 5, 60))
        maps.append(rain_frame(DEFAULT_SHAPE, blobs))
    write_npz_files(root, [(2016, 1, 0, dates, maps)])
    track_run.run_month(2016, 1, TrackerConfig(), out, TrackerState(), None, root,
                        start_time=dates[0], end_time=dates[-1])


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory):
    root, out = tmp_path_factory.mktemp("meteonet"), tmp_path_factory.mktemp("out")
    _write_run(root, out)
    return root, out


def test_load_run_reads_needed_columns_only(fake_run):
    _, out = fake_run
    tables = load_run(out)
    assert set(tables.detections.columns) < set(track_run.COLUMNS["detections"])
    assert "notes" not in tables.events.columns
    assert tables.lineage.empty and tables.detections.track_id.max() > 0
    assert len(load_run(out, months=["2016-01"]).events) == len(tables.events)
    full = load_run(out, with_lineage=True)
    assert list(full.lineage.columns) == track_run.COLUMNS["lineage"]


def test_load_run_without_done_month_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path)


def test_track_spanning_two_months_is_one_track(tmp_path):
    n, start = 12, datetime(2016, 1, 31, 23, 30)
    dates = [start + STEP * i for i in range(n)]
    maps = [rain_frame(DEFAULT_SHAPE, [(30, 10 + 2 * i, 6, 6, 60)]) for i in range(n)]
    jan = [d for d in dates if d.month == 1]
    write_npz_files(tmp_path / "raw", [(2016, 1, 0, jan, maps[:len(jan)]),
                                       (2016, 2, 0, dates[len(jan):], maps[len(jan):])])
    out, cfg = tmp_path / "out", TrackerConfig()
    state, prev, _ = track_run.run_month(2016, 1, cfg, out, TrackerState(), None,
                                         tmp_path / "raw", end_time=dates[-1])
    track_run.run_month(2016, 2, cfg, out, state, prev, tmp_path / "raw", end_time=dates[-1])
    m = compute_metrics(load_run(out))
    assert m["n_tracks"] == 1 and m["length_all"]["median"] == n and m["n_censored"] == 1


# --- planche de revue ----------------------------------------------------------------------------
def test_sample_events_is_stratified_with_sparse_type():
    events = pd.DataFrame({"event": ["birth"] * 20 + ["death"], "t": [_t(i) for i in range(21)],
                           "track_id": range(21), "area": 50})
    counts = sample_events(events, 5, seed=0).event.value_counts().to_dict()
    assert counts == {"birth": 4, "death": 1}
    assert len(sample_events(events, 100, seed=0)) == 21


def test_review_sheet_stratified_pages_and_csv(fake_run, tmp_path):
    root, out = fake_run
    types = set(load_run(out).events.event)
    png = tmp_path / "rev" / "review.png"
    sample = review_sheet(out, root, n=4, seed=0, png_path=png)
    assert len(sample) == 4
    assert len({s.event for s in sample}) == min(4, len(types))
    csv = pd.read_csv(png.parent / "review_events.csv")
    assert len(csv) == 4 and list(csv.columns) == ["event", "t", "track_id", "area", "page", "ok"]
    assert csv.ok.isna().all()
    assert len(list(png.parent.glob("review_*.png"))) == 1


def test_review_sheet_several_pages(fake_run, tmp_path):
    root, out = fake_run
    total = len(load_run(out).events)
    sample = review_sheet(out, root, n=1000, seed=1, png_path=tmp_path / "review.png")
    assert len(sample) == total
    pages = sorted(tmp_path.glob("review_*.png"))
    assert len(pages) == math.ceil(total / track_metrics.EVENTS_PER_PAGE)
    assert all(p.stat().st_size > 0 for p in pages)


def test_review_sheet_is_reproducible(fake_run, tmp_path):
    root, out = fake_run
    a = review_sheet(out, root, n=5, seed=3, png_path=tmp_path / "a.png")
    b = review_sheet(out, root, n=5, seed=3, png_path=tmp_path / "b.png")
    assert [(s.event, s.t, s.track_id) for s in a] == [(s.event, s.t, s.track_id) for s in b]


def test_review_csv_rerun_keeps_filled_ok_and_backs_up_unmatched(fake_run, tmp_path):
    root, out = fake_run
    png = tmp_path / "review.png"
    review_sheet(out, root, n=6, seed=0, png_path=png)
    path = tmp_path / "review_events.csv"
    first = pd.read_csv(path)
    first["ok"] = "y"
    first.to_csv(path, index=False)
    review_sheet(out, root, n=6, seed=0, png_path=png)
    assert (pd.read_csv(path).ok == "y").all()
    review_sheet(out, root, n=2, seed=0, png_path=png)
    backups = list(tmp_path.glob("review_events.prev*.csv"))
    assert len(backups) == 1 and (pd.read_csv(backups[0]).ok == "y").all()
    assert len(pd.read_csv(path)) == 2


def test_overlay_skips_stale_local_ids_with_warning(caplog):
    labels = np.zeros((8, 8), dtype=int)
    labels[2:5, 2:5] = 1
    dets = pd.DataFrame({"local_id": [1, 5], "track_id": [3, 4]})
    with caplog.at_level(logging.WARNING):
        image = overlay(labels, dets, track_id=3)
    assert image.shape == (8, 8, 4) and image[..., 3].max() > 0
    assert "local_id" in caplog.text


def test_write_review_without_frames_still_writes_pages(fake_run, tmp_path):
    _, out = fake_run
    tables = load_run(out)
    sample = write_review(tables.events, tables.detections, tmp_path / "nodata", 3, 0,
                          tmp_path / "r" / "review.png")
    assert len(sample) == 3 and (tmp_path / "r" / "review_01.png").exists()


def test_cli_end_to_end(fake_run):
    root, out = fake_run
    code = track_metrics.main([str(out), "--months", "2016-01", "--review-n", "6",
                               "--data-dir", str(root)])
    report = out / "_report"
    assert code == 0
    metrics = json.loads((report / "metrics.json").read_text())
    assert metrics["n_tracks"] > 0 and "flicker" in metrics and "length_all" in metrics
    assert (report / "metrics_by_class.csv").exists() and (report / "flicker.csv").exists()
    assert (report / "review_events.csv").exists() and list(report.glob("review_*.png"))
