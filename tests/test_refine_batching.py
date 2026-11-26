"""Groupes de raffinement de taille variable (E7, débit) : regroupement, réplication, invariance au regroupement."""
import numpy as np
import pytest
import torch

from encoder_eval import encode_cells
from profile_lut import LogWLutSpec, load_or_build_logw_lut
from refine_batching import make_groups, refine_grouped, replicate
from source_encoder import EncoderConfig, SourceEncoder

THRESHOLD = 0.1  # sous la probabilité initiale de l'encodeur aléatoire : il produit des sources
ITERS = 5
NRMSE_TOL = 1e-4


@pytest.fixture(scope="module")
def lut5(tmp_path_factory):
    return load_or_build_logw_lut(LogWLutSpec(n=32, steps=60, n_r=32), tmp_path_factory.mktemp("l"))["lut"][None, None]


@pytest.fixture
def cells(make_cell):
    return [make_cell(0, size=24), make_cell(1, size=32), make_cell(2, size=20),
            make_cell(3, blobs=((10.0, 10.0, 5.0, 3.0, 3.0),), size=24)]


@pytest.fixture
def encoded(cells):
    cfg = EncoderConfig(base_ch=4, k_train=16, cap=16)
    torch.manual_seed(0)
    return encode_cells(SourceEncoder(cfg).eval(), cfg, cells, torch.device("cpu"), threshold=THRESHOLD)


@pytest.mark.unit
def test_make_groups_sorts_by_shape_and_chunks():
    cells = [dict(id=i, rate=np.zeros(s)) for i, s in enumerate([(30, 30), (10, 10), (20, 20), (10, 12)])]
    groups = make_groups(cells, 3)
    assert [[c["id"] for c in g] for g in groups] == [[1, 3, 2], [0]]


@pytest.mark.unit
def test_make_groups_rejects_non_positive_size():
    with pytest.raises(ValueError):
        make_groups([], 0)


@pytest.mark.unit
def test_replicate_cycles_cells_with_fresh_ids_and_source_ids():
    cells = [dict(id=7, area=10), dict(id=9, area=20)]
    out = replicate(cells, 5)
    assert [c["id"] for c in out] == [0, 1, 2, 3, 4]
    assert [c["source_id"] for c in out] == [7, 9, 7, 9, 7]
    assert [c["area"] for c in out] == [10, 20, 10, 20, 10]
    assert cells[0]["id"] == 7 and "source_id" not in cells[0]


@pytest.mark.unit
def test_replicate_rejects_empty_input():
    with pytest.raises(ValueError):
        replicate([], 3)


@pytest.mark.unit
def test_refinement_result_does_not_depend_on_group_size(cells, encoded, lut5):
    small = refine_grouped(cells, encoded, 1, ITERS, lut5, torch.device("cpu"))
    large = refine_grouped(cells, encoded, len(cells), ITERS, lut5, torch.device("cpu"))
    assert set(small.nrmse) == set(large.nrmse) == {c["id"] for c in cells}
    for cid in small.nrmse:
        assert abs(small.nrmse[cid] - large.nrmse[cid]) <= NRMSE_TOL
    assert small.n_groups == len(cells) and large.n_groups == 1
    assert small.seconds_per_cell > 0 and large.seconds_per_cell > 0
