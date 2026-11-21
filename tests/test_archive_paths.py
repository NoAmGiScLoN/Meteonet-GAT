"""Garde-fous de l'arborescence archive/ : les scripts lisent les bons chemins, aucun payload ignoré n'est suivi."""
import ast
import fnmatch
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "archive"
ARCHIVE_RESULT_DIRS = [
    "archive/E0_segmentation_tracking/results",
    "archive/E1_fixed_vectors/results",
    "archive/E2_variable_sources/01_cpu_prelim/results",
    "archive/E2_variable_sources/02_jetson_baseline/results",
]
IGNORED_PAYLOAD_PATTERNS = ["diffusion_profiles*.pt", "sample_*.pkl", "stability_pairs_*.pkl", "cells.npz"]
FITS_RUNS = ["main", "ablation", "season2017", "restart_seed1"]  # runs avec un fits_<classe>.csv par classe
N_SIZE_BINS = 4


def tracked_archive_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "archive"], cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


@pytest.fixture(scope="module")
def make_figures():
    import make_figures as module
    return module


@pytest.mark.integration
def test_make_figures_input_files_exist(make_figures) -> None:
    # Arrange
    inputs = [
        make_figures.E1_RES / "results.json",
        make_figures.E1_RES / "results_jetson.json",
        make_figures.E1_RES / "cnn_encoder_schematic.png",
        make_figures.E0_RES / "S2_overlay.png",
        make_figures.E0_RES / "T1_tracks.png",
        make_figures.E2_JETSON_RES / "main" / "snapshots.pkl",
        make_figures.E2_JETSON_RES / "stability" / "stability.csv",
    ]
    # Act
    missing = [str(p) for p in inputs if not p.exists()]
    fit_counts = {run: len(list((make_figures.E2_JETSON_RES / run).glob("fits_*.csv"))) for run in FITS_RUNS}
    # Assert
    assert missing == []
    assert fit_counts == {run: N_SIZE_BINS for run in FITS_RUNS}


@pytest.mark.unit
def test_pipeline_figure_detections_path_exists() -> None:
    # Arrange : pipeline_figure importe torch/src au chargement, on lit donc la constante par analyse syntaxique
    tree = ast.parse((ROOT / "scripts" / "pipeline_figure.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "DETECTIONS")
    # Act : évalue l'expression (ROOT / "..." / ...) avec le ROOT du dépôt
    expr = ast.get_source_segment((ROOT / "scripts" / "pipeline_figure.py").read_text(), node.value)
    path = eval(compile(expr, "<DETECTIONS>", "eval"), {"__builtins__": {}}, {"ROOT": ROOT})
    # Assert
    assert path.is_file(), path


@pytest.mark.integration
def test_no_ignored_payload_is_tracked_in_archive() -> None:
    # Arrange
    tracked = tracked_archive_files()
    # Act
    offenders = [f for f in tracked for pat in IGNORED_PAYLOAD_PATTERNS if fnmatch.fnmatch(Path(f).name, pat)]
    # Assert
    assert offenders == []


@pytest.mark.integration
@pytest.mark.parametrize("folder", ARCHIVE_RESULT_DIRS)
def test_archive_results_folder_is_non_empty(folder: str) -> None:
    # Arrange
    path = ROOT / folder
    # Act
    files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else []
    # Assert
    assert files, f"{folder} manquant ou vide"
