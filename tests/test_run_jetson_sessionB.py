"""Tests de scripts/run_jetson_sessionB.sh : syntaxe bash et cohérence des options avec les parseurs argparse."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_jetson_sessionB.sh"
PYTHON_SCRIPTS = ("src/adaptive_run.py", "scripts/build_samples.py")
FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


def script_text() -> str:
    return SCRIPT.read_text()


def help_flags(path: str) -> set[str]:
    """Options exposées par `--help` d'un script du dépôt."""
    out = subprocess.run([sys.executable, str(ROOT / path), "--help"], capture_output=True, text=True,
                         check=True).stdout
    return set(FLAG.findall(out))


def commands_by_script() -> dict[str, set[str]]:
    """Options employées par le script shell pour chaque script python (lignes continuées jointes)."""
    flat = script_text().replace("\\\n", " ")
    used: dict[str, set[str]] = {p: set() for p in PYTHON_SCRIPTS}
    for line in flat.splitlines():
        for path in PYTHON_SCRIPTS:
            if path in line:
                used[path] |= set(FLAG.findall(line.split(path, 1)[1]))
    return used


def adaptive_run_lines() -> list[str]:
    return [ln.strip() for ln in script_text().replace("\\\n", " ").splitlines()
            if ln.lstrip().startswith("run ") and "adaptive_run.py" in ln]


@pytest.mark.integration
def test_script_is_executable_and_bash_syntax_is_valid():
    assert SCRIPT.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@pytest.mark.integration
def test_python_scripts_called_exist():
    for rel in re.findall(r"(?:src|scripts)/\w+\.py", script_text()):
        assert (ROOT / rel).is_file(), rel


@pytest.mark.integration
@pytest.mark.parametrize("path", PYTHON_SCRIPTS)
def test_flags_used_by_script_exist_in_argparse(path):
    used = commands_by_script()[path]
    assert used, f"aucune option repérée pour {path}"
    unknown = used - help_flags(path)
    assert not unknown, f"{path} ne connaît pas {sorted(unknown)}"


@pytest.mark.integration
def test_script_declares_expected_defaults():
    text = script_text()
    for token in ('PY="${PY:-python3}"', 'DEV="${DEV:-cuda}"', 'REQUIRE_CUDA="${REQUIRE_CUDA:-1}"',
                  'STAGES="${STAGES:-0 1 3 2 4}"', "OUT=results/07_sessionB", "SRC=results/03_variable_sources",
                  "KEY_DEFAULT=ace17c5890"):
        assert token in text, token
    assert not re.search(r"^set -\w*e", text, flags=re.M)


@pytest.mark.integration
def test_every_adaptive_run_call_names_its_sample() -> None:
    # Arrange : sans --sample, le script reconstruirait l'échantillon depuis DATA/ (absent sur le Jetson)
    calls = adaptive_run_lines()
    # Act
    missing = [ln for ln in calls if "--sample" not in ln]
    # Assert
    assert len(calls) == 6 and missing == []


LONG_STAGES = ("1_adaptive_2016", "3_floor256", "2_adaptive_2017", "5_adaptive_2016_c050", "6_adaptive_2017_c050")
C050_STAGES = {"5_adaptive_2016_c050": ("sample_40.pkl", "adaptive_2016_c050"),
               "6_adaptive_2017_c050": ("sample_40_2017.pkl", "adaptive_2017_c050")}


def calls_by_stage() -> dict[str, str]:
    return {ln.split()[1]: ln for ln in adaptive_run_lines()}


@pytest.mark.integration
def test_long_stages_are_resumable_and_smoke_is_not() -> None:
    # Arrange
    calls = calls_by_stage()
    # Act / Assert
    for name in LONG_STAGES:
        assert "--resume" in calls[name], name
    assert "--resume" not in calls["0_smoke"]


@pytest.mark.integration
@pytest.mark.parametrize("name", list(C050_STAGES))
def test_c050_rerun_uses_half_compression_and_skips_largest_class(name: str) -> None:
    # Arrange : D23 — compression 0.5 ; >=5000 px inchangé (la limite >= 178 > cap 128 ne lie jamais)
    sample, out = C050_STAGES[name]
    line = calls_by_stage()[name]
    # Act
    bins = line.split("--bins", 1)[1].split("--", 1)[0].split()
    # Assert
    assert "--compression-frac 0.5" in line and "--k-cap 128" in line
    assert f"$SRC/{sample}" in line and f"$OUT/{out}" in line
    assert bins == ["50-200", "200-1000", "1000-5000"]
    assert "--target" not in line  # cible D21 (0.1) par défaut


@pytest.mark.integration
@pytest.mark.parametrize("name", ["0_smoke", "1_adaptive_2016", "3_floor256", "2_adaptive_2017"])
def test_archived_first_run_stages_pin_compression_025(name: str) -> None:
    # Le défaut est passé à 0.5 (D24) : les étapes archivées doivent rester reproductibles et reprenables
    assert "--compression-frac 0.25" in calls_by_stage()[name]


@pytest.mark.integration
def test_stage_dispatch_and_summary_cover_c050_runs() -> None:
    text = script_text()
    assert "5) stage_5 ;;" in text and "6) stage_6 ;;" in text
    summary_loops = re.findall(r"for (?:d|run) in [^;:\n]*", text)
    assert summary_loops and all("adaptive_2016_c050" in s and "adaptive_2017_c050" in s for s in summary_loops)
