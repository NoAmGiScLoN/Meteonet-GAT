"""Tests de scripts/run_jetson_sessionA.sh : syntaxe bash et cohérence des options avec les parseurs argparse."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_jetson_sessionA.sh"
PYTHON_SCRIPTS = ("src/optimizer_check.py", "src/stability_test.py", "scripts/build_samples.py")
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
                  'STAGES="${STAGES:-0 1 2}"', "OUT=results/06_sessionA", "SRC=results/03_variable_sources",
                  "KEY_DEFAULT=ace17c5890"):
        assert token in text, token
    assert not re.search(r"^set -\w*e", text, flags=re.M)


@pytest.mark.integration
def test_every_fit_call_names_its_sample_or_pairs() -> None:
    # Arrange : sans --sample / --pairs, les scripts reconstruisent l'échantillon depuis DATA/ (absent sur le Jetson)
    calls = [ln for ln in script_text().replace("\\\n", " ").splitlines()
             if ln.lstrip().startswith("run ") and ("optimizer_check.py" in ln or "stability_test.py" in ln)]
    # Act
    missing = [ln.strip() for ln in calls if "--sample" not in ln and "--pairs" not in ln]
    # Assert
    assert len(calls) == 4 and missing == []
