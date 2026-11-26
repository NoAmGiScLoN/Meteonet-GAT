"""Tests de scripts/run_jetson_e7.sh : syntaxe bash et cohérence des options avec les parseurs argparse."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_jetson_e7.sh"
PYTHON_SCRIPTS = ("src/encoder_train.py", "src/encoder_eval.py", "scripts/build_samples.py")
FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


def script_text() -> str:
    return SCRIPT.read_text()


def flat_lines() -> list[str]:
    return script_text().replace("\\\n", " ").splitlines()


def help_flags(path: str) -> set[str]:
    out = subprocess.run([sys.executable, str(ROOT / path), "--help"], capture_output=True, text=True,
                         check=True).stdout
    return set(FLAG.findall(out))


def flags_used(path: str) -> set[str]:
    used: set[str] = set()
    for line in flat_lines():
        if path in line:
            used |= set(FLAG.findall(line.split(path, 1)[1]))
    return used


def run_calls() -> dict[str, str]:
    """Appels `run <nom> ...` (lignes continuées jointes), indexés par nom d'étape."""
    found = {}
    for ln in flat_lines():
        m = re.match(r"\s*(?:TRIES=\d+ )?run (\w+) ", ln)
        if m:
            found[m.group(1)] = ln.strip()
    return found


@pytest.mark.integration
def test_script_is_executable_and_bash_syntax_is_valid() -> None:
    assert SCRIPT.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@pytest.mark.integration
def test_python_scripts_called_exist() -> None:
    referenced = re.findall(r"(?:src|scripts)/\w+\.py", script_text())
    assert referenced
    for rel in referenced:
        assert (ROOT / rel).is_file(), rel


@pytest.mark.integration
@pytest.mark.parametrize("path", PYTHON_SCRIPTS)
def test_flags_used_by_script_exist_in_argparse(path: str) -> None:
    used = flags_used(path)
    assert used, f"aucune option repérée pour {path}"
    unknown = used - help_flags(path)
    assert not unknown, f"{path} ne connaît pas {sorted(unknown)}"


@pytest.mark.integration
def test_script_declares_expected_defaults() -> None:
    text = script_text()
    for token in ('PY="${PY:-python3}"', 'DEV="${DEV:-cuda}"', 'REQUIRE_CUDA="${REQUIRE_CUDA:-1}"',
                  'STAGES="${STAGES:-0 1 2 3 4}"', 'EPOCHS="${EPOCHS:-20}"', 'SHARDS="${SHARDS:-results/08_e7_cells}"',
                  "OUT=results/10_e7", "SRC=results/03_variable_sources", "KEY_DEFAULT=ace17c5890"):
        assert token in text, token
    assert not re.search(r"^set -\w*e", text, flags=re.M)
    assert "set -uo pipefail" in text


@pytest.mark.integration
def test_usage_header_documents_nohup_and_rsync_with_placeholder_host() -> None:
    head = "\n".join(script_text().splitlines()[:40])
    assert "nohup bash scripts/run_jetson_e7.sh" in head
    # les shards sont extraits sur le Mac, dans le dépôt principal : source absolue, hôte et utilisateur en marqueurs
    assert ("rsync -a --partial /Users/noam/Documents/RESEARCH/GAT/results/08_e7_cells/ "
            "<user>@<jetson-host>:<repo>/results/08_e7_cells/") in head
    assert not re.search(r"@\d+\.\d+\.\d+\.\d+", head) and "noam@" not in head


@pytest.mark.integration
def test_smoke_wipes_its_folder_and_never_resumes() -> None:
    # encoder_train n'a pas --overwrite : le smoke repart d'un dossier vide ; --resume refuserait un dossier propre
    text = script_text()
    call = run_calls()["0_smoke_train"]
    assert 'rm -rf "$OUT/smoke"' in text
    assert "--resume" not in call and "--max-steps" in call and '--out "$OUT/smoke"' in call
    assert "--iters 0" in run_calls()["0_smoke_infer"]


@pytest.mark.integration
def test_train_stage_is_resumable_and_uses_epochs_variable() -> None:
    call = run_calls()["2_train"]
    assert "--resume" in call and '--epochs "$EPOCHS"' in call and '--out "$TRAIN_DIR"' in call
    assert '--shards "$SHARDS"' in call and '--device "$DEV"' in call


@pytest.mark.integration
def test_train_stage_supports_warm_start_and_train_dir_overrides() -> None:
    # Reprise à chaud depuis un point de reprise (poids seuls) dans un nouveau dossier (correctif du portail)
    text = script_text()
    call = run_calls()["2_train"]
    assert 'TRAIN_DIR="${TRAIN_DIR:-$OUT/train}"' in text and 'WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"' in text
    assert '--warmup-epochs "$WARMUP_EPOCHS"' in call and '${INIT_FROM:+--init-from "$INIT_FROM"}' in call


@pytest.mark.integration
def test_bench_stage_uses_best_checkpoint_both_years_and_is_idempotent() -> None:
    call = run_calls()["3_bench"]
    assert '--ckpt "$TRAIN_DIR/best.pt"' in call and '--out "$OUT/bench"' in call
    assert "--iters 0 25 50 100 200" in call
    assert "2016=$SRC/sample_40.pkl" in call and "2017=$SRC/sample_40_2017.pkl" in call
    assert "archive/E5_adaptive_k/results" in script_text() and "--e5" in call
    assert '"$OUT/bench/adoption_table.csv"' in script_text()  # garde : étape déjà faite -> saut


@pytest.mark.integration
def test_preflight_checks_inputs_shards_and_logw_table() -> None:
    text = script_text()
    for token in ("index.csv", "summary.csv", "sample_40.pkl", "sample_40_2017.pkl", "build_samples.py --logw",
                  "diffusion_profiles_logw_$KEY_DEFAULT.pt", "cells_ge5000.csv", "REQUIRE_CUDA"):
        assert token in text, token


@pytest.mark.integration
def test_summary_always_runs_and_dispatch_covers_stages() -> None:
    text = script_text()
    for n in range(5):
        assert f"{n}) stage_{n}" in text
    assert 'case " $STAGES " in *" 4 "*) ;; *) stage_4 ;; esac' in text
    for name in ("train.csv", "val.csv", "refinement_curve.csv", "adoption_table.csv", "throughput.csv"):
        assert name in text, name
    assert "Bilan" in text


@pytest.mark.integration
def test_verify_stage_runs_on_a_tiny_shard_folder(tmp_path: Path) -> None:
    # Arrange : le stage 1 sur un dossier sans index doit échouer proprement (pas de traceback muet)
    env = {"PATH": "/usr/bin:/bin", "REQUIRE_CUDA": "0", "DEV": "cpu", "PY": sys.executable, "STAGES": "1",
           "SHARDS": str(tmp_path)}
    # Act
    done = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=env, capture_output=True, text=True)
    # Assert
    assert done.returncode != 0
    assert "index.csv" in done.stdout
    assert f"SHARDS={tmp_path}" in done.stdout and "attendu" in done.stdout


def run_function_source() -> str:
    m = re.search(r"^run\(\) \{.*?^\}\n", script_text(), flags=re.S | re.M)
    assert m, "fonction run() introuvable"
    return m.group(0)


def call_run(tmp_path: Path, exit_code: int, extra: str = "") -> tuple[int, str]:
    """Exécute run() seule avec une fausse commande qui compte ses appels et sort avec `exit_code`."""
    calls = tmp_path / "calls"
    body = (f'set -uo pipefail; LOGS={tmp_path}; STATUS=(); RETRIES=3; INTERRUPTED=0; {run_function_source()}\n'
            f'{extra}run fake bash -c "echo c >> {calls}; exit {exit_code}"\n'
            f'run second bash -c "echo c >> {calls}; exit 0"\n'
            'printf "%s\\n" "${STATUS[@]}"; echo "INTERRUPTED=$INTERRUPTED"\n')
    done = subprocess.run(["bash", "-c", body], capture_output=True, text=True, check=True)
    n = len(calls.read_text().split()) if calls.exists() else 0
    return n, done.stdout


@pytest.mark.integration
@pytest.mark.parametrize("code", [130, 143])
def test_run_does_not_retry_a_deliberate_interruption_and_stops_the_rest(tmp_path: Path, code: int) -> None:
    # Arrange / Act : SIGINT/SIGTERM voulus ; l'appel suivant ne doit pas s'exécuter non plus
    n, out = call_run(tmp_path, code)
    # Assert
    assert n == 1
    assert "INTERROMPU" in out and "INTERRUPTED=1" in out and "OK     second" not in out


@pytest.mark.integration
def test_run_retries_ordinary_failures_up_to_retries(tmp_path: Path) -> None:
    n, out = call_run(tmp_path, 1)
    assert n == 3 + 1   # 3 tentatives de la commande qui échoue, puis la suivante s'exécute
    assert "ÉCHEC  fake après 3" in out and "INTERRUPTED=0" in out


@pytest.mark.integration
def test_main_loop_stops_remaining_stages_after_interruption_and_bench_is_single_try() -> None:
    text = script_text()
    assert 'if [ "$INTERRUPTED" = 1 ]; then' in text and "break" in text
    assert re.search(r"TRIES=1 run 3_bench ", text)


@pytest.mark.integration
def test_class_comes_from_train_csv_bin_column_not_from_a_replay() -> None:
    text = script_text()
    assert 't.groupby("bin")' in text and ".batches()" in text   # lots par classe seulement (projection)
    assert "batches[s - 1]" not in text and "bins +=" not in text
    assert "--log-every 1" in run_calls()["0_smoke_train"]


@pytest.mark.integration
def test_projection_drops_warmup_requires_rows_and_states_lower_bound() -> None:
    text = script_text()
    assert "WARMUP_STEPS" in text and "MIN_ROWS" in text and "borne inférieure" in text
    assert "validation de fin d'époque" in text


@pytest.mark.integration
def test_cpu_device_warns_memory_and_verdict_are_not_meaningful() -> None:
    assert re.search(r"DEV.*cpu.*non significatifs|cpu.*non significatifs", script_text())


@pytest.mark.integration
def test_preflight_checks_disk_python_imports_and_shard_contents() -> None:
    text = script_text()
    assert "import encoder_train, encoder_eval" in text
    assert "df -Pk" in text and "5 Go" in text
    assert "bit_offsets" in text and "raw.size" in text   # taille des tableaux vs métadonnées, tous les shards


@pytest.mark.integration
def test_no_and_list_pattern_left_for_optional_output() -> None:
    assert not re.search(r"\[ -f [^\]]*\] && cat", script_text())
    assert "&& printf" not in script_text()


def tiny_shards(tmp_path: Path) -> Path:
    import numpy as np
    from datetime import datetime
    import cell_shards as cs
    from cell_shard_dataset import ShardIndex
    folder = tmp_path / "shards"
    folder.mkdir()
    R = np.zeros((40, 40), np.int16)
    R[5:14, 5:14] = 20
    recs = cs.extract_frame_cells(R, datetime(2016, 2, 3), "train")
    cs.write_shard(folder / "train_50-200_0000.shard", recs)
    ShardIndex.from_dir(folder).to_frame().to_csv(folder / "index.csv", index=False)
    (folder / "summary.csv").write_text("split,bin,n_cells,raw_mb\n")
    return folder


def run_stage_1(folder: Path) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "REQUIRE_CUDA": "0", "DEV": "cpu", "PY": sys.executable, "STAGES": "1",
           "SHARDS": str(folder), "HOME": str(folder)}
    return subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.mark.integration
def test_verify_stage_reports_readable_shards_message(tmp_path: Path) -> None:
    done = run_stage_1(tiny_shards(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1/1 shards lisibles" in done.stdout and "OK     1_verify" in done.stdout


@pytest.mark.integration
def test_verify_stage_detects_a_truncated_shard_array(tmp_path: Path) -> None:
    folder = tiny_shards(tmp_path)
    raw = folder / "train_50-200_0000.shard" / "raw.npy"
    raw.write_bytes(raw.read_bytes()[:-8])
    done = run_stage_1(folder)
    assert "ILLISIBLE" in done.stdout or "n'a pas" in done.stdout or "shards incomplets" in done.stdout.lower()
    assert "ÉCHEC  1_verify" in done.stdout or done.returncode != 0
