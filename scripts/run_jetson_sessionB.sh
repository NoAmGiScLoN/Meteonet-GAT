#!/usr/bin/env bash
# Session B : ajustement diff7 à K adaptatif (vagues, compression, élagage), sur un Jetson Orin Nano.
#   0. smoke         (~5 min)       : adaptive_run, 1 cellule par classe, 5 itérations, k-cap 16 ; secondes par vague et par classe
#   1. adaptive 2016 (~2,3-4,3 h)   : adaptive_run, echantillon sample_40, k-cap 128, reprenable
#   3. test du plancher (~1,3-4 h)  : adaptive_run, 10 plus grandes cellules 2016, k-cap 256, reprenable
#   2. adaptive 2017 (~2,3-4,3 h)   : adaptive_run, echantillon sample_40_2017, k-cap 128, reprenable
#   4. résumé        (~1 min)       : summary_{stop,k,error}.csv et pics memoire de chaque run ; plancher : nrmse median a k<=128 vs final
# Total : 6-12,6 h. Une étape qui échoue n'arrête pas les suivantes ; le bilan s'affiche à la fin.
# Reprenable : relancer la même commande (--resume saute les cellules faites) ; p. ex. STAGES="2 4" pour finir 2017.
# Usage (depuis la racine du dépôt, sur le Jetson) :
#   mkdir -p results/07_sessionB/logs && STAGES="0 1 3 2 4" PY=~/gatenv/bin/python nohup bash scripts/run_jetson_sessionB.sh > results/07_sessionB/logs/run.log 2>&1 &
#   STAGES="0" ...  smoke seul ; STAGES="2 4" ... étapes choisies (le résumé s'affiche toujours à la fin).
# Variables : PY (python3)  DEV (cuda)  REQUIRE_CUDA (1 ; 0 = pas d'assertion CUDA, pour un essai local)  STAGES ("0 1 3 2 4")
# Les sorties vont dans results/07_sessionB/ : non ignoré par git, elles apparaissent comme non suivies.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # racine du dépôt
PY="${PY:-python3}"
DEV="${DEV:-cuda}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
STAGES="${STAGES:-0 1 3 2 4}"
OUT=results/07_sessionB
LOGS=$OUT/logs
SRC=results/03_variable_sources
KEY_DEFAULT=ace17c5890   # table log-w par défaut (w 0.02-0.5, 16 noeuds)
mkdir -p "$LOGS" "$OUT"
STATUS=()
T0=$SECONDS

echo "commit : $(git rev-parse --short HEAD 2>/dev/null || echo inconnu) | $(date '+%F %H:%M')"

REQUIRE_CUDA="$REQUIRE_CUDA" "$PY" - <<'PYEOF' || { echo "Vérification de l'environnement échouée : arrêt."; exit 1; }
import os, sys, torch, numpy, scipy, pandas, matplotlib
cuda = torch.cuda.is_available()
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA :", cuda)
if os.environ.get("REQUIRE_CUDA", "1") != "0":
    assert cuda, "CUDA indisponible : installer le PyTorch de NVIDIA pour votre JetPack"
print("GPU :", torch.cuda.get_device_name(0) if cuda else "aucun (REQUIRE_CUDA=0)")
PYEOF

check_inputs() {
    local f missing=0
    for f in "$SRC/sample_40.pkl" "$SRC/sample_40_2017.pkl" "$SRC/diffusion_profiles.pt"; do
        [ -f "$f" ] || { echo "Entrée manquante : $f"; missing=1; }
    done
    return $missing
}

build_logw_tables() {
    "$PY" scripts/build_samples.py --logw || return 1
}

preflight() {
    check_inputs || { echo "Entrées manquantes (voir scripts/build_samples.py) : arrêt."; return 1; }
    if [ ! -f "$SRC/diffusion_profiles_logw_$KEY_DEFAULT.pt" ]; then
        build_logw_tables || { echo "Construction de la table log-w échouée : arrêt."; return 1; }
    fi
    [ -f "$SRC/diffusion_profiles_logw_$KEY_DEFAULT.pt" ] || { echo "Table log-w $KEY_DEFAULT absente de $SRC : arrêt."; return 1; }
}
preflight || exit 1

# chaque dossier de résultats reçoit les mêmes tables (pas de recalcul, clés identiques)
prep() {  # prep <dossier>
    mkdir -p "$1"
    cp -n "$SRC/diffusion_profiles.pt" "$SRC/diffusion_profiles_logw_$KEY_DEFAULT.pt" "$1/"
}

RETRIES="${RETRIES:-3}"   # tentatives par étape : --resume rend une reprise peu coûteuse (OOM noyau, kill)

run() {  # run <nom> <commande...>
    local name=$1; shift
    local attempt
    echo; echo "=== $name — début $(date +%H:%M) ==="
    : > "$LOGS/$name.log"
    for attempt in $(seq 1 "$RETRIES"); do
        if "$@" 2>&1 | tee -a "$LOGS/$name.log"; then
            STATUS+=("OK     $name (tentative $attempt)"); echo "=== $name — fin $(date +%H:%M) ==="
            return 0
        fi
        echo "=== $name — tentative $attempt échouée $(date +%H:%M) ===" | tee -a "$LOGS/$name.log"
    done
    STATUS+=("ÉCHEC  $name après $RETRIES tentatives (voir $LOGS/$name.log)"); echo "=== $name — ÉCHEC $(date +%H:%M) ==="
}

stage_0() {
    prep "$OUT/smoke"
    run 0_smoke "$PY" src/adaptive_run.py --sample "$SRC/sample_40.pkl" --max-per-bin 1 --iters 5 --k-cap 16 \
                --overwrite --device "$DEV" --out "$OUT/smoke"
    echo "--- secondes par vague et par cellule, par classe (smoke)"
    SMOKE_DIR="$OUT/smoke" "$PY" - <<'PYEOF' || echo "Résumé du smoke échoué."
import glob, os
import pandas as pd
files = sorted(glob.glob(os.path.join(os.environ["SMOKE_DIR"], "waves_*.csv")))
if not files:
    print("absent")
for f in files:
    w = pd.read_csv(f)
    per_wave = w.groupby("wave").seconds.sum()   # secondes par cellule et vague, sommées sur les cellules vivantes
    print(f"{os.path.basename(f)} : {per_wave.mean():.2f} s/vague (moyenne sur {len(per_wave)} vagues), {per_wave.sum():.1f} s au total")
print("Projection grossière : durée d'une classe ~ nb cellules x vagues x s/vague (5 it. ici ; x iterations reelles/5 pour le run complet).")
PYEOF
}

stage_1() {
    prep "$OUT/adaptive_2016"
    run 1_adaptive_2016 "$PY" src/adaptive_run.py --sample "$SRC/sample_40.pkl" --k-cap 128 --resume \
                        --device "$DEV" --out "$OUT/adaptive_2016"
}

stage_3() {
    prep "$OUT/floor256"
    run 3_floor256 "$PY" src/adaptive_run.py --sample "$SRC/sample_40.pkl" --largest 10 --k-cap 256 --resume \
                   --device "$DEV" --out "$OUT/floor256"
}

stage_2() {
    prep "$OUT/adaptive_2017"
    run 2_adaptive_2017 "$PY" src/adaptive_run.py --sample "$SRC/sample_40_2017.pkl" --k-cap 128 --resume \
                        --device "$DEV" --out "$OUT/adaptive_2017"
}

show() {  # show <fichier> : affiche un fichier s'il existe
    if [ -f "$1" ]; then echo "--- $1"; cat "$1"; else echo "--- $1 : absent"; fi
}

stage_4() {
    local d
    for d in adaptive_2016 floor256 adaptive_2017; do
        show "$OUT/$d/summary_stop.csv"
        show "$OUT/$d/summary_k.csv"
        show "$OUT/$d/summary_error.csv"
    done
    RUNS_ROOT="$OUT" "$PY" - <<'PYEOF' || echo "Résumé mémoire/plancher échoué."
import os
import pandas as pd
root = os.environ["RUNS_ROOT"]
K_REF = 128
for run in ("adaptive_2016", "floor256", "adaptive_2017"):
    mem = os.path.join(root, run, "memory.csv")
    print(f"--- {mem}")
    if not os.path.isfile(mem):
        print("absent")
        continue
    m = pd.read_csv(mem)
    print(f"pic memoire max : {m.peak_bytes.max() / 2**20:.0f} Mo | temps cumule des groupes : {m.seconds.sum() / 60:.1f} min")
    print(m.groupby("bin").agg(peak_Mo=("peak_bytes", lambda s: s.max() / 2**20), seconds=("seconds", "sum")).round(1).to_string())
print(f"--- floor256 : nrmse median au dernier palier k_target <= {K_REF} vs final")
import glob
files = sorted(glob.glob(os.path.join(root, "floor256", "waves_*.csv")))
if not files:
    print("absent")
else:
    import numpy as np
    w = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values(["id", "wave"])
    w = w[np.isfinite(w.nrmse)]   # cellules divergées exclues, comme dans les résumés
    final = w.groupby("id").nrmse.last()
    at_ref = w[w.k_target <= K_REF].groupby("id").nrmse.last()
    print(f"cellules : {len(final)} (attendu 10 ; moins = groupe interrompu, voir cells_ge5000.csv) | "
          f"nrmse median a k<={K_REF} : {at_ref.median():.4f} | final : {final.median():.4f}")
    print(f"temps total floor256 : {w.seconds.sum() / 60:.1f} min (somme par cellule et vague)")
PYEOF
}

for s in $STAGES; do
    case "$s" in
        0) stage_0 ;; 1) stage_1 ;; 2) stage_2 ;; 3) stage_3 ;; 4) stage_4 ;;
        *) echo "Étape inconnue : $s" ;;
    esac
done
case " $STAGES " in *" 4 "*) ;; *) stage_4 ;; esac   # le résumé s'affiche toujours

echo; echo "================ Bilan ($(date +%H:%M)) ================"
[ "${#STATUS[@]}" -gt 0 ] && printf '%s\n' "${STATUS[@]}"
echo "Temps total : $(( (SECONDS - T0) / 60 )) min"
echo "Résultats : $OUT/"
