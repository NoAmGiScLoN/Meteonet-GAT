#!/usr/bin/env bash
# Phases 1-2 du bras diff7 (7 paramètres, tau = 0.5 fixe, table de profils à w log-espacé) sur un Jetson Orin Nano :
# diff7 vs bras diffusion à 8 paramètres, puis contrôle de convergence du nombre d'itérations.
#   0. smoke         (~3 min)      : 1 cellule par classe, 5 itérations ; convergence_check sur 1 cellule par classe
#   1. principal 2016 (~93 min)    : diffusion + diff7, 160 cellules, 200 it. ; comparaison + reproduction de l'archive
#   2. diff7, table w large (~48 min) : w de 0.005 à 0.5 (24 noeuds) ; comparé au bras diffusion de l'étape 1
#   3. convergence   (~2.5-2.8 h)  : diff7, 10 cellules par classe, 200 / 500 / 2000 itérations
#   4. diff7 en 2017 (~48 min)     : comparé au bras diffusion archivé (sans params)
#   5. bilan         (~1 min)      : récapitulatif compact des tables produites et temps total
# Total : ~5.8 h. Une étape qui échoue n'arrête pas les suivantes ; le bilan s'affiche à la fin.
# Usage (depuis la racine du dépôt, sur le Jetson) :
#   mkdir -p results/05_diff7_convergence/logs
#   PY=~/gatenv/bin/python nohup bash scripts/run_jetson_diff7.sh > results/05_diff7_convergence/logs/run.log 2>&1 &
#   STAGES="0" ...  smoke seul ; STAGES="1 2" ... étapes choisies.
# Variables : PY (python3)  DEV (cuda)  REQUIRE_CUDA (1 ; 0 = pas d'assertion CUDA, pour un essai local)  STAGES ("0 1 2 3 4 5")
# Les sorties vont dans results/05_diff7_convergence/ : non ignoré par git (seuls les fichiers d'entrée volumineux le sont),
# elles apparaissent donc comme non suivies et seront archivées ensuite.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # racine du dépôt
PY="${PY:-python3}"
DEV="${DEV:-cuda}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
STAGES="${STAGES:-0 1 2 3 4 5}"
OUT=results/05_diff7_convergence
LOGS=$OUT/logs
SRC=results/03_variable_sources
ARCH=archive/E2_variable_sources/02_jetson_baseline/results
KEY_DEFAULT=ace17c5890   # table log-w par défaut (w 0.02-0.5, 16 noeuds)
KEY_WIDE=1e980880f3      # table log-w large (w 0.005-0.5, 24 noeuds)
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
    "$PY" scripts/build_samples.py --logw --w-min 0.005 --w-max 0.5 --n-w 24 || return 1
}

preflight() {
    check_inputs || { echo "Entrées manquantes (voir scripts/build_samples.py) : arrêt."; return 1; }
    local key
    if [ ! -f "$SRC/diffusion_profiles_logw_$KEY_DEFAULT.pt" ] || [ ! -f "$SRC/diffusion_profiles_logw_$KEY_WIDE.pt" ]; then
        build_logw_tables || { echo "Construction des tables log-w échouée : arrêt."; return 1; }
    fi
    for key in "$KEY_DEFAULT" "$KEY_WIDE"; do
        [ -f "$SRC/diffusion_profiles_logw_$key.pt" ] || { echo "Table log-w $key absente de $SRC : arrêt."; return 1; }
    done
}
preflight || exit 1

# chaque dossier de résultats reçoit les mêmes tables (pas de recalcul, clés identiques)
prep() {  # prep <dossier>
    mkdir -p "$1"
    cp -n "$SRC/diffusion_profiles.pt" "$SRC/diffusion_profiles_logw_$KEY_DEFAULT.pt" \
          "$SRC/diffusion_profiles_logw_$KEY_WIDE.pt" "$1/"
}

run() {  # run <nom> <commande...>
    local name=$1; shift
    echo; echo "=== $name — début $(date +%H:%M) ==="
    if "$@" 2>&1 | tee "$LOGS/$name.log"; then
        STATUS+=("OK     $name"); echo "=== $name — fin $(date +%H:%M) ==="
    else
        STATUS+=("ÉCHEC  $name (voir $LOGS/$name.log)"); echo "=== $name — ÉCHEC $(date +%H:%M) ==="
    fi
}

vsf() { "$PY" src/variable_source_fit.py --device "$DEV" "$@"; }

stage_0() {
    prep "$OUT/smoke"; prep "$OUT/smoke_conv"
    run 0_smoke_fit  vsf --sample "$SRC/sample_40.pkl" --arms diffusion diff7 --max-per-bin 1 --iters 5 --save-params --out "$OUT/smoke"
    run 0_smoke_conv "$PY" src/convergence_check.py --sample "$SRC/sample_40.pkl" --arm diff7 --per-bin 1 \
                     --iters-list 2 4 --device "$DEV" --out "$OUT/smoke_conv"
}

stage_1() {
    prep "$OUT/main"
    run 1_main    vsf --sample "$SRC/sample_40.pkl" --arms diffusion diff7 --iters 200 --save-params --out "$OUT/main"
    run 1_compare "$PY" src/arm_comparison.py --runs "$OUT/main" --logw-key "$KEY_DEFAULT" \
                  --archived "$ARCH/main" --out "$OUT/compare_main"
}

stage_2() {
    prep "$OUT/main_wext"
    run 2_main_wext vsf --sample "$SRC/sample_40.pkl" --arms diff7 --iters 200 --save-params \
                    --w-min 0.005 --w-max 0.5 --n-w 24 --out "$OUT/main_wext"
    run 2_compare   "$PY" src/arm_comparison.py --runs "$OUT/main_wext" --reference "$OUT/main" \
                    --logw-key "$KEY_WIDE" --out "$OUT/compare_wext"
}

stage_3() {
    prep "$OUT/convergence"
    run 3_convergence "$PY" src/convergence_check.py --sample "$SRC/sample_40.pkl" --arm diff7 --per-bin 10 --seed 0 \
                      --iters-list 200 500 2000 --device "$DEV" --out "$OUT/convergence"
}

stage_4() {
    prep "$OUT/season2017_diff7"
    run 4_season2017 vsf --sample "$SRC/sample_40_2017.pkl" --arms diff7 --iters 200 --save-params \
                     --out "$OUT/season2017_diff7"
    run 4_compare    "$PY" src/arm_comparison.py --runs "$OUT/season2017_diff7" --reference "$ARCH/season2017" \
                     --logw-key "$KEY_DEFAULT" --out "$OUT/compare_2017"
}

show() {  # show <fichier> : affiche un fichier s'il existe
    if [ -f "$1" ]; then echo "--- $1"; cat "$1"; else echo "--- $1 : absent"; fi
}

stage_5() {
    local f
    for f in "$OUT"/compare_*/*_equivalence.csv "$OUT"/compare_*/*_w_pileup.csv \
             "$OUT/compare_main/reproduction.csv" "$OUT/convergence/convergence_choice.csv"; do
        show "$f"
    done
}

for s in $STAGES; do
    case "$s" in
        0) stage_0 ;; 1) stage_1 ;; 2) stage_2 ;; 3) stage_3 ;; 4) stage_4 ;; 5) stage_5 ;;
        *) echo "Étape inconnue : $s" ;;
    esac
done

echo; echo "================ Bilan ($(date +%H:%M)) ================"
[ "${#STATUS[@]}" -gt 0 ] && printf '%s\n' "${STATUS[@]}"
echo "Temps total : $(( (SECONDS - T0) / 60 )) min"
echo "Résultats : $OUT/"
