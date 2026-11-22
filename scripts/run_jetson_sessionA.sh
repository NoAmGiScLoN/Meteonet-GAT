#!/usr/bin/env bash
# Session A : choix de l'optimiseur (lr, planning cosinus) du bras diff7, puis référence de stabilité diff7,
# sur un Jetson Orin Nano.
#   0. smoke         (~3 min)       : optimizer_check (1 cellule par classe, 5 itérations) ; stability_test (2 itérations)
#   1. optimiseur    (~2 h)         : optimizer_check, 10 cellules par classe, fixed:0.05 / cosine:0.05 à 200 et 500 it., cosine:0.1 à 200 it.
#   2. stabilité diff7 (~55-60 min) : stability_test, bras diff7, itérations par classe, 25 paires par classe
#   3. résumé        (~1 min)       : optimizer_choice.csv, optimizer_pileup.csv, ratio cost_same/cost_other par classe, temps total
# Total : ~3 h. Une étape qui échoue n'arrête pas les suivantes ; le bilan s'affiche à la fin.
# Usage (depuis la racine du dépôt, sur le Jetson) :
#   mkdir -p results/06_sessionA/logs && STAGES="0 1 2" PY=~/gatenv/bin/python nohup bash scripts/run_jetson_sessionA.sh > results/06_sessionA/logs/run.log 2>&1 &
#   STAGES="0" ...  smoke seul ; STAGES="1 2" ... étapes choisies (le résumé s'affiche toujours à la fin).
# Variables : PY (python3)  DEV (cuda)  REQUIRE_CUDA (1 ; 0 = pas d'assertion CUDA, pour un essai local)  STAGES ("0 1 2")
# Les sorties vont dans results/06_sessionA/ : non ignoré par git, elles apparaissent comme non suivies.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # racine du dépôt
PY="${PY:-python3}"
DEV="${DEV:-cuda}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
STAGES="${STAGES:-0 1 2}"
OUT=results/06_sessionA
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
    for f in "$SRC/sample_40.pkl" "$SRC/stability_pairs_25.pkl" "$SRC/diffusion_profiles.pt"; do
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

run() {  # run <nom> <commande...>
    local name=$1; shift
    echo; echo "=== $name — début $(date +%H:%M) ==="
    if "$@" 2>&1 | tee "$LOGS/$name.log"; then
        STATUS+=("OK     $name"); echo "=== $name — fin $(date +%H:%M) ==="
    else
        STATUS+=("ÉCHEC  $name (voir $LOGS/$name.log)"); echo "=== $name — ÉCHEC $(date +%H:%M) ==="
    fi
}

stage_0() {
    prep "$OUT/smoke_opt"; prep "$OUT/smoke_stab"
    run 0_smoke_opt  "$PY" src/optimizer_check.py --sample "$SRC/sample_40.pkl" --per-bin 1 \
                     --plan fixed:0.05@5 cosine:0.05@5 --baseline-iters 5 --device "$DEV" --out "$OUT/smoke_opt"
    run 0_smoke_stab "$PY" src/stability_test.py --arms diff7 --iters 2 --pairs "$SRC/stability_pairs_25.pkl" \
                     --device "$DEV" --out "$OUT/smoke_stab"
}

stage_1() {
    prep "$OUT/optimizer"
    run 1_optimizer "$PY" src/optimizer_check.py --sample "$SRC/sample_40.pkl" --per-bin 10 --seed 0 \
                    --plan fixed:0.05@200,500 cosine:0.05@200,500 cosine:0.1@200 --device "$DEV" --out "$OUT/optimizer"
}

stage_2() {
    prep "$OUT/stability"
    run 2_stability "$PY" src/stability_test.py --arms diff7 --iters-by-bin --pairs "$SRC/stability_pairs_25.pkl" \
                    --device "$DEV" --out "$OUT/stability"
}

show() {  # show <fichier> : affiche un fichier s'il existe
    if [ -f "$1" ]; then echo "--- $1"; cat "$1"; else echo "--- $1 : absent"; fi
}

stage_3() {
    show "$OUT/optimizer/optimizer_choice.csv"
    show "$OUT/optimizer/optimizer_pileup.csv"
    echo "--- $OUT/stability/stability.csv : médiane cost_same/cost_other et part same<other par classe"
    STAB_CSV="$OUT/stability/stability.csv" "$PY" - <<'PYEOF' || echo "Résumé de stabilité échoué."
import os
import pandas as pd
path = os.environ["STAB_CSV"]
if not os.path.isfile(path):
    print("absent")
else:
    df = pd.read_csv(path)
    df["ratio"] = df.cost_same / df.cost_other
    df["same_lt_other"] = df.cost_same < df.cost_other
    # ratio_median_of_pairs = définition des rapports archivés (E2b) ; ratio_of_medians = ce qu'affiche stability_test
    summ = df.groupby(["bin", "arm"]).agg(n=("pair", "count"), ratio_median_of_pairs=("ratio", "median"),
                                          med_same=("cost_same", "median"), med_other=("cost_other", "median"),
                                          share_same_lt_other=("same_lt_other", "mean"))
    summ["ratio_of_medians"] = summ.med_same / summ.med_other
    print(summ.drop(columns=["med_same", "med_other"]).round(3).to_string())
PYEOF
}

for s in $STAGES; do
    case "$s" in
        0) stage_0 ;; 1) stage_1 ;; 2) stage_2 ;; 3) stage_3 ;;
        *) echo "Étape inconnue : $s" ;;
    esac
done
case " $STAGES " in *" 3 "*) ;; *) stage_3 ;; esac   # le résumé s'affiche toujours

echo; echo "================ Bilan ($(date +%H:%M)) ================"
[ "${#STATUS[@]}" -gt 0 ] && printf '%s\n' "${STATUS[@]}"
echo "Temps total : $(( (SECONDS - T0) / 60 )) min"
echo "Résultats : $OUT/"
