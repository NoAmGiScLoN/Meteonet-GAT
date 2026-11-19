#!/usr/bin/env bash
# Lance sur le GPU du Jetson tous les tests en attente, en une seule session.
#   1. Sources de taille variable : gauss vs diffusion, 4 classes de taille x 40 cellules (janv./juil. 2016)
#   2. Stabilité temporelle : même cellule à t et t + 5 min (25 paires par classe)
#   3. Ablation : seul tau libre (diff_tau) / seul w libre (diff_w), mêmes 160 cellules
#   4. Autre saison / autre année : avril + septembre 2017 (160 cellules)
#   5. Redémarrages : initialisation perturbée (seed 1), mêmes 160 cellules
#   7. Benchmark à vecteur fixe : ae d=48 (sous-entraîné auparavant)
#   (l'étape 6, gmm d=48, a été retirée avec le bras gmm : effondré à deux taux d'apprentissage)
# Une étape qui échoue n'arrête pas les suivantes ; le bilan s'affiche à la fin.
# Usage (depuis la racine du dépôt) :
#   nohup bash scripts/run_jetson.sh > results/04_jetson_run/logs/run_jetson.log 2>&1 &   (variables : PY=python3  DEV=cuda)
set -uo pipefail
cd "$(dirname "$0")/.."   # racine du dépôt
PY="${PY:-python3}"
DEV="${DEV:-cuda}"
OUT=results/04_jetson_run
LOGS=results/04_jetson_run/logs
SRC=results/03_variable_sources
mkdir -p "$LOGS" "$OUT"
STATUS=()

"$PY" - <<'EOF' || { echo "Vérification de l'environnement échouée : arrêt."; exit 1; }
import torch, numpy, scipy, pandas, matplotlib
print("torch", torch.__version__, "| numpy", numpy.__version__, "| CUDA :", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA indisponible : installer le PyTorch de NVIDIA pour votre JetPack"
print("GPU :", torch.cuda.get_device_name(0))
EOF

# chaque dossier de résultats reçoit la même table de profils de diffusion (pas de recalcul)
prep() { mkdir -p "$1"; cp -n "$SRC/diffusion_profiles.pt" "$1/"; }

run() {  # run <nom> <commande...>
    local name=$1; shift
    echo; echo "=== $name — début $(date +%H:%M) ==="
    if "$@" 2>&1 | tee "$LOGS/$name.log"; then
        STATUS+=("OK     $name"); echo "=== $name — fin $(date +%H:%M) ==="
    else
        STATUS+=("ÉCHEC  $name (voir $LOGS/$name.log)"); echo "=== $name — ÉCHEC $(date +%H:%M) ==="
    fi
}

prep "$OUT/main"; prep "$OUT/stability"; prep "$OUT/ablation"; prep "$OUT/season2017"; prep "$OUT/restart_seed1"
cp -n "$SRC/stability_pairs_25.pkl" "$OUT/stability/"

run 1_principal   "$PY" src/variable_source_fit.py --sample "$SRC/sample_40.pkl" --arms gauss diffusion \
                  --iters 200 --device "$DEV" --out "$OUT/main"
run 2_stabilite   "$PY" src/stability_test.py --per-bin 25 --iters 200 --device "$DEV" --out "$OUT/stability"
run 3_ablation    "$PY" src/variable_source_fit.py --sample "$SRC/sample_40.pkl" --arms diff_tau diff_w \
                  --iters 200 --device "$DEV" --out "$OUT/ablation"
run 4_saison2017  "$PY" src/variable_source_fit.py --sample "$SRC/sample_40_2017.pkl" --arms gauss diffusion \
                  --iters 200 --device "$DEV" --out "$OUT/season2017"
run 5_redemarrage "$PY" src/variable_source_fit.py --sample "$SRC/sample_40.pkl" --arms gauss diffusion --seed 1 \
                  --iters 200 --device "$DEV" --out "$OUT/restart_seed1"
run 7_ae48        "$PY" src/compression_benchmark.py --arms ae --dims 48 --epochs 60 --device "$DEV" --tag results_jetson

echo; echo "================ Bilan ($(date +%H:%M)) ================"
printf '%s\n' "${STATUS[@]}"
echo "Résultats : $OUT/ (main, stability, ablation, season2017, restart_seed1) et results/02_fixed_vector_benchmark/results_jetson.json"
