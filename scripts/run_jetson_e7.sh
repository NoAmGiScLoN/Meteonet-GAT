#!/usr/bin/env bash
# E7 : encodeur de sources (entraînement, benchmark d'inférence + raffinement) sur un Jetson Orin Nano.
#   0. smoke      (~10 min)    : 200 pas d'entraînement (dossier vidé), ms/pas, data_ms, render_ms et pic CUDA par classe,
#                                inférence sur 1 cellule par classe (--iters 0), projection du temps d'une époque par classe
#                                (décision : entraîner sur le Jetson si une époque < 4 h)
#   1. vérification des shards  : intégrité rapide (un enregistrement lu en mmap par shard)
#   2. entraînement (EPOCHS=20) : encoder_train, reprenable (--resume), dans $OUT/train
#   3. benchmark                : encoder_eval depuis $OUT/train/best.pt, --iters 0 25 50 100 200, 2016 et 2017, vs E5
#   4. résumé     (~1 min)      : fins de train.csv / val.csv par classe, refinement_curve, adoption_table, throughput
# Une étape qui échoue n'arrête pas les suivantes ; le bilan s'affiche à la fin. Un SIGINT/SIGTERM voulu (code 130/143)
# n'est pas relancé : il est noté INTERROMPU et arrête les étapes restantes (le résumé et le bilan s'affichent).
# Reprenable : relancer la même commande (--resume reprend au dernier point de reprise) ; p. ex. STAGES="2 3 4".
# Les shards sont extraits sur le Mac, dans le dépôt principal (scripts/extract_cells.py), puis copiés ; la commande
# est lancée depuis le Mac (<user>, <jetson-host> et <repo> : utilisateur, hôte et dépôt du Jetson, à remplacer) :
#   rsync -a --partial /Users/noam/Documents/RESEARCH/GAT/results/08_e7_cells/ <user>@<jetson-host>:<repo>/results/08_e7_cells/
# Usage (depuis la racine du dépôt, sur le Jetson) :
#   mkdir -p results/10_e7/logs && STAGES="0 1 2 3 4" PY=~/gatenv/bin/python nohup bash scripts/run_jetson_e7.sh > results/10_e7/logs/run.log 2>&1 &
#   STAGES="0" ...  smoke seul ; STAGES="3 4" ... étapes choisies (le résumé s'affiche toujours à la fin).
# Variables : PY (python3)  DEV (cuda)  REQUIRE_CUDA (1 ; 0 = pas d'assertion CUDA)  STAGES ("0 1 2 3 4")
#             EPOCHS (20)  SHARDS (results/08_e7_cells)  RETRIES (3)  TRAIN_DIR ($OUT/train)
#             WARMUP_EPOCHS (2)  INIT_FROM (vide ; p. ex. results/10_e7/train/best.pt avec TRAIN_DIR=results/10_e7/train_v2)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # racine du dépôt
PY="${PY:-python3}"
DEV="${DEV:-cuda}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
STAGES="${STAGES:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-20}"
SHARDS="${SHARDS:-results/08_e7_cells}"
OUT=results/10_e7
TRAIN_DIR="${TRAIN_DIR:-$OUT/train}"   # dossier d'entraînement (nouveau dossier pour une reprise à chaud)
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
INIT_FROM="${INIT_FROM:-}"             # point de reprise dont on charge les poids seuls (optimiseur neuf)
LOGS=$OUT/logs
SRC=results/03_variable_sources
E5=archive/E5_adaptive_k/results
KEY_DEFAULT=ace17c5890   # table log-w par défaut (w 0.02-0.5, 16 noeuds)
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export SHARDS
mkdir -p "$LOGS" "$OUT"
STATUS=()
INTERRUPTED=0
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
    for f in "$SHARDS/index.csv" "$SHARDS/summary.csv" "$SRC/sample_40.pkl" "$SRC/sample_40_2017.pkl" \
             "$SRC/diffusion_profiles.pt" "$E5/adaptive_2016/cells_ge5000.csv" "$E5/adaptive_2017/cells_ge5000.csv" \
             "$E5/adaptive_2016_c050/cells_50_200.csv" "$E5/adaptive_2017_c050/cells_50_200.csv"; do
        [ -f "$f" ] || { echo "Entrée manquante : $f"; missing=1; }
    done
    if [ ! -f "$SHARDS/index.csv" ]; then
        echo "SHARDS=$SHARDS : dossier de shards attendu avec index.csv et summary.csv (extraits sur le Mac, voir rsync en en-tête)."
    fi
    return $missing
}

check_shard_count() {  # les shards sur disque doivent être exactement ceux de index.csv
    "$PY" - <<'PYEOF'
import os, sys
import pandas as pd
d = os.environ["SHARDS"]
named = set(pd.read_csv(os.path.join(d, "index.csv"), usecols=["shard"]).shard)
on_disk = {n for n in os.listdir(d) if n.endswith(".shard")}
print(f"shards : {len(on_disk)} sur disque, {len(named)} dans index.csv")
if named != on_disk:
    print("manquants :", sorted(named - on_disk)[:5], "| en trop :", sorted(on_disk - named)[:5])
    sys.exit(1)
PYEOF
}

check_disk_and_imports() {
    local free_kb
    free_kb=$(df -Pk . | awk 'NR==2 {print $4}')
    if [ -n "$free_kb" ] && [ "$free_kb" -lt 5242880 ]; then
        echo "ATTENTION : moins de 5 Go libres ($((free_kb / 1048576)) Go) : checkpoints et logs risquent de saturer le disque."
    fi
    "$PY" -c "import encoder_train, encoder_eval" || { echo "Import de src/encoder_train / encoder_eval impossible (PYTHONPATH=$PYTHONPATH)."; return 1; }
}

build_logw_tables() {
    "$PY" scripts/build_samples.py --logw || return 1
}

preflight() {
    check_inputs || { echo "Entrées manquantes (shards : rsync depuis le Mac, voir l'en-tête) : arrêt."; return 1; }
    check_disk_and_imports || return 1
    check_shard_count || { echo "Shards incomplets (copie interrompue ?) : arrêt."; return 1; }
    if [ ! -f "$SRC/diffusion_profiles_logw_$KEY_DEFAULT.pt" ]; then
        build_logw_tables || { echo "Construction de la table log-w échouée : arrêt."; return 1; }
    fi
    [ -f "$SRC/diffusion_profiles_logw_$KEY_DEFAULT.pt" ] || { echo "Table log-w $KEY_DEFAULT absente de $SRC : arrêt."; return 1; }
}
preflight || exit 1

RETRIES="${RETRIES:-3}"   # tentatives par étape : --resume rend une reprise peu coûteuse (OOM noyau, kill)

run() {  # run <nom> <commande...> ; TRIES=1 pour une étape non reprenable ; 130/143 = arrêt voulu, jamais relancé
    local name=$1; shift
    local attempt rc tries="${TRIES:-$RETRIES}"
    if [ "$INTERRUPTED" = 1 ]; then return 0; fi
    echo; echo "=== $name — début $(date +%H:%M) ==="
    : > "$LOGS/$name.log"
    for attempt in $(seq 1 "$tries"); do
        "$@" 2>&1 | tee -a "$LOGS/$name.log"
        rc=${PIPESTATUS[0]}
        if [ "$rc" -eq 0 ]; then
            STATUS+=("OK     $name (tentative $attempt)"); echo "=== $name — fin $(date +%H:%M) ==="
            return 0
        fi
        if [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then
            INTERRUPTED=1
            STATUS+=("INTERROMPU $name (code $rc, tentative $attempt) : relancer la même commande pour reprendre")
            echo "=== $name — INTERROMPU (code $rc) $(date +%H:%M) ==="
            return 0
        fi
        echo "=== $name — tentative $attempt échouée $(date +%H:%M) ===" | tee -a "$LOGS/$name.log"
    done
    STATUS+=("ÉCHEC  $name après $tries tentatives (voir $LOGS/$name.log)"); echo "=== $name — ÉCHEC $(date +%H:%M) ==="
}

stage_0() {
    # encoder_train n'a pas --overwrite : le dossier du smoke est vidé, puis l'entraînement repart de zéro (sans --resume)
    rm -rf "$OUT/smoke"; mkdir -p "$OUT/smoke"
    TRIES=1 run 0_smoke_train "$PY" src/encoder_train.py --shards "$SHARDS" --out "$OUT/smoke" --device "$DEV" \
                --epochs 1 --max-steps 200 --log-every 1 --val-every-steps 100000 --val-max-per-class 20
    echo "--- ms/pas, data_ms, render_ms, pic CUDA par classe ; projection d'une époque (smoke)"
    SMOKE_DIR="$OUT/smoke" DEV="$DEV" "$PY" - <<'PYEOF' || echo "Résumé du smoke échoué."
import os
import pandas as pd
from encoder_train import TrainConfig
from size_batching import PixelBudgetSampler, load_index
HOURS_LIMIT = 4.0
MS_PER_HOUR = 3.6e6
WARMUP_STEPS = 5   # premiers pas de chaque classe écartés (démarrage, autotune)
MIN_ROWS = 5       # lignes retenues minimales par classe
index = load_index(os.environ["SHARDS"])
batches = PixelBudgetSampler(index, "train", TrainConfig.px_budget, 0, 0).batches()
counts = pd.Series([index.entries[b[0]].bin for b in batches]).value_counts()   # lots par classe et par époque
path = os.path.join(os.environ["SMOKE_DIR"], "train.csv")
if not os.path.isfile(path):
    raise SystemExit("train.csv absent")
t = pd.read_csv(path).sort_values("step")
t = t[t.groupby("bin").cumcount() >= WARMUP_STEPS]   # sans apply : pandas récent retire la colonne de groupe
g = t.groupby("bin").agg(lignes=("step", "size"), ms_step=("ms_step", "median"), data_ms=("data_ms", "median"),
                         render_ms=("render_ms", "median"), peak_mem_mb=("peak_mem_mb", "max")).round(1)
g = g[g.lignes >= MIN_ROWS]
g["lots_par_epoque"] = counts.reindex(g.index)
g["h_epoque"] = (g.ms_step * g.lots_par_epoque / MS_PER_HOUR).round(2)
print(g.to_string())
missing = sorted(set(counts.index) - set(g.index))
if missing:
    print(f"classes sans mesure suffisante (< {MIN_ROWS} lignes après {WARMUP_STEPS} pas de chauffe) : {missing}")
total = float(g.h_epoque.sum())
print(f"Époque projetée : {total:.2f} h ({len(batches)} lots) : borne inférieure (hors validation de fin d'époque, "
      "points de reprise et surcoût des cartes de pics de l'échauffement).")
if os.environ["DEV"] == "cpu":
    print("DEV=cpu : pic mémoire (0 hors CUDA) et verdict de budget non significatifs ; mesurer sur le Jetson.")
else:
    ok = total < HOURS_LIMIT and not missing
    print("Verdict : " + ("ENTRAÎNER sur le Jetson (< 4 h)" if ok else "entraîner ailleurs (>= 4 h ou projection incomplète)"))
PYEOF
    echo "--- inférence : 1 cellule par classe, --iters 0"
    SRC="$SRC" SMOKE_DIR="$OUT/smoke" "$PY" - <<'PYEOF' || echo "Échantillon du smoke échoué."
import os, pickle
cells = pickle.load(open(os.path.join(os.environ["SRC"], "sample_40.pkl"), "rb"))
first = {}
for c in cells:
    first.setdefault(c["bin"], c)
with open(os.path.join(os.environ["SMOKE_DIR"], "sample_1_per_class.pkl"), "wb") as fh:
    pickle.dump(list(first.values()), fh)
print(len(first), "cellules")
PYEOF
    if [ -f "$OUT/smoke/best.pt" ]; then
        TRIES=1 run 0_smoke_infer "$PY" src/encoder_eval.py --ckpt "$OUT/smoke/best.pt" \
                    --samples "2016=$OUT/smoke/sample_1_per_class.pkl" --iters 0 --device "$DEV" --out "$OUT/smoke/infer"
        if [ -f "$OUT/smoke/infer/throughput.csv" ]; then cat "$OUT/smoke/infer/throughput.csv"; fi
    else
        echo "best.pt absent : inférence non mesurée"
        STATUS+=("ÉCHEC  0_smoke_infer (pas de best.pt)")
    fi
}

stage_1() {
    echo "--- vérification des shards : tableaux entiers (tailles vs métadonnées) et premier/dernier enregistrement"
    TRIES=1 run 1_verify "$PY" - <<'PYEOF'
import os, sys
import pandas as pd
from pathlib import Path
from cell_shards import open_shard
d = Path(os.environ["SHARDS"])
index = pd.read_csv(d / "index.csv", usecols=["shard"])
rows = index.shard.value_counts().to_dict()
bad = 0
for n in sorted(rows):
    try:
        h = open_shard(d / n)
        m = h.meta
        assert len(h) == rows[n], f"{len(h)} enregistrements, index.csv en liste {rows[n]}"
        assert h.raw.size == int(m["offsets"][-1]), f"raw.size {h.raw.size} != offsets[-1] {int(m['offsets'][-1])}"
        assert h.mask_bits.size == int(m["bit_offsets"][-1]), "mask_bits tronqué"
        assert h.invalid_bits.size == int(m["bit_offsets"][-1]), "invalid_bits tronqué"
        h.record(0)
        h.record(len(h) - 1)
    except Exception as exc:   # shard tronqué ou illisible
        bad += 1
        print("ILLISIBLE", n, repr(exc))
print(f"{len(rows) - bad}/{len(rows)} shards lisibles")
sys.exit(1 if bad else 0)
PYEOF
}

stage_2() {
    run 2_train "$PY" src/encoder_train.py --shards "$SHARDS" --out "$TRAIN_DIR" --device "$DEV" --epochs "$EPOCHS" \
        --warmup-epochs "$WARMUP_EPOCHS" ${INIT_FROM:+--init-from "$INIT_FROM"} --resume
}

stage_3() {
    if [ -f "$OUT/bench/adoption_table.csv" ]; then
        echo "Benchmark déjà fait ($OUT/bench/adoption_table.csv) : saut ; supprimer $OUT/bench pour le refaire."
        STATUS+=("OK     3_bench (déjà fait)")
        return 0
    fi
    [ -f "$TRAIN_DIR/best.pt" ] || { echo "best.pt absent : lancer l'étape 2 d'abord."; STATUS+=("ÉCHEC  3_bench (pas de best.pt)"); return 0; }
    TRIES=1 run 3_bench "$PY" src/encoder_eval.py --ckpt "$TRAIN_DIR/best.pt" --iters 0 25 50 100 200 --device "$DEV" \
                --samples "2016=$SRC/sample_40.pkl" "2017=$SRC/sample_40_2017.pkl" \
                --e5 "2016=$E5/adaptive_2016_c050,$E5/adaptive_2016" "2017=$E5/adaptive_2017_c050,$E5/adaptive_2017" \
                --out "$OUT/bench"
}

show() {  # show <fichier> : affiche un fichier s'il existe
    if [ -f "$1" ]; then echo "--- $1"; cat "$1"; else echo "--- $1 : absent"; fi
}

stage_4() {
    OUT="$OUT" "$PY" - <<'PYEOF' || echo "Résumé de l'entraînement échoué."
import os
import pandas as pd
TAIL = 3
folder = os.path.join(os.environ["OUT"], "train")
train, val = os.path.join(folder, "train.csv"), os.path.join(folder, "val.csv")
if os.path.isfile(train):
    t = pd.read_csv(train)
    print(f"--- {train} : {len(t)} lignes, dernier pas {t.step.max()}, derniere epoque {t.epoch.max()}")
    cols = ["step", "epoch", "loss", "recon", "k_active", "ms_step", "peak_mem_mb"]
    for name, part in t.groupby("bin"):
        print(f"[{name}]")
        print(part.tail(TAIL)[cols].to_string(index=False))
else:
    print(f"--- {train} : absent")
if os.path.isfile(val):
    v = pd.read_csv(val)
    print(f"--- {val} : dernières validations par classe")
    print(v.groupby("bin").tail(2).sort_values(["bin", "step"]).to_string(index=False))
else:
    print(f"--- {val} : absent")
PYEOF
    show "$OUT/bench/refinement_curve.csv"
    show "$OUT/bench/adoption_table.csv"
    show "$OUT/bench/throughput.csv"
}

for s in $STAGES; do
    if [ "$INTERRUPTED" = 1 ]; then
        echo "Interruption demandée : étapes restantes ignorées."
        break
    fi
    case "$s" in
        0) stage_0 ;; 1) stage_1 ;; 2) stage_2 ;; 3) stage_3 ;; 4) stage_4 ;;
        *) echo "Étape inconnue : $s" ;;
    esac
done
case " $STAGES " in *" 4 "*) ;; *) stage_4 ;; esac   # le résumé s'affiche toujours

echo; echo "================ Bilan ($(date +%H:%M)) ================"
if [ "${#STATUS[@]}" -gt 0 ]; then printf '%s\n' "${STATUS[@]}"; fi
echo "Temps total : $(( (SECONDS - T0) / 60 )) min"
echo "Résultats : $OUT/"
