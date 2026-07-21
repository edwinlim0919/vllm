#!/bin/bash
# Sequential MTP x concurrency ENERGY sweep for ONE model, on a SINGLE GPU.
# Same grid + measurement as the gpt-oss-120b run: gamma 0..5 x C 8..512.
# Per gamma: launch server (plain for gamma0, eagle3 draft for gamma>0) ->
# wait /health -> run the concurrency sweep -> tear down. Neighbors stay idle so
# GPU 0's power isn't polluted; do NOT run two of these at once on one node.
#
# Run INSIDE the container:
#   MTAG=llama31_8b bash /app/energy_harness/run_energy_model.sh
# Resumable: whole gamma levels and individual C points are skipped once their
# row.json exists. Aborts a spec level if acceptance_rate never populated.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/models.sh"

MTAG=${MTAG:?set MTAG=llama31_8b|gptoss20b|llama33_70b}
model_cfg "$MTAG" || { echo "[energy] unknown MTAG=$MTAG"; exit 1; }

GPU=${GPU:-0}
PORT=${PORT:-8776}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}   # cap context; 70B+eagle3 KV won't fit the 128k default -> engine OOMs at init
GAMMAS=${GAMMAS:-"0 1 2 3 4 5"}
CONCURRENCIES=${CONCURRENCIES:-"8 16 32 64 128 256 512"}
RESULT_DIR=${RESULT_DIR:-/app/data/results}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1200}
WINDOW=${WINDOW:-15}
SWEEP=${SWEEP:-${HERE}/run_concurrency_sweep.sh}
LOGDIR="${RESULT_DIR}/orchestrator_logs"
mkdir -p "$LOGDIR"
N_CONC=$(echo $CONCURRENCIES | wc -w)

server_up() { curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; }

stop_server() {   # graceful, then hard, then let the GPU free
    local pid="${1:-}"
    [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true
    pkill -TERM -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    for _ in $(seq 1 40); do
        pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null || break
        sleep 2
    done
    pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 15
}

level_done() {    # $1=gamma ; true iff every concurrency point has a row.json
    local g=$1 n=0 sfx=on
    [ "$g" -eq 0 ] && sfx=off
    for c in $CONCURRENCIES; do
        [ -f "${RESULT_DIR}/${MTAG}_mtp${g}_C${c}_spec${sfx}/row.json" ] && n=$((n+1))
    done
    [ "$n" -eq "$N_CONC" ]
}

acceptance_of() { # $1=run_tag -> prints acceptance_rate (or empty)
    RESULT_DIR="$RESULT_DIR" python3 - "$1" <<'PY' 2>/dev/null
import csv,sys,os
tag=sys.argv[1]; p=os.environ.get("RESULT_DIR","/app/data/results")+"/measurements.csv"
rows=[r for r in csv.DictReader(open(p)) if r["run_tag"]==tag] if os.path.exists(p) else []
print(rows[-1]["acceptance_rate"] if rows else "")
PY
}

trap 'echo "[energy] interrupted; stopping server"; stop_server; exit 130' INT TERM

echo "[energy] MTAG=$MTAG MODEL=$MODEL DRAFT=$DRAFT gammas=[$GAMMAS] C=[$CONCURRENCIES] GPU=$GPU"
[ -f "$MODEL/config.json" ] || { echo "[energy] MODEL not found: $MODEL/config.json -> abort"; exit 1; }
stop_server   # clear any server already holding the GPU

for g in $GAMMAS; do
    echo "================ ${MTAG} MTP gamma=${g} ================"
    if level_done "$g"; then echo "[energy] gamma=${g} already complete -> skip"; continue; fi

    SPEC=off; spec_args=()
    if [ "$g" -gt 0 ]; then
        if [ ! -f "$DRAFT/config.json" ]; then
            echo "[energy] gamma=${g}: DRAFT not found ($DRAFT/config.json) -> skip level"; continue
        fi
        SPEC=on
        spec_args=(--speculative-config \
            "{\"method\":\"eagle3\",\"model\":\"${DRAFT}\",\"num_speculative_tokens\":${g}}")
    fi

    slog="${LOGDIR}/server_${MTAG}_mtp${g}.log"
    echo "[energy] launching gamma=${g} server (spec=${SPEC}, log: ${slog})"
    HIP_VISIBLE_DEVICES=${GPU} python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --served-model-name "$SERVED" \
        --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 1 \
        --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
        "${spec_args[@]}" > "$slog" 2>&1 &
    SPID=$!

    ready=0
    for _ in $(seq 1 $((STARTUP_TIMEOUT/5))); do
        server_up && { ready=1; break; }
        kill -0 "$SPID" 2>/dev/null || { echo "[energy] server died during startup"; break; }
        sleep 5
    done
    if [ "$ready" -ne 1 ]; then
        echo "[energy] gamma=${g}: server not ready in ${STARTUP_TIMEOUT}s -> skip. Last log:"
        tail -n 15 "$slog"; stop_server "$SPID"; continue
    fi
    echo "[energy] gamma=${g} server ready -> concurrency sweep"

    SPEC="$SPEC" NSPEC="$g" TAG_PREFIX="${MTAG}_mtp${g}" \
        MODEL="$SERVED" TOKENIZER="$TOKENIZER" MODEL_LABEL="$MTAG" \
        BASE_URL="http://localhost:${PORT}" GPU="$GPU" \
        RESULT_DIR="$RESULT_DIR" WINDOW="$WINDOW" CONCURRENCIES="$CONCURRENCIES" \
        bash "$SWEEP" || echo "[energy] WARNING: gamma=${g} sweep returned nonzero"

    stop_server "$SPID"

    if [ "$g" -gt 0 ]; then
        firsttag="${MTAG}_mtp${g}_C$(echo $CONCURRENCIES | awk '{print $1}')_specon"
        acc=$(acceptance_of "$firsttag")
        if [ -z "$acc" ] || [ "$acc" = "None" ]; then
            echo "[energy] ABORT: acceptance_rate='$acc' on ${firsttag} -> spec metrics NOT captured."
            exit 2
        fi
        echo "[energy] gamma=${g} OK (first-point acceptance_rate=${acc})"
    fi
done

echo "[energy] ${MTAG} ALL gammas done -> ${RESULT_DIR}/measurements.csv"
