#!/bin/bash
# Orchestrate the MTP>0 energy sweep, hands-off. For each gamma in GAMMAS:
#   launch eagle3 server -> wait /health -> run concurrency sweep -> tear down.
# Resumable: whole levels and individual points are skipped once complete
# (row.json). Aborts after the first spec level if acceptance_rate didn't
# populate (so we don't waste the whole run on broken spec metrics).
#
# Run INSIDE the container (ee-spec-dev) on the host it will serve from.
#   bash /app/energy_harness/run_mtp_sweep.sh
set -u

MODEL=${MODEL:-/models/models/gpt-oss-120b}
SERVED=${SERVED:-gpt-oss-120b}
DRAFT=${DRAFT:-/models/models/eagle3-redhat}
GPU=${GPU:-0}
PORT=${PORT:-8776}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}
GAMMAS=${GAMMAS:-"1 2 3 4 5"}
CONCURRENCIES=${CONCURRENCIES:-"8 16 32 64 128 256 512"}
RESULT_DIR=${RESULT_DIR:-/app/data/results}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-480}          # seconds to wait for /health
SWEEP=${SWEEP:-/app/energy_harness/run_concurrency_sweep.sh}
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
    local g=$1 n=0
    for c in $CONCURRENCIES; do
        [ -f "${RESULT_DIR}/gptoss_mtp${g}_C${c}_specon/row.json" ] && n=$((n+1))
    done
    [ "$n" -eq "$N_CONC" ]
}

acceptance_of() { # $1=run_tag -> prints acceptance_rate (or empty)
    python3 - "$1" <<'PY' 2>/dev/null
import csv,sys,os
tag=sys.argv[1]; p=os.environ.get("RESULT_DIR","/app/data/results")+"/measurements.csv"
rows=[r for r in csv.DictReader(open(p)) if r["run_tag"]==tag] if os.path.exists(p) else []
print(rows[-1]["acceptance_rate"] if rows else "")
PY
}

trap 'echo "[orch] interrupted; stopping server"; stop_server; exit 130' INT TERM

echo "[orch] MODEL=$MODEL DRAFT=$DRAFT gammas=[$GAMMAS] concurrencies=[$CONCURRENCIES]"
stop_server   # clear any server already holding the GPU (e.g. the MTP0 one)

for g in $GAMMAS; do
    echo "================ MTP gamma=${g} ================"
    if level_done "$g"; then echo "[orch] gamma=${g} already complete -> skip"; continue; fi

    slog="${LOGDIR}/server_mtp${g}.log"
    echo "[orch] launching eagle3 server gamma=${g} (log: ${slog})"
    HIP_VISIBLE_DEVICES=${GPU} python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --served-model-name "$SERVED" \
        --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 1 \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --speculative-config "{\"method\":\"eagle3\",\"model\":\"${DRAFT}\",\"num_speculative_tokens\":${g}}" \
        > "$slog" 2>&1 &
    SPID=$!

    ready=0
    for _ in $(seq 1 $((STARTUP_TIMEOUT/5))); do
        server_up && { ready=1; break; }
        kill -0 "$SPID" 2>/dev/null || { echo "[orch] server process died during startup"; break; }
        sleep 5
    done
    if [ "$ready" -ne 1 ]; then
        echo "[orch] gamma=${g}: server not ready in ${STARTUP_TIMEOUT}s -> skipping. Last log lines:"
        tail -n 15 "$slog"
        stop_server "$SPID"
        continue
    fi
    echo "[orch] gamma=${g} server ready -> running concurrency sweep"

    SPEC=on NSPEC="$g" TAG_PREFIX="gptoss_mtp${g}" RESULT_DIR="$RESULT_DIR" \
        CONCURRENCIES="$CONCURRENCIES" bash "$SWEEP" \
        || echo "[orch] WARNING: gamma=${g} sweep returned nonzero"

    stop_server "$SPID"

    # After the first spec level, verify acceptance actually populated.
    firsttag="gptoss_mtp${g}_C$(echo $CONCURRENCIES | awk '{print $1}')_specon"
    acc=$(RESULT_DIR="$RESULT_DIR" acceptance_of "$firsttag")
    if [ -z "$acc" ] || [ "$acc" = "None" ]; then
        echo "[orch] ABORT: acceptance_rate='$acc' on ${firsttag} -> spec metrics NOT captured."
        echo "[orch] Energy/throughput are fine, but acceptance is missing; fix measure_one before continuing."
        exit 2
    fi
    echo "[orch] gamma=${g} OK (first-point acceptance_rate=${acc})"
done

echo "[orch] ALL MTP levels done -> ${RESULT_DIR}/measurements.csv"
