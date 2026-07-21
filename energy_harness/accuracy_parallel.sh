#!/bin/bash
# Parallel GSM8K accuracy / losslessness gate for the MoE-vs-dense comparison.
#
# One job = (model, gamma). For each job: launch a 1-GPU vLLM server (plain for
# gamma=0, eagle3 draft for gamma>0), run GSM8K (EVAL_LIMIT prompts, InferenceX
# recipe), record exact_match, tear the server down. A work-stealing scheduler
# keeps all NGPU GPUs busy until every job is done -> no idle GPU.
#
# At temp 0 EAGLE-3 is lossless, so every gamma>0 must match its own gamma=0
# accuracy for a model. A drop => spec broken for that combo (stop before energy).
#
# Run INSIDE the container, on the node holding the models:
#   bash /app/energy_harness/accuracy_parallel.sh
# Resumable: a job whose results json already exists is skipped.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/models.sh"

MODELS=${MODELS:-$MODELS_ALL}
GAMMAS=${GAMMAS:-"0 1 2 3 4 5"}
NGPU=${NGPU:-8}
BASE_PORT=${BASE_PORT:-8800}
EVAL_LIMIT=${EVAL_LIMIT:-256}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}      # e.g. "--enforce-eager" for the eagle3-llama31-8b coredump
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1200}      # 70B cold-load can be slow
ACC_DIR=${ACC_DIR:-/app/data/accuracy}
GSM8K=${GSM8K:-${HERE}/run_gsm8k.sh}
SUMMARY="${ACC_DIR}/summary.csv"
LOGDIR="${ACC_DIR}/logs"
mkdir -p "$ACC_DIR" "$LOGDIR"
[ -f "$SUMMARY" ] || echo "model,gamma,strict_match,flexible_match,n,status" > "$SUMMARY"

# Install lm-eval + API extras (tenacity/aiohttp for local-chat-completions) ONCE
# up front so the parallel workers don't race on pip.
python3 -c "import lm_eval, tenacity" 2>/dev/null || pip install --quiet "lm-eval[api]"

# run_job <model_tag> <gamma> <gpu> : full server-up -> gsm8k -> teardown for one job.
run_job() {
    local mtag=$1 g=$2 gpu=$3
    local port=$((BASE_PORT + gpu))
    local label="${mtag}_mtp${g}"
    local slog="${LOGDIR}/${label}.log"
    if ! model_cfg "$mtag"; then echo "[acc] unknown model $mtag"; return 1; fi

    if find "${ACC_DIR}/${label}" -name 'results*.json' 2>/dev/null | grep -q .; then
        echo "[acc] skip ${label} (already done)"; return 0
    fi

    # Pre-flight: fail in ms (not after the 1200s health timeout) if the model
    # or (for spec) the draft head isn't a real local dir.
    if [ ! -f "$MODEL/config.json" ]; then
        echo "[acc] ${label}: MODEL not found ($MODEL/config.json) -> SKIP"
        echo "${mtag},${g},,,${EVAL_LIMIT},no_model_dir" >> "$SUMMARY"; return 1
    fi
    if [ "$g" -gt 0 ] && [ ! -f "$DRAFT/config.json" ]; then
        echo "[acc] ${label}: DRAFT not found ($DRAFT/config.json) -> SKIP"
        echo "${mtag},${g},,,${EVAL_LIMIT},no_draft_dir" >> "$SUMMARY"; return 1
    fi

    local spec_args=()
    [ "$g" -gt 0 ] && spec_args=(--speculative-config \
        "{\"method\":\"eagle3\",\"model\":\"${DRAFT}\",\"num_speculative_tokens\":${g}}")

    echo "[acc] START ${label}  GPU=${gpu} port=${port}"
    # setsid -> own process group so teardown kills ONLY this server's tree
    # (a global pkill would nuke the other GPUs' servers).
    HIP_VISIBLE_DEVICES=${gpu} setsid python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --served-model-name "$SERVED" \
        --host 0.0.0.0 --port "$port" --tensor-parallel-size 1 \
        --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" $EXTRA_SERVER_ARGS \
        "${spec_args[@]}" > "$slog" 2>&1 &
    local spid=$!

    local ready=0
    for _ in $(seq 1 $((STARTUP_TIMEOUT/5))); do
        curl -sf "http://localhost:${port}/health" >/dev/null 2>&1 && { ready=1; break; }
        kill -0 "$spid" 2>/dev/null || break
        sleep 5
    done

    if [ "$ready" -eq 1 ]; then
        MODEL_NAME="$SERVED" EVAL_LIMIT="$EVAL_LIMIT" NUM_CONCURRENT=$((MAX_NUM_SEQS-4)) \
            bash "$GSM8K" "http://localhost:${port}" "$label" "$ACC_DIR" \
            >> "$slog" 2>&1 || echo "[acc] ${label}: gsm8k returned nonzero"
    else
        echo "[acc] ${label}: server NOT ready in ${STARTUP_TIMEOUT}s -> FAIL"; tail -n 20 "$slog"
    fi

    # Teardown: kill the whole process group, wait for the GPU to free.
    kill -TERM -"$spid" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$spid" 2>/dev/null || break; sleep 2; done
    kill -KILL -"$spid" 2>/dev/null; wait "$spid" 2>/dev/null
    sleep 5

    python3 - "$mtag" "$g" "$ACC_DIR" "$label" "$EVAL_LIMIT" "$SUMMARY" "$ready" <<'PY'
import sys, glob, json, os
mtag, g, accdir, label, lim, summ, ready = sys.argv[1:8]
files = glob.glob(os.path.join(accdir, label, "**", "results*.json"), recursive=True)
strict = flex = ""; status = "server_fail" if ready != "1" else "no_results"
if files:
    r = json.load(open(sorted(files)[-1])).get("results", {}).get("gsm8k", {})
    strict = r.get("exact_match,strict-match", "")
    flex = r.get("exact_match,flexible-extract", "")
    status = "ok" if strict != "" else status
open(summ, "a").write(f"{mtag},{g},{strict},{flex},{lim},{status}\n")
print(f"[acc] DONE {label}: strict={strict} flexible={flex} ({status})")
PY
}

# ---- build job list ----
JOBS=()
for m in $MODELS; do for g in $GAMMAS; do JOBS+=("${m}:${g}"); done; done
NJOBS=${#JOBS[@]}
echo "[acc] ${NJOBS} jobs across ${NGPU} GPUs: models=[${MODELS}] gammas=[${GAMMAS}] limit=${EVAL_LIMIT}"

# ---- work-stealing scheduler over NGPU GPUs (FIFO signals a freed GPU) ----
FIFO=$(mktemp -u); mkfifo "$FIFO"; exec 3<>"$FIFO"; rm -f "$FIFO"
launch() {   # $1="model:gamma"  $2=gpu ; worker signals its GPU back when done
    local job=$1 gpu=$2
    ( run_job "${job%:*}" "${job#*:}" "$gpu"; echo "$gpu" >&3 ) &
}
i=0
primed=$(( NJOBS < NGPU ? NJOBS : NGPU ))
for gpu in $(seq 0 $((primed-1))); do launch "${JOBS[$i]}" "$gpu"; i=$((i+1)); done
while [ $i -lt $NJOBS ]; do read -u 3 freed; launch "${JOBS[$i]}" "$freed"; i=$((i+1)); done
for _ in $(seq 1 $primed); do read -u 3 _; done
wait
exec 3>&-

echo "[acc] ================ SUMMARY ================"
python3 - "$SUMMARY" <<'PY'
import csv, collections
rows = list(csv.DictReader(open(csv_path := __import__("sys").argv[1])))
by = collections.defaultdict(dict)
for r in rows: by[r["model"]][r["gamma"]] = r
for m, gs in by.items():
    base = gs.get("0", {}).get("strict_match", "")
    print(f"\n{m}   (gamma0 strict={base or 'n/a'})")
    for g in sorted(gs, key=int):
        r = gs[g]; s = r["strict_match"]; f = r["flexible_match"]; st = r["status"]
        flag = ""
        try:
            if base and s and abs(float(s) - float(base)) > 0.02: flag = "  <-- DEVIATES from gamma0"
            if st != "ok": flag = f"  <-- {st}"
        except ValueError: pass
        print(f"  mtp{g}: strict={s or '----'} flexible={f or '----'}{flag}")
print("\n[acc] lossless if every mtp>0 strict ~= its mtp0 strict (within noise).")
PY
