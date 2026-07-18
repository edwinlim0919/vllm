#!/bin/bash
# Closed-loop concurrency sweep for energy measurement (inferenceX style), Spec-Bench.
#
# A single `vllm serve` must ALREADY be running, launched with --max-num-seqs >=
# the largest C. Concurrency is set per point on the CLIENT via --max-concurrency,
# so the server is never relaunched. Per point: warmup = 2*C, benchmark = 10*C,
# fixed window T. Resumable: a C whose result dir already exists is skipped.
#
# Requires the Spec-Bench question.jsonl (download once, into DATASET_PATH):
#   wget -O /app/energy_harness/spec_bench_question.jsonl \
#     https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl
#
# Override any knob via env, e.g.:  WINDOW=10 SPEC=off bash run_concurrency_sweep.sh
set -u

HARNESS=${HARNESS:-/app/energy_harness/measure_one.py}
BASE_URL=${BASE_URL:-http://localhost:8776}
MODEL=${MODEL:-gpt-oss-120b}
TOKENIZER=${TOKENIZER:-/models/models/gpt-oss-120b}
GPU=${GPU:-0}
DATASET_PATH=${DATASET_PATH:-/app/data/spec_bench_question.jsonl}
CATEGORY=${CATEGORY:-}          # empty = all Spec-Bench categories
OUT_LEN=${OUT_LEN:-256}
WINDOW=${WINDOW:-15}
SEED=${SEED:-0}
TEMP=${TEMP:-0.0}               # 0 = greedy -> identical spec-on/off outputs
SPEC=${SPEC:-off}
NSPEC=${NSPEC:-0}
MODEL_LABEL=${MODEL_LABEL:-gpt-oss-120b}
RESULT_DIR=${RESULT_DIR:-/app/data/results}   # NFS-mounted -> persists past the --rm container
TAG_PREFIX=${TAG_PREFIX:-gptoss_specbench}
CONCURRENCIES=${CONCURRENCIES:-"8 16 32 64 128 256 512"}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-512}       # one-time pre-sweep warmup (0 = skip)
WARMUP_CONCURRENCY=${WARMUP_CONCURRENCY:-64}

echo "[sweep] server=${BASE_URL} model=${MODEL} dataset=spec_bench spec=${SPEC} nspec=${NSPEC} window=${WINDOW}s"
echo "[sweep] concurrencies: ${CONCURRENCIES}"

CAT_ARG=""
[ -n "$CATEGORY" ] && CAT_ARG="--spec-bench-category $CATEGORY"

# One-time warmup so the first (lowest-C) measured point isn't cold: a substantial
# blast that JIT-compiles kernels (aiter) and ramps GPU clocks. Result discarded.
if [ "$WARMUP_REQUESTS" -gt 0 ]; then
    echo "[sweep] pre-sweep warmup: ${WARMUP_REQUESTS} reqs @ concurrency ${WARMUP_CONCURRENCY} (discarded)"
    vllm bench serve --backend openai --base-url "$BASE_URL" --endpoint /v1/completions \
        --model "$MODEL" --tokenizer "$TOKENIZER" \
        --dataset-name spec_bench --dataset-path "$DATASET_PATH" \
        --spec-bench-output-len "$OUT_LEN" $CAT_ARG \
        --num-prompts "$WARMUP_REQUESTS" --max-concurrency "$WARMUP_CONCURRENCY" \
        --request-rate inf --seed "$SEED" --temperature "$TEMP" \
        >/dev/null 2>&1 && echo "[sweep] warmup done" || echo "[sweep] WARNING: warmup failed (continuing)"
fi

for C in $CONCURRENCIES; do
    NUM_PROMPTS=$((10 * C))
    WARMUP=$((2 * C))
    RUN_TAG="${TAG_PREFIX}_C${C}_spec${SPEC}"
    # Skip only COMPLETED points (row.json is written by measure_one on success);
    # a failed/empty dir from an earlier crash re-runs instead of being skipped.
    if [ -f "${RESULT_DIR}/${RUN_TAG}/row.json" ]; then
        echo "[sweep] skip ${RUN_TAG} (already completed)"
        continue
    fi
    echo "[sweep] ===== C=${C}  num_prompts=${NUM_PROMPTS}  warmup=${WARMUP}  tag=${RUN_TAG} ====="
    python "$HARNESS" \
        --base-url "$BASE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
        --gpu "$GPU" --max-concurrency "$C" \
        --dataset-name spec_bench --dataset-path "$DATASET_PATH" \
        --spec-bench-output-len "$OUT_LEN" $CAT_ARG \
        --num-prompts "$NUM_PROMPTS" --warmup-prompts "$WARMUP" \
        --window-seconds "$WINDOW" --seed "$SEED" --temperature "$TEMP" \
        --max-num-seqs "$C" --spec "$SPEC" --num-speculative-tokens "$NSPEC" \
        --model-label "$MODEL_LABEL" --result-dir "$RESULT_DIR" --run-tag "$RUN_TAG" \
        || echo "[sweep] WARNING: C=${C} failed (continuing)"
done
echo "[sweep] done -> ${RESULT_DIR}/measurements.csv"
