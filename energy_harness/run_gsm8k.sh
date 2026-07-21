#!/bin/bash
# GSM8K accuracy against an ALREADY-RUNNING vLLM OpenAI endpoint, replicating the
# InferenceX recipe: lm-eval-harness `local-chat-completions`, --apply_chat_template,
# 5-shot, GREEDY (temperature 0), custom gsm8k.yaml, long context/generation.
#
# At temp 0, EAGLE-3 is lossless, so spec-off and spec-on runs must yield the SAME
# accuracy (and per-question answers). A collapse => the vLLM #27626 accuracy bug
# reproducing on ROCm -> stop and debug before any energy runs.
#
# Usage:  run_gsm8k.sh <base-url> <label> <result-dir>
#   e.g.  run_gsm8k.sh http://localhost:8776 gptoss_specoff /app/data/accuracy
#
# The server MUST be launched with a long context (--max-model-len 32768) and
# --max-num-seqs >= NUM_CONCURRENT, identical for spec-off and spec-on.
set -u

BASE_URL="${1:?usage: run_gsm8k.sh <base-url> <label> <result-dir>}"
LABEL="${2:?usage: run_gsm8k.sh <base-url> <label> <result-dir>}"
RESULT_DIR="${3:?usage: run_gsm8k.sh <base-url> <label> <result-dir>}"

MODEL_NAME="${MODEL_NAME:-gpt-oss-120b}"
TASK_YAML="${TASK_YAML:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/evals/gsm8k.yaml}"
NUM_CONCURRENT="${NUM_CONCURRENT:-60}"
MAX_LEN="${MAX_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TIMEOUT="${TIMEOUT:-1800}"
LIMIT_ARG=""
[ -n "${EVAL_LIMIT:-}" ] && LIMIT_ARG="--limit ${EVAL_LIMIT}"   # e.g. EVAL_LIMIT=50 for a quick smoke

# lm-eval + API extras on demand (bake into the image later if we run this a lot).
python3 -c "import lm_eval, tenacity" 2>/dev/null || pip install --quiet "lm-eval[api]"

mkdir -p "$RESULT_DIR"
CHAT_URL="${BASE_URL%/}/v1/chat/completions"
OUT="${RESULT_DIR}/${LABEL}"

echo "[gsm8k] endpoint=${CHAT_URL} model=${MODEL_NAME} conc=${NUM_CONCURRENT} max_len=${MAX_LEN} max_tokens=${MAX_TOKENS} ${LIMIT_ARG}"
python3 -m lm_eval --model local-chat-completions --apply_chat_template \
    --tasks "${TASK_YAML}" \
    --model_args "model=${MODEL_NAME},base_url=${CHAT_URL},api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=${NUM_CONCURRENT},timeout=${TIMEOUT},tokenized_requests=False,max_length=${MAX_LEN}" \
    --gen_kwargs "max_tokens=${MAX_TOKENS},temperature=0,top_p=1" \
    --output_path "${OUT}" --log_samples ${LIMIT_ARG} \
    2>&1 | tee "${RESULT_DIR}/${LABEL}.log"

echo "[gsm8k] results -> ${OUT}   (exact_match under strict-match / flexible-extract)"
