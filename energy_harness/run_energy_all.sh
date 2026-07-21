#!/bin/bash
# Run the full energy sweep for every comparison model, STRICTLY SEQUENTIALLY on
# one GPU. Nothing runs in parallel: a neighbor GPU under load would pollute the
# power reading. Each model does gamma 0..5 x C 8..512 (same grid as gpt-oss-120b).
#
# Run INSIDE the container, alone on the node:
#   bash /app/energy_harness/run_energy_all.sh
# Resumable end-to-end (per-model, per-gamma, per-C via row.json).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/models.sh"

MODELS=${MODELS:-$MODELS_ALL}
export GPU=${GPU:-0}

echo "[energy-all] models=[${MODELS}] GPU=${GPU} (sequential, no parallelism)"
for m in $MODELS; do
    echo "############################ ENERGY: ${m} ############################"
    MTAG="$m" bash "${HERE}/run_energy_model.sh" \
        || echo "[energy-all] WARNING: ${m} returned nonzero (continuing)"
done
echo "[energy-all] ALL MODELS DONE -> ${RESULT_DIR:-/app/data/results}/measurements.csv"
