#!/bin/bash
# Shared model matrix for the MoE-vs-dense comparison (accuracy + energy sweeps).
# Each combo is a (target, eagle3 draft) pair, one GPU each. Edit the *_PATH
# overrides if downloads landed somewhere other than the defaults below.
#
# ALL drafts are RedHat "speculators"-format EAGLE-3 heads (vLLM-canonical loader,
# same path proven stable for gpt-oss). The yuhuili original-EAGLE Llama heads were
# dropped: the 8B one hit a GPU memory-access fault on this ROCm build.
#
# Sourced by accuracy_parallel.sh and run_energy_model.sh.

# All comparison combos (small->large). gpt-oss-120b is already done separately.
MODELS_ALL=${MODELS_ALL:-"llama31_8b gptoss20b llama33_70b gptoss120b"}

# model_cfg <tag> -> sets MODEL DRAFT SERVED TOKENIZER; returns 1 on unknown tag.
model_cfg() {
    case "$1" in
        llama31_8b)
            MODEL=${LLAMA31_8B_PATH:-/models/models/meta-llama--Llama-3.1-8B-Instruct}
            DRAFT=${LLAMA31_8B_DRAFT:-/models/models/eagle3-llama31-8b-redhat}
            SERVED=llama31-8b ;;
        llama33_70b)
            MODEL=${LLAMA33_70B_PATH:-/models/models/Llama-3.3-70B-Instruct}
            DRAFT=${LLAMA33_70B_DRAFT:-/models/models/eagle3-llama33-70b-redhat}
            SERVED=llama33-70b ;;
        gptoss20b)
            MODEL=${GPTOSS20B_PATH:-/models/models/gpt-oss-20b}
            DRAFT=${GPTOSS20B_DRAFT:-/models/models/eagle3-gptoss20b}
            SERVED=gpt-oss-20b ;;
        gptoss120b)
            MODEL=${GPTOSS120B_PATH:-/models/models/gpt-oss-120b}
            DRAFT=${GPTOSS120B_DRAFT:-/models/models/eagle3-redhat}
            SERVED=gpt-oss-120b ;;
        *) return 1 ;;
    esac
    TOKENIZER="$MODEL"
    return 0
}
