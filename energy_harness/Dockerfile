# Overlay image: bakes energy-measurement tooling (Zeus) onto an already-built
# vLLM ROCm base image, WITHOUT a full rebuild (no build cache needed).
#
#   docker build -t vllm/ee-spec - < energy_harness/Dockerfile
#
# docker/Dockerfile.rocm also installs zeus, so a full from-source rebuild bakes
# it in too; this overlay is the fast path for an existing/transferred image.
FROM vllm/ee-spec
RUN python3 -m pip install --no-cache-dir zeus
