#!/bin/bash
# Save/restore the vllm/ee-spec image as a LOCAL-NVMe tar, so a docker-pruner sweep
# or a node reboot doesn't cost a ~30-min from-scratch rebuild. The tar survives
# both (unlike the keepalive container, which dies on reboot); `docker load` is
# ~minutes.
#
#   image_backup.sh save     [image]   -> docker save  -> $DTAR_DIR/<image>.tar
#   image_backup.sh restore  [image]   -> docker load  + re-pin keepalive
#
# IMPORTANT: DTAR_DIR must be a roomy LOCAL volume that is NOT the docker storage
# volume (/models here) -- adding a big tar to a near-full docker volume can trip
# the disk-threshold pruner. Default: /mnt/tmp (local nvme). Override per node.
set -u

CMD="${1:?usage: image_backup.sh save|restore [image]}"
IMAGE="${2:-vllm/ee-spec}"
DTAR_DIR="${DTAR_DIR:-/mnt/tmp/dtar}"      # local nvme; override per node
TAR="${DTAR_DIR}/$(echo "$IMAGE" | tr '/:' '__').tar"
KEEP_NAME="${KEEP_NAME:-ee-spec-keepalive}"

pin() {   # a running container marks the image in-use so the pruner skips it
    docker rm -f "$KEEP_NAME" >/dev/null 2>&1 || true
    docker run -d --restart always --name "$KEEP_NAME" --entrypoint sleep "$IMAGE" infinity >/dev/null
    docker ps --filter "name=$KEEP_NAME" --format 'pinned: {{.Names}} {{.Status}}'
}

case "$CMD" in
  save)
    mkdir -p "$DTAR_DIR"
    echo "[save] $IMAGE -> $TAR"; df -h "$DTAR_DIR" | tail -1
    docker save "$IMAGE" -o "$TAR" && ls -lh "$TAR" && echo "[save] OK"
    ;;
  restore)
    [ -f "$TAR" ] || { echo "[restore] no tar at $TAR (set DTAR_DIR?)" >&2; exit 1; }
    echo "[restore] $TAR -> docker"
    docker load -i "$TAR" && pin && echo "[restore] OK"
    ;;
  *) echo "usage: image_backup.sh save|restore [image]" >&2; exit 2 ;;
esac
