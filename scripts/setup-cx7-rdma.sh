#!/usr/bin/env bash
# =============================================================================
# setup-cx7-rdma.sh — enable PFC (Priority Flow Control, priority 3) on all 8
# GPU data NICs of AI2 + AI3, so the in-house ConnectX-7 RoCEv2 fabric stays
# lossless under load, BEFORE launching P/D servers.
#
# WHY
#   RDMA/RoCEv2 assumes a lossless network. Under high-concurrency KV transfer
#   (gsm8k conc=512) the fabric congests; without PFC the switch/NIC DROPS
#   packets, the RDMA QP retry-exhausts into Error State, and every in-flight
#   work request completes with "Work Request Flushed Error" (the flood we saw =
#   ~56% request aborts). PFC replaces drop-under-congestion with a link-level
#   PAUSE on the RDMA priority -> lossless -> no QP errors, no flush cascade.
#
# REFERENCE (matched to OUR fabric: Mellanox/mlx5 ConnectX-7, NOT Thor2/Broadcom)
#   amdsow/network_testing/TROUBLESHOOTING.md, "RoCE Fabric Issues":
#     PFC : mlnx_qos -i <nd> --pfc 0,0,0,1,0,0,0,0   (priority 3)
#   8 GPU-CNIC netdev names from amdsow/pruning/setup_alltoall_cnic.sh
#   (identical on AI2/AI3). NOTE: mlnx_qos was ported to AI3 from AI2's
#   mlnx-tools package (AI3 had no Mellanox repo).
#
# SCOPE (operator decisions, 2026-07-21)
#   - PFC on priority 3, ALL 8 data NICs, both nodes. Idempotent.
#   - ECN/DCQCN deferred: it is secondary rate-control (PFC is the flush fix),
#     and the mlx5 enable method differs from the reference (no roce_np sysfs;
#     lives in debugfs cc_params) — worked out separately.
#   - Does NOT touch MTU (left 4500) or IP routing.
#
# NOTES
#   - Host-wide config on a SHARED node; PFC-on is the correct RoCEv2 state.
#   - PFC is END-TO-END: the switch fabric must also honor PFC for full effect.
#   - Needs passwordless sudo on each node (confirmed for edwin.lim).
#
# USAGE
#   bash setup-cx7-rdma.sh
#   NODES="10.1.1.182 10.1.1.183" bash setup-cx7-rdma.sh
# =============================================================================
set -uo pipefail

NODES="${NODES:-10.1.1.182 10.1.1.183}"
PFC_PRIO="${PFC_PRIO:-0,0,0,1,0,0,0,0}"     # PFC on priority 3
# 8 GPU-CNIC data netdevs, COMMA-separated so the list survives ssh as ONE arg
# (a space-separated list gets re-split by ssh into separate positional args).
NETDEVS_CSV="ens41f0np0,ens42f0np0,ens32f0np0,ens31f0np0,ens21f0np0,ens22f0np0,ens12f0np0,ens11f0np0"

rc=0
for H in $NODES; do
  echo "############ $H ############"
  ssh -o BatchMode=yes "$H" 'bash -s' "$PFC_PRIO" "$NETDEVS_CSV" <<'REMOTE' || rc=1
set -uo pipefail
PFC_PRIO="$1"; IFS=, read -ra NETDEVS <<< "$2"
echo "== $(hostname) =="
for nd in "${NETDEVS[@]}"; do
  if [ ! -e "/sys/class/net/$nd" ]; then echo "  $nd: ABSENT, skip"; continue; fi
  sudo ethtool -A "$nd" rx on tx on >/dev/null 2>&1 || true
  sudo mlnx_qos -i "$nd" --pfc "$PFC_PRIO" >/dev/null 2>&1 && p=ok || p=FAIL
  got=$(mlnx_qos -i "$nd" 2>/dev/null | awk '/enabled/{$1="";gsub(/[[:space:]]/,"");print;exit}')
  echo "  $nd: pfc=$p  enabled=[$got]"
done
REMOTE
done
echo "=== done. Every NIC should read enabled=[00010000] (PFC on priority 3). ==="
exit $rc
