#!/usr/bin/env bash
# =============================================================================
# setup-cx7-rdma.sh — put the in-house ConnectX-7 RoCEv2 fabric into the GOOD
# STATE on AI2 + AI3 before launching P/D servers.
#
# WHAT / WHY
#   Enables PFC (Priority Flow Control, priority 3) + RoCE ECN on all 8 GPU data
#   NICs of each node. PFC being OFF is the root cause of "Work Request Flushed
#   Error / QP Error State" flush cascades under high-concurrency KV transfer
#   (observed ~56% request aborts at gsm8k conc=512). PFC gives lossless pause
#   under congestion so QPs don't retry-exhaust into Error State and flush.
#
# REFERENCE (matched to OUR fabric: Mellanox/mlx5 ConnectX-7 — NOT Thor2/Broadcom)
#   amdsow/network_testing/TROUBLESHOOTING.md, "RoCE Fabric Issues":
#     PFC : mlnx_qos -i <nd> --pfc 0,0,0,1,0,0,0,0    (priority 3)
#     ECN : mlnx_qos -i <nd> --ecn ; sysctl net.ipv4.tcp_ecn=1
#   8 GPU-CNIC netdev names taken from amdsow/pruning/setup_alltoall_cnic.sh
#   (identical names on AI2 and AI3). Verified live: mlnx_qos present, PFC was
#   "enabled 0 0 0 0 0 0 0 0" (off) before this script.
#
# SCOPE (operator decisions, 2026-07-21)
#   - Configure ALL 8 data NICs.
#   - Do NOT change MTU (left at 4500 — jumbo/9000 needs switch+peer agreement).
#   - Do NOT touch IP addressing or routing (our path uses GID3 / per-NIC
#     100.33.4X, which is already unambiguous).
#
# NOTES
#   - Host-wide config on a SHARED node: PFC/ECN affect every RDMA user of these
#     NICs. PFC-on is the correct RoCEv2 state, so this helps rather than harms.
#   - PFC is END-TO-END: the switch fabric must also honor PFC for full effect.
#   - Requires passwordless sudo on each node (confirmed for edwin.lim).
#   - Idempotent: re-running just re-asserts the same settings.
#
# USAGE
#   bash setup-cx7-rdma.sh                              # default NODES = AI2 AI3
#   NODES="10.1.1.182 10.1.1.183" bash setup-cx7-rdma.sh
# =============================================================================
set -uo pipefail

NODES="${NODES:-10.1.1.182 10.1.1.183}"
PFC_PRIO="${PFC_PRIO:-0,0,0,1,0,0,0,0}"     # PFC on priority 3
# 8 GPU-CNIC data netdevs (from setup_alltoall_cnic.sh; identical on AI2/AI3)
NETDEVS="ens41f0np0 ens42f0np0 ens32f0np0 ens31f0np0 ens21f0np0 ens22f0np0 ens12f0np0 ens11f0np0"

rc=0
for H in $NODES; do
  echo "############ $H ############"
  ssh -o BatchMode=yes "$H" 'bash -s' "$PFC_PRIO" "$NETDEVS" <<'REMOTE' || rc=1
set -uo pipefail
PFC_PRIO="$1"; read -ra NETDEVS <<< "$2"
echo "== $(hostname) =="
sudo sysctl -qw net.ipv4.tcp_ecn=1 >/dev/null 2>&1 && echo "  tcp_ecn=1" || echo "  tcp_ecn: could not set"
for nd in "${NETDEVS[@]}"; do
  if [ ! -e "/sys/class/net/$nd" ]; then echo "  $nd: ABSENT, skip"; continue; fi
  sudo ethtool -A "$nd" rx on tx on >/dev/null 2>&1 || true
  sudo mlnx_qos -i "$nd" --pfc "$PFC_PRIO" >/dev/null 2>&1 && p=ok || p=FAIL
  sudo mlnx_qos -i "$nd" --ecn           >/dev/null 2>&1 && e=ok || e=skip
  got=$(mlnx_qos -i "$nd" 2>/dev/null | awk '/enabled/{$1="";gsub(/[[:space:]]/,"");print;exit}')
  echo "  $nd: pfc=$p ecn=$e  enabled=[$got]"
done
REMOTE
done

echo "=== done. Each NIC should show enabled=[00010000] (PFC on priority 3). ==="
exit $rc
