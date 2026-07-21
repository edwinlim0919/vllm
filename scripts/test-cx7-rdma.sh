#!/usr/bin/env bash
# =============================================================================
# test-cx7-rdma.sh — validate the ConnectX-7 RoCEv2 fabric for the PD workload
# (MoRIIO READ). Run AFTER setup-cx7-rdma.sh (PFC on).
#
# PD mapping (why this shape): in MoRIIO READ mode the DECODE side is the RDMA
# READ *initiator* and the PREFILL side is the *target* whose KV memory is read.
# So ib_read_bw server = prefill (AI2, memory source), client = decode (AI3,
# initiator). Rails + GID index match the sweep (rocep6s0f0/rocep35s0f0, GID 3).
#
#   Layer A — path sanity : 1-QP RDMA READ per rail. QP must come up; note BW.
#   Layer B — congestion  : heavy READ (16 QP x 1MB, both rails). PFC PASS =
#                           NIC PAUSES (prio3_pause climbs) instead of DROPPING
#                           (rx_out_of_buffer / *_discards_phy stay 0).
#   Layer C — MoRI layer  : TODO — mori benchmark needs torchrun+mori; run from
#                           the image (vllm-mori-pd:inhouse-cx7-replant-infx).
#
# Sources: perftest ib_read_bw (linux-rdma/perftest) is the standard RDMA-READ
# benchmark; READ direction matches MoRIIO READ mode (decode logs: "MoRIIO READ
# transfer failed"). PFC pause-vs-drop is the RoCE lossless mechanism
# (network_testing/TROUBLESHOOTING.md; DCQCN paper Zhu et al. SIGCOMM'15).
# =============================================================================
set -uo pipefail

PREFILL="${PREFILL:-10.1.1.182}"   # AI2 = READ target (ib_read_bw server)
DECODE="${DECODE:-10.1.1.183}"     # AI3 = READ initiator (ib_read_bw client)
GID="${GID:-3}"
RAILS=(rocep6s0f0 rocep35s0f0)
declare -A SIP=([rocep6s0f0]=100.33.40.2 [rocep35s0f0]=100.33.41.2)   # AI2 rail IPs
declare -A ND=([rocep6s0f0]=ens41f0np0 [rocep35s0f0]=ens42f0np0)      # netdevs
declare -A PORT=([rocep6s0f0]=18515 [rocep35s0f0]=18516)              # distinct ports

sshb(){ ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$@" 2>/dev/null; }
kill_all(){ sshb "$PREFILL" 'pkill -f ib_read_bw 2>/dev/null'; sshb "$DECODE" 'pkill -f ib_read_bw 2>/dev/null'; }
trap kill_all EXIT
kill_all; sleep 1

# ---------------- Layer A ----------------
echo "################ Layer A: RDMA READ path sanity (1 QP/rail, GID$GID) ################"
printf "  %-14s %-22s %s\n" "rail" "AI3 decode -> AI2" "result (#bytes iters BWpeak BWavg Gb/s)"
for d in "${RAILS[@]}"; do
  sshb "$PREFILL" "nohup ib_read_bw -d $d -x $GID -p ${PORT[$d]} -F -D 5 --report_gbits >/tmp/a_$d 2>&1 &"
  sleep 2
  row=$(sshb "$DECODE" "ib_read_bw -d $d -x $GID -p ${PORT[$d]} -F -D 5 --report_gbits ${SIP[$d]} 2>/dev/null" \
        | grep -E '^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9.]+' | tail -1)
  printf "  %-14s %-22s %s\n" "$d" "${SIP[$d]}" "${row:-NO RESULT (QP did not come up)}"
  kill_all; sleep 1
done

# ---------------- Layer B ----------------
echo
echo "################ Layer B: congestion / PFC (heavy READ 16 QP x 1MB, both rails) ################"
CTRS='prio3_pause|rx_out_of_buffer|_discards_phy'
snap(){ local H=$1 tag=$2; for d in "${RAILS[@]}"; do
  sshb "$H" "ethtool -S ${ND[$d]} 2>/dev/null | grep -E '$CTRS'" | sed "s/^ *//; s#:# #; s/^/$tag ${ND[$d]} /"
done; }

{ snap "$PREFILL" AI2; snap "$DECODE" AI3; } > /tmp/rdma_ctrs_before 2>/dev/null

for d in "${RAILS[@]}"; do
  sshb "$PREFILL" "nohup ib_read_bw -d $d -x $GID -p ${PORT[$d]} -q 16 -s 1048576 -F -D 12 --report_gbits >/tmp/bs_$d 2>&1 &"
done
sleep 2
for d in "${RAILS[@]}"; do
  sshb "$DECODE" "nohup ib_read_bw -d $d -x $GID -p ${PORT[$d]} -q 16 -s 1048576 -F -D 12 --report_gbits ${SIP[$d]} >/tmp/bc_$d 2>&1 &"
done
echo "  ...driving ~12s of heavy READ load..."
sleep 15

{ snap "$PREFILL" AI2; snap "$DECODE" AI3; } > /tmp/rdma_ctrs_after 2>/dev/null

echo "  per-rail aggregate BW (AI3 client, 16 QP):"
for d in "${RAILS[@]}"; do
  row=$(sshb "$DECODE" "cat /tmp/bc_$d 2>/dev/null" | grep -E '^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9.]+' | tail -1)
  printf "    %-14s %s\n" "$d" "${row:-NO RESULT}"
done

echo "  counter deltas (after-before) — pause MAY climb (PFC engaged); discards/out_of_buffer MUST be 0:"
awk 'NR==FNR{b[$1" "$2" "$3]=$4; next}
     { key=$1" "$2" "$3; d=$4-(b[key]+0); if(d!=0) printf "    %-4s %-12s %-26s +%d\n",$1,$2,$3,d }' \
     /tmp/rdma_ctrs_before /tmp/rdma_ctrs_after | sort || echo "    (no counter changes)"
verdict=$(awk 'NR==FNR{b[$1" "$2" "$3]=$4; next}
     { key=$1" "$2" "$3; d=$4-(b[key]+0); if($3 ~ /out_of_buffer|discards_phy/ && d>0) bad++ }
     END{ print (bad>0)? "FAIL: drops occurred (PFC not holding)" : "PASS: no drops (PFC lossless under load)" }' \
     /tmp/rdma_ctrs_before /tmp/rdma_ctrs_after)
echo "  >>> Layer B verdict: $verdict"

echo
echo "################ Layer C: MoRI-layer — TODO (run mori benchmark via the image) ################"
echo "  mori benchmark needs torchrun + mori (built in vllm-mori-pd:inhouse-cx7-replant-infx)."
echo "  Deferred: run after A/B confirm the raw fabric is clean."
