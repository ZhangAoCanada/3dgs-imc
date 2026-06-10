#!/usr/bin/env bash
# Round-4: stack/efficiency probes after the nview3 breakthrough (room@60, 15k protocol).
#   nview3fresh15k : nview3 + field refresh 1000@30k-equiv (both consolidation levers stacked)
#   nview2_15k     : 2 views — is most of the nview3 gain available at ~1.7x cost instead of 2.5x?
# Waits for the round-3 driver's 30k confirm to START (driver makes no further launches then),
# so adding one training at a time stays within the 2-wide / memory budget on GPU 1.
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
DIAG_OUT=/data1/zhangao/outputs/diag/mipnerf360
LOGDIR=/data1/zhangao/outputs/probe_r3fast_logs
mkdir -p "$LOGDIR"

P15K="ITERATIONS=15000 DENSIFY_UNTIL_ITER=12500 DENSIFICATION_INTERVAL=100 NN_START_ITER=1250 NN_DERIVATIVES_ITER=1500 NN_UPDATE_INTERVAL=1500"

n_train() { pgrep -af 'train_experimental_branch3_mc.py' | grep -v 'conda run' | grep -cv 'grep' || true; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1 || echo 0; }
wait_for_slot() { while [ "$(n_train)" -ge 2 ] || [ "$(gpu_used)" -gt 17000 ]; do sleep 60; done; }

# wait for the auto-confirm run to start (any *confirm* model dir), or all r3fast probes done + driver gone
echo "[r4] waiting for round-3 confirm phase..."
while true; do
    ls -d "$DIAG_OUT"/room_angle_60.0_*confirm_i30000 >/dev/null 2>&1 && break
    pgrep -f run_probes_round3_fast.sh >/dev/null 2>&1 || break   # driver exited (no confirm triggered)
    sleep 120
done
echo "[r4] confirm phase detected (or driver done) — starting round-4 probes"

run_probe() {  # run_probe <tag> <extra env...> (foreground, one at a time)
    local tag=$1; shift
    wait_for_slot
    echo "[r4 $(date +%H:%M)] running $tag"
    ( export SCENE=room ANGLE=60.0 TAG="$tag" CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1
      for kv in $P15K "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1
}

run_probe nview3fresh15k RANDOM_VIEW_NUM=3 NN_UPDATE_INTERVAL=500
run_probe nview2_15k     RANDOM_VIEW_NUM=2

echo "================ ROUND-4 RESULTS (15k, room@60) ================"
for t in nview3fresh15k nview2_15k; do
    f="$DIAG_OUT/room_angle_60.0_${t}_i15000/metrics.csv"
    echo "== $t"; awk -F, 'NR==2{printf "  SSIM %.3f PSNR %.3f LPIPS %.3f\n", $4,$5,$6}' "$f" 2>/dev/null || echo "  (missing)"
done
echo "  ref: nview3_15k=16.565, locked15k=(see r3fast log)"
