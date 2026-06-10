#!/usr/bin/env bash
# Round-6: (A) treehill@80 remaining single-lever suspects at 15k (fresh, noiselight);
#          (B) FULL 30k validation of the adopted room nview3 gate via the patched run.sh
#              (room@20/40/80/100; @60 already confirmed 16.634). Two parallel run.sh
#              instances with disjoint angle sets = 2-wide on GPU 1 (~11GB each).
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
DIAG_OUT=/data1/zhangao/outputs/diag/mipnerf360
LOGDIR=/data1/zhangao/outputs/probe_r3fast_logs
mkdir -p "$LOGDIR"

P15K="ITERATIONS=15000 DENSIFY_UNTIL_ITER=12500 DENSIFICATION_INTERVAL=100 NN_START_ITER=1250 NN_DERIVATIVES_ITER=1500 NN_UPDATE_INTERVAL=1500"
NOISELIGHT="CPF_NOISE_TANGENT=0 CPF_NOISE_OBS_GATE=0 CPF_MAX_NOISE_NORM=0.0005 CPF_DERIVATIVE_INTERVAL=8 CPF_NOISE_WEIGHT=0.5"

n_train() { pgrep -af 'train_experimental_branch3_mc.py' | grep -v 'conda run' | grep -cv 'grep' || true; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1 || echo 0; }
wait_until() { local mt=$1 mm=$2; while [ "$(n_train)" -ge "$mt" ] || [ "$(gpu_used)" -gt "$mm" ]; do sleep 60; done; }

launch_bg() {  # launch_bg <tag> <extra env...>   (treehill@80, outdoor gates)
    local tag=$1; shift
    local before; before=$(n_train)
    echo "[r6 $(date +%H:%M)] launching $tag (gpu=$(gpu_used)MiB)"
    ( export SCENE=treehill ANGLE=80.0 TAG="$tag" CAP_MAX=3300000 ANTIALIASING=0 CPF_FIELD_OPACITY_REG=0
      for kv in $P15K "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do [ "$(n_train)" -gt "$before" ] && break; sleep 5; done
}

echo "===== Phase A: treehill@80 fresh / noiselight (15k) ====="
wait_until 1 8000
launch_bg th80_fresh15k NN_UPDATE_INTERVAL=500
wait_until 2 12000
launch_bg th80_noiselight15k $NOISELIGHT
wait
echo "---- Phase A results (ref: th80_locked15k=14.921, th80_nv3_15k=14.700) ----"
for t in th80_fresh15k th80_noiselight15k; do
    f="$DIAG_OUT/treehill_angle_80.0_${t}_i15000/metrics.csv"
    echo "== $t"; awk -F, 'NR==2{printf "  SSIM %.3f PSNR %.3f LPIPS %.3f\n", $4,$5,$6}' "$f" 2>/dev/null || echo "  (missing)"
done

echo "===== Phase B: room nview3 gate — FULL 30k validation via run.sh ====="
wait_until 1 4000
( SCENES=room ANGLES="20.0 40.0"  bash run.sh ) > "$LOGDIR/room_val_a.log" 2>&1 &
sleep 90
( SCENES=room ANGLES="80.0 100.0" bash run.sh ) > "$LOGDIR/room_val_b.log" 2>&1 &
wait

echo "================ ROOM 30k VALIDATION (nview3 gate) ================"
for a in 20.0 40.0 80.0 100.0; do
    f="outputs/sweep_results/mipnerf360/room_angle_${a}_i30000_aa/metrics.csv"
    echo "== room@${a}"; awk -F, 'NR==2{printf "  SSIM %.3f PSNR %.3f LPIPS %.3f\n", $4,$5,$6}' "$f" 2>/dev/null || echo "  (missing)"
done
echo "v1 (nview1) refs: room@20=28.824 @40=22.271 @80=15.631 @100=12.587 ; old-best: 29.610/23.242/17.782/15.013"
