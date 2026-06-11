#!/usr/bin/env bash
# Round-8 (15k, 2-wide GPU1): qualify eps0.5 for the room gate across angles + treehill seed-2.
#   room@60: nv3+eps05 vs existing nv3 anchor (16.565)
#   room@80: (nv3, nv3+eps05) pair
#   room@20: (nv3, nv3+eps05) pair  (regression check at well-covered angle)
#   treehill@80: noiselight seed-2 (was +0.31 single-seed; confirm before gating)
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

launch_bg() {  # launch_bg <scene> <angle> <cap> <tag> <extra...>  (indoor gates AA=1 floater=0.1)
    local sc=$1 ang=$2 cap=$3 tag=$4; shift 4
    local before; before=$(n_train)
    echo "[r8 $(date +%H:%M)] launching $tag (gpu=$(gpu_used)MiB)"
    ( export SCENE=$sc ANGLE=$ang TAG="$tag" CAP_MAX=$cap ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1
      for kv in $P15K "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do [ "$(n_train)" -gt "$before" ] && break; sleep 5; done
}

wait_until 1 8000
launch_bg room 60.0 1200000 rm60_nv3eps05_15k RANDOM_VIEW_NUM=3 EPS=0.5
wait_until 2 14000
( export SCENE=treehill ANGLE=80.0 TAG=th80_noiselight15k_s2 CAP_MAX=3300000 ANTIALIASING=0 CPF_FIELD_OPACITY_REG=0
  for kv in $P15K $NOISELIGHT; do export "${kv?}"; done
  bash diag_run.sh ) > "$LOGDIR/th80_noiselight15k_s2.log" 2>&1 &
sleep 60
wait

wait_until 1 8000
launch_bg room 80.0 1200000 rm80_nv3_15k RANDOM_VIEW_NUM=3
wait_until 2 14000
launch_bg room 80.0 1200000 rm80_nv3eps05_15k RANDOM_VIEW_NUM=3 EPS=0.5
wait

wait_until 1 8000
launch_bg room 20.0 1200000 rm20_nv3_15k RANDOM_VIEW_NUM=3
wait_until 2 14000
launch_bg room 20.0 1200000 rm20_nv3eps05_15k RANDOM_VIEW_NUM=3 EPS=0.5
wait

echo "================ ROUND-8 RESULTS (15k) ================"
for spec in "room 60.0 rm60_nv3eps05_15k" "treehill 80.0 th80_noiselight15k_s2" \
            "room 80.0 rm80_nv3_15k" "room 80.0 rm80_nv3eps05_15k" \
            "room 20.0 rm20_nv3_15k" "room 20.0 rm20_nv3eps05_15k"; do
    set -- $spec
    f="$DIAG_OUT/${1}_angle_${2}_${3}_i15000/metrics.csv"
    echo "== $3"; awk -F, 'NR==2{printf "  SSIM %.3f PSNR %.3f LPIPS %.3f\n", $4,$5,$6}' "$f" 2>/dev/null || echo "  (missing)"
done
echo "anchors: rm60 nv3=16.565 | th80 locked=14.921 noiselight_s1=15.231 | rm100 nv3=12.623 nv3eps05=13.213"
