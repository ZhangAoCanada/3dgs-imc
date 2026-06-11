#!/usr/bin/env bash
# Round-9 (15k, 2-wide GPU1): CPF-GUIDED DENSIFICATION (existing gated feature, never validated).
# Mechanism: relocation/birth targets scored by field-support x Gaussian-sparsity -> adds capacity
# exactly where the gap region is under-filled (inverse of floater pruning; counters budget drain).
# Window scaled to 15k protocol: 3000-12000@30k -> 1500-6000@15k.
#   rm60_nv3_cpfdens  vs nv3 anchor 16.565
#   rm100_nv3_cpfdens vs nv3 anchor 12.623
#   bo100_cpfdens     vs locked anchor 16.980
#   th80_cpfdens      vs locked anchor 14.921
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
DIAG_OUT=/data1/zhangao/outputs/diag/mipnerf360
LOGDIR=/data1/zhangao/outputs/probe_r3fast_logs
mkdir -p "$LOGDIR"

P15K="ITERATIONS=15000 DENSIFY_UNTIL_ITER=12500 DENSIFICATION_INTERVAL=100 NN_START_ITER=1250 NN_DERIVATIVES_ITER=1500 NN_UPDATE_INTERVAL=1500"
CPFD="CPF_DENSIFY=1 CPF_DENSIFY_START_ITER=1500 CPF_DENSIFY_UNTIL_ITER=6000"

n_train() { pgrep -af 'train_experimental_branch3_mc.py' | grep -v 'conda run' | grep -cv 'grep' || true; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1 || echo 0; }
wait_until() { local mt=$1 mm=$2; while [ "$(n_train)" -ge "$mt" ] || [ "$(gpu_used)" -gt "$mm" ]; do sleep 60; done; }

launch_bg() {  # launch_bg <scene> <angle> <cap> <aa> <floater> <tag> <extra...>
    local sc=$1 ang=$2 cap=$3 aa=$4 fl=$5 tag=$6; shift 6
    local before; before=$(n_train)
    echo "[r9 $(date +%H:%M)] launching $tag (gpu=$(gpu_used)MiB)"
    ( export SCENE=$sc ANGLE=$ang TAG="$tag" CAP_MAX=$cap ANTIALIASING=$aa CPF_FIELD_OPACITY_REG=$fl
      for kv in $P15K "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do [ "$(n_train)" -gt "$before" ] && break; sleep 5; done
}

wait_until 1 8000
launch_bg room 60.0 1200000 1 0.1 rm60_nv3_cpfdens15k RANDOM_VIEW_NUM=3 $CPFD
wait_until 2 14000
launch_bg room 100.0 1200000 1 0.1 rm100_nv3_cpfdens15k RANDOM_VIEW_NUM=3 $CPFD
wait

wait_until 1 8000
launch_bg bonsai 100.0 1000000 1 0.1 bo100_cpfdens15k $CPFD
wait_until 2 14000
launch_bg treehill 80.0 3300000 0 0 th80_cpfdens15k $CPFD
wait

echo "================ ROUND-9 RESULTS (15k) ================"
for spec in "room 60.0 rm60_nv3_cpfdens15k 16.565" "room 100.0 rm100_nv3_cpfdens15k 12.623" \
            "bonsai 100.0 bo100_cpfdens15k 16.980" "treehill 80.0 th80_cpfdens15k 14.921"; do
    set -- $spec
    f="$DIAG_OUT/${1}_angle_${2}_${3}_i15000/metrics.csv"
    echo "== $3 (anchor $4)"; awk -F, 'NR==2{printf "  SSIM %.3f PSNR %.3f LPIPS %.3f\n", $4,$5,$6}' "$f" 2>/dev/null || echo "  (missing)"
done
