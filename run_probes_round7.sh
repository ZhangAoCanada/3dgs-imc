#!/usr/bin/env bash
# Round-7 (15k protocol, 2-wide GPU1): does nview3's consolidation win extend to the
# remaining big 100-degree deficit cells?
#   kitchen@100 pair: locked vs nview3        (deficit -0.96 vs MH; never diagnosed)
#   bonsai@100  pair: locked vs nview3        (variance cell: gradient averaging may lift mean AND spread)
#   room@100    pair: nview3 vs nview3+eps0.5 (field-construction x batch lever; all old room runs used eps0.5)
# Pairs run together under identical contention.
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
DIAG_OUT=/data1/zhangao/outputs/diag/mipnerf360
LOGDIR=/data1/zhangao/outputs/probe_r3fast_logs
mkdir -p "$LOGDIR"

P15K="ITERATIONS=15000 DENSIFY_UNTIL_ITER=12500 DENSIFICATION_INTERVAL=100 NN_START_ITER=1250 NN_DERIVATIVES_ITER=1500 NN_UPDATE_INTERVAL=1500"

n_train() { pgrep -af 'train_experimental_branch3_mc.py' | grep -v 'conda run' | grep -cv 'grep' || true; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1 || echo 0; }
wait_until() { local mt=$1 mm=$2; while [ "$(n_train)" -ge "$mt" ] || [ "$(gpu_used)" -gt "$mm" ]; do sleep 60; done; }

launch_bg() {  # launch_bg <scene> <angle> <cap> <aa> <floater> <tag> <extra...>
    local sc=$1 ang=$2 cap=$3 aa=$4 fl=$5 tag=$6; shift 6
    local before; before=$(n_train)
    echo "[r7 $(date +%H:%M)] launching $tag (gpu=$(gpu_used)MiB)"
    ( export SCENE=$sc ANGLE=$ang TAG="$tag" CAP_MAX=$cap ANTIALIASING=$aa CPF_FIELD_OPACITY_REG=$fl
      for kv in $P15K "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do [ "$(n_train)" -gt "$before" ] && break; sleep 5; done
}

wait_until 1 8000
launch_bg kitchen 100.0 1500000 1 0.1 ki100_locked15k ""
wait_until 2 14000
launch_bg kitchen 100.0 1500000 1 0.1 ki100_nv3_15k RANDOM_VIEW_NUM=3
wait

wait_until 1 8000
launch_bg bonsai 100.0 1000000 1 0.1 bo100_locked15k ""
wait_until 2 14000
launch_bg bonsai 100.0 1000000 1 0.1 bo100_nv3_15k RANDOM_VIEW_NUM=3
wait

wait_until 1 8000
launch_bg room 100.0 1200000 1 0.1 rm100_nv3_15k RANDOM_VIEW_NUM=3
wait_until 2 14000
launch_bg room 100.0 1200000 1 0.1 rm100_nv3eps05_15k RANDOM_VIEW_NUM=3 EPS=0.5
wait

echo "================ ROUND-7 RESULTS (15k) ================"
for spec in "kitchen 100.0 ki100_locked15k" "kitchen 100.0 ki100_nv3_15k" \
            "bonsai 100.0 bo100_locked15k" "bonsai 100.0 bo100_nv3_15k" \
            "room 100.0 rm100_nv3_15k" "room 100.0 rm100_nv3eps05_15k"; do
    set -- $spec
    f="$DIAG_OUT/${1}_angle_${2}_${3}_i15000/metrics.csv"
    echo "== $3"; awk -F, 'NR==2{printf "  SSIM %.3f PSNR %.3f LPIPS %.3f\n", $4,$5,$6}' "$f" 2>/dev/null || echo "  (missing)"
done
echo "refs(30k): kitchen@100 v1=21.132 old=21.359 | bonsai@100 v2=16.18 v1=17.42 old=18.05 | room@100 nv3=12.955 old=15.013"
