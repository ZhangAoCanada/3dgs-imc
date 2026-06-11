#!/usr/bin/env bash
# Round-12: NVS (gap pseudo-view supervision) probes with FIXED geometry
# (anti-resultant gap detection: test-inside 100% on room@100+@60; invdepth fix;
#  content-centroid look-at; dual ring heights). Self-driving with auto-escalation.
#   7.5k screen: rm100+nv3 NVS w=0.1 / w=0.3 vs anchor 12.501
#   escalate best (>= +0.15) to 15k: rm100 (anchor 12.623) + rm60 (anchor 16.565)
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
DIAG_OUT=/data1/zhangao/outputs/diag/mipnerf360
LOGDIR=/data1/zhangao/outputs/probe_r3fast_logs
mkdir -p "$LOGDIR"

P15K="ITERATIONS=15000 DENSIFY_UNTIL_ITER=12500 DENSIFICATION_INTERVAL=100 NN_START_ITER=1250 NN_DERIVATIVES_ITER=1500 NN_UPDATE_INTERVAL=1500 METRICS_SKIP_NOREF=1"
P75K="ITERATIONS=7500 DENSIFY_UNTIL_ITER=6250 DENSIFICATION_INTERVAL=50 NN_START_ITER=625 NN_DERIVATIVES_ITER=750 NN_UPDATE_INTERVAL=750 METRICS_SKIP_NOREF=1"

n_train() { pgrep -af 'train_experimental_branch3_mc.py' | grep -v 'conda run' | grep -cv 'grep' || true; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1 || echo 0; }
ram_avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
wait_until() { local mt=$1 mm=$2; while [ "$(n_train)" -ge "$mt" ] || [ "$(gpu_used)" -gt "$mm" ] || [ "$(ram_avail_gb)" -lt 20 ]; do sleep 60; done; }
psnr_of() { awk -F, 'NR==2{printf "%.3f", $5}' "$DIAG_OUT/${1}_angle_${2}_${3}_i${4}/metrics.csv" 2>/dev/null; }
ge() { awk "BEGIN{exit !($1 >= $2)}"; }

launch() { # launch <scene> <angle> <iters> <tag> <extra env...>
    local sc=$1 ang=$2 it=$3 tag=$4; shift 4
    [ -f "$DIAG_OUT/${sc}_angle_${ang}_${tag}_i${it}/metrics.csv" ] && { echo "[r12] skip $tag"; return 0; }
    local proto="$P15K"; [ "$it" = "7500" ] && proto="$P75K"
    wait_until 2 14000
    local before; before=$(n_train)
    echo "[r12 $(date +%H:%M)] launching $tag (gpu=$(gpu_used)MiB)"
    ( export SCENE=$sc ANGLE=$ang TAG="$tag" CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1
      for kv in $proto "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do [ "$(n_train)" -gt "$before" ] && break; sleep 5; done
}

echo "===== 7.5k NVS screen (rm100+nv3, anchor 12.501) ====="
launch room 100.0 7500 rm100_nv3_nvsA_p75 RANDOM_VIEW_NUM=3 NVS_WEIGHT=0.1 NVS_START_ITER=1500
launch room 100.0 7500 rm100_nv3_nvsB_p75 RANDOM_VIEW_NUM=3 NVS_WEIGHT=0.3 NVS_START_ITER=1500
wait
ANCHOR=12.501
A=$(psnr_of room 100.0 rm100_nv3_nvsA_p75 7500)
B=$(psnr_of room 100.0 rm100_nv3_nvsB_p75 7500)
echo "[r12] 7.5k: anchor=$ANCHOR nvs_w0.1=$A nvs_w0.3=$B"
BESTW=""; BESTV=""
if [ -n "$A" ] && ge "$A" "$(awk "BEGIN{print $ANCHOR+0.15}")"; then BESTW=0.1; BESTV=$A; fi
if [ -n "$B" ] && ge "$B" "$(awk "BEGIN{print $ANCHOR+0.15}")"; then
    if [ -z "$BESTV" ] || ge "$B" "$BESTV"; then BESTW=0.3; BESTV=$B; fi
fi
if [ -z "$BESTW" ]; then
    echo "[r12] VERDICT: NVS no 7.5k signal (w0.1=$A w0.3=$B vs $ANCHOR) — appearance transport closed"
    exit 0
fi
echo "[r12] NVS signal (w=$BESTW: $BESTV) -> 15k"
launch room 100.0 15000 rm100_nv3_nvs15k RANDOM_VIEW_NUM=3 NVS_WEIGHT=$BESTW NVS_START_ITER=3000
launch room 60.0  15000 rm60_nv3_nvs15k  RANDOM_VIEW_NUM=3 NVS_WEIGHT=$BESTW NVS_START_ITER=3000
wait
echo "[r12] 15k: rm100=$(psnr_of room 100.0 rm100_nv3_nvs15k 15000) (anchor 12.623) | rm60=$(psnr_of room 60.0 rm60_nv3_nvs15k 15000) (anchor 16.565)"
echo "[r12] done"
