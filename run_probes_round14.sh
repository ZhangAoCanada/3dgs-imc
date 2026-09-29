#!/usr/bin/env bash
# Round-14 SELF-DRIVING (server GPU1): Fix B = exogenous MVS prior into the CPF field,
# then A+B composition. Waits for both MVS priors (fused.ply) and a free slot.
#   rm60 nv3 + MVS(merge)   vs anchor 16.565
#   rm100 nv3 + MVS(merge)  vs anchor 12.623
#   if either >= +0.15 -> compose with Fix A (MH) on that cell
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
DIAG_OUT=/data1/zhangao/outputs/diag/mipnerf360
LOGDIR=/data1/zhangao/outputs/probe_r3fast_logs
MVS=/data1/zhangao/repos/3dgs-imc-codex/outputs/mvs_priors
mkdir -p "$LOGDIR"
P15K="ITERATIONS=15000 DENSIFY_UNTIL_ITER=12500 DENSIFICATION_INTERVAL=100 NN_START_ITER=1250 NN_DERIVATIVES_ITER=1500 NN_UPDATE_INTERVAL=1500 METRICS_SKIP_NOREF=1"
MH="MH_RELOC=1 MH_INTERVAL=300 MH_START_ITER=3000 MH_BATCH=512"

n_train() { pgrep -af 'train_experimental_branch3_mc.py' | grep -v 'conda run' | grep -cv 'grep' || true; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1 || echo 0; }
ram_avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
wait_until() { local mt=$1 mm=$2; while [ "$(n_train)" -ge "$mt" ] || [ "$(gpu_used)" -gt "$mm" ] || [ "$(ram_avail_gb)" -lt 20 ]; do sleep 60; done; }
psnr_of() { awk -F, 'NR==2{printf "%.3f", $5}' "$DIAG_OUT/${1}_angle_${2}_${3}_i15000/metrics.csv" 2>/dev/null; }
ge() { awk "BEGIN{exit !($1 >= $2)}"; }

launch() { # launch <scene> <angle> <tag> <extra...>
    local sc=$1 ang=$2 tag=$3; shift 3
    [ -f "$DIAG_OUT/${sc}_angle_${ang}_${tag}_i15000/metrics.csv" ] && { echo "[r14] skip $tag"; return 0; }
    wait_until 2 14000
    local before; before=$(n_train)
    echo "[r14 $(date +%H:%M)] launching $tag (gpu=$(gpu_used)MiB)"
    ( export SCENE=$sc ANGLE=$ang TAG="$tag" CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1
      for kv in $P15K "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do [ "$(n_train)" -gt "$before" ] && break; sleep 5; done
}

echo "[r14] waiting for MVS priors..."
while [ ! -f "$MVS/room_angle_60.0/fused.ply" ] || [ ! -f "$MVS/room_angle_100.0/fused.ply" ]; do sleep 120; done
# also wait for round-13 to finish so server isn't 3-wide
while pgrep -f run_probes_round13.sh >/dev/null 2>&1; do sleep 120; done
echo "[r14] priors ready, round-13 done — starting Fix B"

launch room 60.0  rm60_nv3_mvs15k  RANDOM_VIEW_NUM=3 CPF_MVS_PRIOR_PATH="$MVS/room_angle_60.0/fused.ply"  CPF_MVS_MODE=merge
launch room 100.0 rm100_nv3_mvs15k RANDOM_VIEW_NUM=3 CPF_MVS_PRIOR_PATH="$MVS/room_angle_100.0/fused.ply" CPF_MVS_MODE=merge
wait
B60=$(psnr_of room 60.0 rm60_nv3_mvs15k); B100=$(psnr_of room 100.0 rm100_nv3_mvs15k)
echo "[r14] Fix B: rm60 nv3+MVS=$B60 (anchor 16.565) | rm100 nv3+MVS=$B100 (anchor 12.623)"

# compose A+B where B showed signal
if [ -n "$B100" ] && ge "$B100" "12.773"; then
    launch room 100.0 rm100_nv3_mvs_mh15k RANDOM_VIEW_NUM=3 CPF_MVS_PRIOR_PATH="$MVS/room_angle_100.0/fused.ply" CPF_MVS_MODE=merge $MH
fi
if [ -n "$B60" ] && ge "$B60" "16.715"; then
    launch room 60.0 rm60_nv3_mvs_mh15k RANDOM_VIEW_NUM=3 CPF_MVS_PRIOR_PATH="$MVS/room_angle_60.0/fused.ply" CPF_MVS_MODE=merge $MH
fi
wait
echo "================ ROUND-14 (Fix B + A+B) ================"
echo "rm60  nv3+MVS    : $B60  (anchor 16.565)"
echo "rm100 nv3+MVS    : $B100 (anchor 12.623)"
echo "rm100 nv3+MVS+MH : $(psnr_of room 100.0 rm100_nv3_mvs_mh15k)"
echo "rm60  nv3+MVS+MH : $(psnr_of room 60.0 rm60_nv3_mvs_mh15k)"
echo "[r14] done"
