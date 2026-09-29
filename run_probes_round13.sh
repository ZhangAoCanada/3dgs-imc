#!/usr/bin/env bash
# Round-13 (server GPU1): Fix A = field-proposed, photometrically-verified relocation (MH step).
# Self-driving 15k probes vs validated nview3 anchors.
#   rm100 nv3 + MH               vs anchor 12.623   (hardest cell)
#   rm100 nv3 + MH + cpf_densify vs anchor 12.623   (births also field-guided)
#   rm60  nv3 + MH               vs anchor 16.565   (regression: does MH hurt the fixed cell?)
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
DIAG_OUT=/data1/zhangao/outputs/diag/mipnerf360
LOGDIR=/data1/zhangao/outputs/probe_r3fast_logs
mkdir -p "$LOGDIR"
P15K="ITERATIONS=15000 DENSIFY_UNTIL_ITER=12500 DENSIFICATION_INTERVAL=100 NN_START_ITER=1250 NN_DERIVATIVES_ITER=1500 NN_UPDATE_INTERVAL=1500 METRICS_SKIP_NOREF=1"
MH="MH_RELOC=1 MH_INTERVAL=300 MH_START_ITER=3000 MH_BATCH=512"

n_train() { pgrep -af 'train_experimental_branch3_mc.py' | grep -v 'conda run' | grep -cv 'grep' || true; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1 || echo 0; }
ram_avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
wait_until() { local mt=$1 mm=$2; while [ "$(n_train)" -ge "$mt" ] || [ "$(gpu_used)" -gt "$mm" ] || [ "$(ram_avail_gb)" -lt 20 ]; do sleep 60; done; }
psnr_of() { awk -F, 'NR==2{printf "%.3f", $5}' "$DIAG_OUT/${1}_angle_${2}_${3}_i15000/metrics.csv" 2>/dev/null; }

launch() { # launch <scene> <angle> <tag> <extra...>
    local sc=$1 ang=$2 tag=$3; shift 3
    [ -f "$DIAG_OUT/${sc}_angle_${ang}_${tag}_i15000/metrics.csv" ] && { echo "[r13] skip $tag"; return 0; }
    wait_until 2 14000
    local before; before=$(n_train)
    echo "[r13 $(date +%H:%M)] launching $tag (gpu=$(gpu_used)MiB)"
    ( export SCENE=$sc ANGLE=$ang TAG="$tag" CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1
      for kv in $P15K "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do [ "$(n_train)" -gt "$before" ] && break; sleep 5; done
}

launch room 100.0 rm100_nv3_mh15k       RANDOM_VIEW_NUM=3 $MH
launch room 100.0 rm100_nv3_mhdens15k   RANDOM_VIEW_NUM=3 $MH CPF_DENSIFY=1 CPF_DENSIFY_START_ITER=3000 CPF_DENSIFY_UNTIL_ITER=10000
wait
launch room 60.0  rm60_nv3_mh15k        RANDOM_VIEW_NUM=3 $MH
wait

echo "================ ROUND-13 (Fix A / MH) RESULTS (15k) ================"
echo "rm100 nv3+MH      : $(psnr_of room 100.0 rm100_nv3_mh15k)     (anchor 12.623)"
echo "rm100 nv3+MH+dens : $(psnr_of room 100.0 rm100_nv3_mhdens15k) (anchor 12.623)"
echo "rm60  nv3+MH      : $(psnr_of room 60.0  rm60_nv3_mh15k)      (anchor 16.565)"
echo "[r13] done"
