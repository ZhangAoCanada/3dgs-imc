#!/usr/bin/env bash
# Round-15 (server GPU1): does Fix A (MH-relocation) GENERALIZE beyond room@100?
# The decisive test — probe MH on the other MH-dominated severe cells + confirm room@100 at 30k.
#   30k: room@100 nv3+MH+dens   (user-rule confirm; dual-ckpt final cell)  anchor 12.96(30k)
#   15k: kitchen@100 +MH        vs locked 21.149
#   15k: treehill@80 +MH        vs locked 14.921
#   15k: bonsai@100  +MH        vs locked 16.980  (variance cell — interpret w/ caution)
#   15k: room@80     nv3+MH     vs nv3   16.153
# MH on locked-config cells (kitchen/bonsai/treehill are NOT nview3-gated).
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
psnr_at() { awk -F, 'NR==2{printf "%.3f", $5}' "$DIAG_OUT/${1}_angle_${2}_${3}_i${4}/metrics.csv" 2>/dev/null; }

launch() { # launch <scene> <angle> <cap> <floater> <iters> <tag> <extra...>
    local sc=$1 ang=$2 cap=$3 fl=$4 it=$5 tag=$6; shift 6
    [ -f "$DIAG_OUT/${sc}_angle_${ang}_${tag}_i${it}/metrics.csv" ] && { echo "[r15] skip $tag"; return 0; }
    local proto="$P15K"; [ "$it" = "30000" ] && proto="ITERATIONS=30000 DENSIFY_UNTIL_ITER=25000 DENSIFICATION_INTERVAL=200 NN_START_ITER=2500 NN_DERIVATIVES_ITER=3000 NN_UPDATE_INTERVAL=3000 SAVE_ITERATIONS=\"15000 30000\""
    wait_until 2 14000
    local before; before=$(n_train)
    echo "[r15 $(date +%H:%M)] launching $tag (it=$it, gpu=$(gpu_used)MiB)"
    ( export SCENE=$sc ANGLE=$ang TAG="$tag" CAP_MAX=$cap ANTIALIASING=1 CPF_FIELD_OPACITY_REG=$fl
      for kv in $proto "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do [ "$(n_train)" -gt "$before" ] && break; sleep 5; done
}

# generality probes first (cheap 15k), 30k confirm last
launch kitchen  100.0 1500000 0.1 15000 ki100_mh15k       $MH
launch treehill 80.0  3300000 0   15000 th80_mh15k        $MH
wait
launch bonsai   100.0 1000000 0.1 15000 bo100_mh15k       $MH
launch room     80.0  1200000 0.1 15000 rm80_nv3_mh15k    RANDOM_VIEW_NUM=3 $MH
wait
launch room     100.0 1200000 0.1 30000 rm100_nv3_mhdens_confirm RANDOM_VIEW_NUM=3 $MH CPF_DENSIFY=1 CPF_DENSIFY_START_ITER=6000 CPF_DENSIFY_UNTIL_ITER=20000
wait

echo "================ ROUND-15: does Fix A generalize? ================"
echo "kitchen@100 +MH : $(psnr_at kitchen 100.0 ki100_mh15k 15000)   (locked 21.149)"
echo "treehill@80 +MH : $(psnr_at treehill 80.0 th80_mh15k 15000)    (locked 14.921)"
echo "bonsai@100  +MH : $(psnr_at bonsai 100.0 bo100_mh15k 15000)    (locked 16.980, variance)"
echo "room@80 nv3+MH  : $(psnr_at room 80.0 rm80_nv3_mh15k 15000)    (nv3 16.153)"
echo "room@100 confirm: 15k=$(psnr_at room 100.0 rm100_nv3_mhdens_confirm 15000) 30k=$(psnr_at room 100.0 rm100_nv3_mhdens_confirm 30000) (anchor 30k 12.96)"
echo "[r15] done"
