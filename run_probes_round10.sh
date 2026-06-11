#!/usr/bin/env bash
# Round-10 SELF-DRIVING: appearance-consensus supervision (Phase 1 of the sequence).
# Funnel: 7.5k existence screen -> 15k ranking/attribution -> regression pair. Decisions
# are made by the driver; idempotent (skips any probe whose metrics already exist).
# New gated knobs: CPF_FIELD_RGB_LOSS (train field rgb head; drafted weight 1e3),
#                  CPF_FIELD_COLOR_REG (obs-gated DC-color <- field rgb consensus).
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
DIAG_OUT=/data1/zhangao/outputs/diag/mipnerf360
LOGDIR=/data1/zhangao/outputs/probe_r3fast_logs
mkdir -p "$LOGDIR"

P15K="ITERATIONS=15000 DENSIFY_UNTIL_ITER=12500 DENSIFICATION_INTERVAL=100 NN_START_ITER=1250 NN_DERIVATIVES_ITER=1500 NN_UPDATE_INTERVAL=1500 METRICS_SKIP_NOREF=1"
P75K="ITERATIONS=7500 DENSIFY_UNTIL_ITER=6250 DENSIFICATION_INTERVAL=50 NN_START_ITER=625 NN_DERIVATIVES_ITER=750 NN_UPDATE_INTERVAL=750 METRICS_SKIP_NOREF=1"
RGB="CPF_FIELD_RGB_LOSS=1e3"

n_train() { pgrep -af 'train_experimental_branch3_mc.py' | grep -v 'conda run' | grep -cv 'grep' || true; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1 || echo 0; }
ram_avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
wait_until() { local mt=$1 mm=$2; while [ "$(n_train)" -ge "$mt" ] || [ "$(gpu_used)" -gt "$mm" ] || [ "$(ram_avail_gb)" -lt 20 ]; do sleep 60; done; }

psnr_of() { # psnr_of <scene> <angle> <tag> <iters>
    awk -F, 'NR==2{printf "%.3f", $5}' "$DIAG_OUT/${1}_angle_${2}_${3}_i${4}/metrics.csv" 2>/dev/null
}

launch() { # launch <scene> <angle> <cap> <aa> <floater> <iters:15000|7500> <tag> <extra env...>
    local sc=$1 ang=$2 cap=$3 aa=$4 fl=$5 it=$6 tag=$7; shift 7
    if [ -f "$DIAG_OUT/${sc}_angle_${ang}_${tag}_i${it}/metrics.csv" ]; then
        echo "[r10] skip $tag (metrics exist)"; return 0
    fi
    local proto="$P15K"; [ "$it" = "7500" ] && proto="$P75K"
    local maxmem=14000; [ "$cap" -ge 4000000 ] && maxmem=4000   # big caps run alone
    local maxtr=2; [ "$cap" -ge 4000000 ] && maxtr=1
    wait_until "$maxtr" "$maxmem"
    local before; before=$(n_train)
    echo "[r10 $(date +%H:%M)] launching $tag (it=$it, gpu=$(gpu_used)MiB)"
    ( export SCENE=$sc ANGLE=$ang TAG="$tag" CAP_MAX=$cap ANTIALIASING=$aa CPF_FIELD_OPACITY_REG=$fl
      for kv in $proto "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do [ "$(n_train)" -gt "$before" ] && break; sleep 5; done
}

ge() { awk "BEGIN{exit !($1 >= $2)}"; }   # float >=

echo "===== Stage S: 7.5k existence screen (room@100) ====="
launch room 100.0 1200000 1 0.1 7500 rm100_nv3_p75 RANDOM_VIEW_NUM=3
launch room 100.0 1200000 1 0.1 7500 rm100_nv3_rgbcr02_p75 RANDOM_VIEW_NUM=3 $RGB CPF_FIELD_COLOR_REG=0.2
wait
A=$(psnr_of room 100.0 rm100_nv3_p75 7500); B=$(psnr_of room 100.0 rm100_nv3_rgbcr02_p75 7500)
echo "[r10] screen: anchor=$A cand(rgb+cr0.2)=$B"
if [ -z "$A" ] || [ -z "$B" ]; then echo "[r10] SCREEN FAILED (missing metrics)"; exit 1; fi
if ! ge "$B" "$(awk "BEGIN{print $A+0.15}")"; then
    echo "[r10] no signal at cr=0.2 -> trying cr=0.05"
    launch room 100.0 1200000 1 0.1 7500 rm100_nv3_rgbcr005_p75 RANDOM_VIEW_NUM=3 $RGB CPF_FIELD_COLOR_REG=0.05
    wait
    C=$(psnr_of room 100.0 rm100_nv3_rgbcr005_p75 7500)
    echo "[r10] screen cr=0.05: $C"
    if [ -z "$C" ] || ! ge "$C" "$(awk "BEGIN{print $A+0.15}")"; then
        echo "[r10] VERDICT: NO-SIGNAL at 7.5k for color consensus -> stopping (Phase 2 decision manual)"
        exit 0
    fi
fi

echo "===== Stage A: 15k ranking + attribution (room@100) ====="
launch room 100.0 1200000 1 0.1 15000 rm100_nv3_rgbonly15k RANDOM_VIEW_NUM=3 $RGB CPF_FIELD_COLOR_REG=0
launch room 100.0 1200000 1 0.1 15000 rm100_nv3_rgbcr005_15k RANDOM_VIEW_NUM=3 $RGB CPF_FIELD_COLOR_REG=0.05
wait
launch room 100.0 1200000 1 0.1 15000 rm100_nv3_rgbcr02_15k RANDOM_VIEW_NUM=3 $RGB CPF_FIELD_COLOR_REG=0.2
wait
R0=$(psnr_of room 100.0 rm100_nv3_rgbonly15k 15000)
R5=$(psnr_of room 100.0 rm100_nv3_rgbcr005_15k 15000)
R2=$(psnr_of room 100.0 rm100_nv3_rgbcr02_15k 15000)
ANCHOR=12.623
echo "[r10] 15k rm100: anchor=$ANCHOR rgbonly=$R0 cr005=$R5 cr02=$R2"
BESTW="0.2"; BESTV=${R2:-0}
if [ -n "$R5" ] && ge "$R5" "$BESTV"; then BESTW="0.05"; BESTV=$R5; fi
if ! ge "$BESTV" "$(awk "BEGIN{print $ANCHOR+0.3}")"; then
    echo "[r10] VERDICT: color consensus < +0.3 at 15k on rm100 -> NO ADOPT (report only)"
    exit 0
fi
echo "[r10] best color-reg weight: $BESTW ($BESTV) -> extending to other cells"

echo "===== Stage A2: extend best config ====="
launch room 60.0 1200000 1 0.1 15000 rm60_nv3_rgbcr15k RANDOM_VIEW_NUM=3 $RGB CPF_FIELD_COLOR_REG=$BESTW
launch treehill 80.0 3300000 0 0 15000 th80_rgbcr15k $RGB CPF_FIELD_COLOR_REG=$BESTW
wait
launch bonsai 100.0 1000000 1 0.1 15000 bo100_rgbcr15k $RGB CPF_FIELD_COLOR_REG=$BESTW
echo "===== Stage R: regression pair ====="
launch counter 80.0 1000000 1 0.1 15000 ct80_rgbcr15k $RGB CPF_FIELD_COLOR_REG=$BESTW
wait
launch bicycle 20.0 5000000 0 0 15000 bi20_rgbcr15k $RGB CPF_FIELD_COLOR_REG=$BESTW
wait

echo "================ ROUND-10 SUMMARY ================"
echo "rm100: anchor=12.623 rgbonly=$R0 cr005=$R5 cr02=$R2 (bestw=$BESTW)"
echo "rm60:  anchor=16.565 -> $(psnr_of room 60.0 rm60_nv3_rgbcr15k 15000)"
echo "th80:  anchor=14.921 -> $(psnr_of treehill 80.0 th80_rgbcr15k 15000)"
echo "bo100: anchor=16.980 -> $(psnr_of bonsai 100.0 bo100_rgbcr15k 15000)"
echo "ct80:  anchor=20.803 -> $(psnr_of counter 80.0 ct80_rgbcr15k 15000)"
echo "bi20:  anchor=22.661 -> $(psnr_of bicycle 20.0 bi20_rgbcr15k 15000)"
