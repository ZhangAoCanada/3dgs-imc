#!/usr/bin/env bash
# Round-11 SELF-DRIVING (waits for round-10 driver to exit):
#   A. Phase-1 dose closure: CPF_FIELD_COLOR_REG=1.0 at 7.5k (screen weights may be diluted
#      by the obs-gate mean; one strong dose settles whether field-color can matter at all).
#   B. NVS smoke test (4k iters, eval off): mechanics of the new gap pseudo-view loss.
#   C. NVS 7.5k probes (w=0.1, 0.3) on room@100+nview3 vs anchor 12.501;
#      auto-escalate the best to 15k on room@100 + room@60 if >= +0.15.
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

launch() { # launch <scene> <angle> <cap> <aa> <floater> <iters> <tag> <extra env...>
    local sc=$1 ang=$2 cap=$3 aa=$4 fl=$5 it=$6 tag=$7; shift 7
    [ -f "$DIAG_OUT/${sc}_angle_${ang}_${tag}_i${it}/metrics.csv" ] && { echo "[r11] skip $tag"; return 0; }
    local proto="$P15K"; [ "$it" = "7500" ] && proto="$P75K"
    wait_until 2 14000
    local before; before=$(n_train)
    echo "[r11 $(date +%H:%M)] launching $tag (it=$it, gpu=$(gpu_used)MiB)"
    ( export SCENE=$sc ANGLE=$ang TAG="$tag" CAP_MAX=$cap ANTIALIASING=$aa CPF_FIELD_OPACITY_REG=$fl
      for kv in $proto "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do [ "$(n_train)" -gt "$before" ] && break; sleep 5; done
}

# wait for round-10 driver to exit
while pgrep -f run_probes_round10.sh >/dev/null 2>&1; do sleep 60; done
echo "[r11] round-10 done; starting"

echo "===== B: NVS smoke (4k iters, eval off) ====="
wait_until 1 8000
( export SCENE=room ANGLE=100.0 TAG=nvs_smoke CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1
  for kv in $P75K; do export "${kv?}"; done
  export ITERATIONS=4000 RUN_EVAL=0 RANDOM_VIEW_NUM=3 NVS_WEIGHT=0.1 NVS_START_ITER=1000
  bash diag_run.sh ) > "$LOGDIR/nvs_smoke.log" 2>&1
if grep -q "\[NVS\] gap width" "$LOGDIR/nvs_smoke.log" && ! grep -qiE "Traceback|Error" "$LOGDIR/nvs_smoke.log"; then
    echo "[r11] NVS smoke OK: $(grep '\[NVS\]' "$LOGDIR/nvs_smoke.log" | head -1)"
else
    echo "[r11] NVS SMOKE FAILED — see $LOGDIR/nvs_smoke.log (skipping NVS probes)"
    grep -iE "Traceback|Error" "$LOGDIR/nvs_smoke.log" | head -5
    NVS_BROKEN=1
fi

echo "===== A + C: dose probe + NVS 7.5k ====="
launch room 100.0 1200000 1 0.1 7500 rm100_nv3_rgbcr1_p75 RANDOM_VIEW_NUM=3 CPF_FIELD_RGB_LOSS=1e3 CPF_FIELD_COLOR_REG=1.0
if [ -z "${NVS_BROKEN:-}" ]; then
    launch room 100.0 1200000 1 0.1 7500 rm100_nv3_nvsw01_p75 RANDOM_VIEW_NUM=3 NVS_WEIGHT=0.1 NVS_START_ITER=750
    wait
    launch room 100.0 1200000 1 0.1 7500 rm100_nv3_nvsw03_p75 RANDOM_VIEW_NUM=3 NVS_WEIGHT=0.3 NVS_START_ITER=750
fi
wait

ANCHOR=12.501
D=$(psnr_of room 100.0 rm100_nv3_rgbcr1_p75 7500)
N1=$(psnr_of room 100.0 rm100_nv3_nvsw01_p75 7500)
N3=$(psnr_of room 100.0 rm100_nv3_nvsw03_p75 7500)
echo "[r11] 7.5k rm100: anchor=$ANCHOR rgbcr1.0=$D nvs_w0.1=$N1 nvs_w0.3=$N3"

BESTW=""; BESTV=""
if [ -n "$N1" ] && ge "$N1" "$(awk "BEGIN{print $ANCHOR+0.15}")"; then BESTW=0.1; BESTV=$N1; fi
if [ -n "$N3" ] && ge "$N3" "$(awk "BEGIN{print $ANCHOR+0.15}")"; then
    if [ -z "$BESTV" ] || ge "$N3" "$BESTV"; then BESTW=0.3; BESTV=$N3; fi
fi
if [ -n "$BESTW" ]; then
    echo "[r11] NVS signal (w=$BESTW: $BESTV) -> 15k ranking"
    launch room 100.0 1200000 1 0.1 15000 rm100_nv3_nvs15k RANDOM_VIEW_NUM=3 NVS_WEIGHT=$BESTW NVS_START_ITER=1500
    launch room 60.0 1200000 1 0.1 15000 rm60_nv3_nvs15k RANDOM_VIEW_NUM=3 NVS_WEIGHT=$BESTW NVS_START_ITER=1500
    wait
    echo "[r11] 15k: rm100 $(psnr_of room 100.0 rm100_nv3_nvs15k 15000) (anchor 12.623) | rm60 $(psnr_of room 60.0 rm60_nv3_nvs15k 15000) (anchor 16.565)"
else
    echo "[r11] VERDICT: NVS no signal at 7.5k (and rgbcr1.0=$D) — gap appearance mechanisms exhausted at probe level"
fi
echo "[r11] done"
