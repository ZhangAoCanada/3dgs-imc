#!/usr/bin/env bash
# Round-3 FAST probes — 15k proportionally-compressed protocol, managed 2-wide on GPU 1 ONLY.
#
# Compression (mirrors the 30k run's phase structure, not a truncation):
#   ITERATIONS 30000->15000 (lr schedule auto-compresses: position_lr_max_steps=ITERATIONS)
#   DENSIFY until 25000->12500, interval 200->100  => same ~120 growth steps, REACHES CAP
#   NN_START 2500->1250, NN_DERIVATIVES 3000->1500, NN_UPDATE_INTERVAL 3000->1500 (=
#     same count of field updates); per-iter CPF rates (train 8 / derivative 2) unchanged.
#   Freshness probes scale the same way: nnup1000@30k == NN_UPDATE_INTERVAL=500@15k.
# Fidelity anchors (15k probe has a same-protocol 30k reference on this machine):
#   locked15k    <-> locked@30k 15.27 (v1) ;  p2anchor15k <-> 15.35 (round-1 P2)
#   fresh15k     <-> nnup1000@30k 15.88 (R2a) ; freshhi roughly <-> nnup500_rv32@30k (R2b)
# RULE (user): any 15k winner is only "progress" after a FULL 30k confirmation -> auto-launched.
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
DIAG_OUT=/data1/zhangao/outputs/diag/mipnerf360
LOGDIR=/data1/zhangao/outputs/probe_r3fast_logs
mkdir -p "$LOGDIR"

P15K="ITERATIONS=15000 DENSIFY_UNTIL_ITER=12500 DENSIFICATION_INTERVAL=100 NN_START_ITER=1250 NN_DERIVATIVES_ITER=1500 NN_UPDATE_INTERVAL=1500"
NOISELIGHT="CPF_NOISE_TANGENT=0 CPF_NOISE_OBS_GATE=0 CPF_MAX_NOISE_NORM=0.0005 CPF_DERIVATIVE_INTERVAL=8 CPF_NOISE_WEIGHT=0.5"

n_train() { pgrep -af 'train_experimental_branch3_mc.py' | grep -v 'conda run' | grep -cv 'grep' || true; }   # real python procs only (conda wrapper also matches the module name)
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1 || echo 0; }

wait_for_slot() {  # max 2 trainings AND enough free memory for a new ~13GB run
    while [ "$(n_train)" -ge 2 ] || [ "$(gpu_used)" -gt 17000 ]; do sleep 60; done
}

launch() {  # launch <tag> <extra env...>
    local tag=$1; shift
    wait_for_slot
    local before; before=$(n_train)
    echo "[r3fast $(date +%H:%M)] launching $tag (trainings=$before, gpu=$(gpu_used)MiB)"
    ( export SCENE=room ANGLE=60.0 TAG="$tag" CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1
      for kv in $P15K "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do   # wait until this run's python is visible so slot accounting stays truthful
        [ "$(n_train)" -gt "$before" ] && break
        sleep 5
    done
}

# hypothesis probes + anchors; longest job first so it overlaps the most
launch nview3_15k      RANDOM_VIEW_NUM=3
launch locked15k       ""
launch noiselight15k   $NOISELIGHT
launch fresh15k        NN_UPDATE_INTERVAL=500
launch combo15k        NN_UPDATE_INTERVAL=500 $NOISELIGHT
launch eps05_15k       EPS=0.5
launch p2anchor15k     CPF_FIELD_OPACITY_REG=0
wait

echo "================ ROUND-3 FAST RESULTS (15k, room@60) ================"
get_psnr() { awk -F, 'NR==2{printf "%.3f", $5}' "$DIAG_OUT/room_angle_60.0_${1}_i15000/metrics.csv" 2>/dev/null; }
for t in locked15k noiselight15k nview3_15k fresh15k combo15k eps05_15k p2anchor15k; do
    echo "  $t: PSNR=$(get_psnr $t)"
done
echo "  30k anchors: locked=15.27(v1) p2=15.35 nnup1000=15.88(R2a) (see also R2b metrics)"

# ---- auto 30k confirmation of the best hypothesis probe (user rule) ----
ref=$(get_psnr locked15k)
best_tag=""; best_val=""
for t in noiselight15k nview3_15k fresh15k combo15k eps05_15k; do
    v=$(get_psnr $t); [ -n "$v" ] || continue
    if [ -n "$ref" ] && awk "BEGIN{exit !($v > $ref + 0.3)}"; then
        if [ -z "$best_val" ] || awk "BEGIN{exit !($v > $best_val)}"; then best_tag=$t; best_val=$v; fi
    fi
done
declare -A CONFIRM_ENV=(
    [noiselight15k]="$NOISELIGHT"
    [nview3_15k]="RANDOM_VIEW_NUM=3"
    [fresh15k]="NN_UPDATE_INTERVAL=1000"
    [combo15k]="NN_UPDATE_INTERVAL=1000 $NOISELIGHT"
    [eps05_15k]="EPS=0.5"
)
if [ -n "$best_tag" ]; then
    echo "[r3fast] WINNER=$best_tag (${best_val} vs ref ${ref}) -> FULL 30k confirmation"
    wait_for_slot
    ( export SCENE=room ANGLE=60.0 TAG="${best_tag%15k}confirm" CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1
      for kv in ${CONFIRM_ENV[$best_tag]}; do export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/${best_tag}_confirm30k.log" 2>&1
    echo "== ${best_tag%15k}confirm (30k)"
    cat "$DIAG_OUT/room_angle_60.0_${best_tag%15k}confirm_i30000/metrics.csv" 2>/dev/null || echo "  (missing)"
else
    echo "[r3fast] no probe beat locked15k by >0.3 — no 30k confirm"
fi
