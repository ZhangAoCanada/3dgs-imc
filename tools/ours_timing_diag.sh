#!/usr/bin/env bash
# Ours-FAST timing diagnostic on server GPU1 (gated; waits until the Table-5/6 SLOW jobs free a slot).
# Run 1: shipped Eff config + TIMING_BREAKDOWN=1  -> where does the 39 min go?
# Run 2: lighter CPF (derivative 2->8, train 8->16) + eval -> faster time + does PSNR hold?
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
export CUDA_VISIBLE_DEVICES=1
# TIMING MUST RUN ALONE (user 2026-06-20): a 2nd co-resident training shares SM+bandwidth and
# inflates the measured time. Require GPU1 essentially idle (>=30GB free of 32GB) AND no other
# train_experimental_branch process anywhere -> the timed run is the only thing on the GPU.
FREE_NEED=${FREE_NEED:-30000}
OUT=/data1/zhangao/outputs/ours_timing_diag; mkdir -p "$OUT"
LOG="$OUT/_diag.log"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
wait_gpu(){ while :; do
  # Free-memory IS the alone-detector: GPU1 is 32GB; any real co-resident training uses >=10GB, so
  # free >= FREE_NEED (30GB) <=> nothing else of substance is on the GPU. No pgrep (self-match-prone).
  local free; free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|tr -d ' '); free=${free:-0}
  if [ "$free" -ge "$FREE_NEED" ]; then say "GPU1 ${free}MiB free (>=${FREE_NEED}) -> launch ALONE"; return; fi
  say "GPU1 only ${free}MiB free (<${FREE_NEED}); co-resident work present, wait 120s"; sleep 120; done; }
harvest(){ local mp="$OUT/tandt/truck_resize_discontinuous_$1_i30000"
  local t=$(grep -oE '[0-9.]+seconds' "$mp/training_time.txt" 2>/dev/null|tr -d 'seconds')
  local p=$(awk -F, '$3=="ours_30000"{print $5}' "$mp/metrics.csv" 2>/dev/null|tail -1)
  say "RESULT $1: time=${t:-NA}s ($(awk "BEGIN{printf \"%.1f\", ${t:-0}/60}")min) PSNR=${p:-NA}"
  [ -f "$mp/timing_breakdown.json" ] && { say "breakdown:"; cat "$mp/timing_breakdown.json" | tee -a "$LOG"; }; }
run_one(){ local tag=$1; shift; wait_gpu; say "RUN $tag : $*"
  ( set -a; SCENE=truck_resize; DATASET=tandt; SPLIT=discontinuous; SLOW=0; ANGLE=0
    CONDA_EXE=/opt/miniconda3/bin/conda; ENV_NAME=3dgs-imc; DATA_ROOT=/data1/zhangao/DATA
    IMAGES=images; DEPTHS=dav2; CAP_MAX=2000000; EPS=0.25; ITER=30000; ITERATIONS=30000
    SAVE_ITERATIONS=30000; OUT_ROOT="$OUT"; eval "$*"; TAG="$tag"; bash diag_run.sh ) >>"$LOG" 2>&1
  harvest "$tag"; }
say "===== ours timing diagnostic START (FREE_NEED=${FREE_NEED}, ALONE-gated) ====="
# Run 0: CLEAN re-time of the shipped Eff/FAST config (NO breakdown instrumentation, NO eval) ->
#        authoritative alone-measured time that replaces the ambiguous 2347.95s.
run_one clean_shipped  'RUN_EVAL=0'
# Run 1: same config + TIMING_BREAKDOWN -> proportions (its TOTAL may be slightly inflated by syncs).
run_one diag_breakdown 'TIMING_BREAKDOWN=1 RUN_EVAL=0'
# Run 2: lighter CPF (derivative 2->8, train 8->16) + eval -> faster time AND does PSNR hold?
run_one diag_lightcpf  'CPF_DERIVATIVE_INTERVAL=8 CPF_TRAIN_INTERVAL=16 TIMING_BREAKDOWN=1 RUN_EVAL=1'
say "===== diagnostic DONE. clean_shipped time = authoritative Table-7 ours number ====="
