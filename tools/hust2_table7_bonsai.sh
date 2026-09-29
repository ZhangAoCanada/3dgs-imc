#!/usr/bin/env bash
# Table-7 training-time on bonsai@60 (Mip-NeRF 360 coverage-gap), hust2 GPU 0, ALL methods on the
# SAME GPU (fixes the 4090-vs-5090 mismatch). Each method times SOLO (alone-guard) for accuracy.
# Baselines: external wall-clock of train(+eval), 30k iters, -i images_4 (angular FoV => 1/4-res OK).
# Ours (GS-LSS): bonsai@60 FAST/Eff via run.sh, internal training timer (training_time.txt).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
LOCK=/tmp/t7_bonsai.lock; exec 9>"$LOCK"; flock -n 9 || { echo "[t7] another instance holds lock -> exit"; exit 0; }
C=/opt/anaconda3/bin/conda
SRC=/data1/zhangao/DATA/mipnerf360/bonsai
OUT=/data1/zhangao/outputs/t7_bonsai; mkdir -p "$OUT"
LOG="$OUT/_t7_bonsai.log"
FREE_NEED=${FREE_NEED:-22000}    # GPU0 essentially idle (>=22GB of 24GB) => solo timing
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
have_env(){ $C env list 2>/dev/null | grep -q "^$1 "; }
wait_alone(){ while :; do
  local f; f=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|tr -d ' '||echo 0)
  [ "${f:-0}" -ge "$FREE_NEED" ] && { say "GPU0 ${f}MiB free -> SOLO"; return; }
  say "GPU0 ${f}MiB<${FREE_NEED}; wait 60s"; sleep 60; done; }
# run name env repo entry viewer extra...   (adds -i images_4 + unique --port; external wall-clock)
run(){ local name=$1 env=$2 repo=$3 entry=$4 viewer=$5; shift 5
  local mp="$OUT/$name"
  [ -f "$mp/wall_seconds.txt" ] && { say "SKIP $name ($(cat "$mp/wall_seconds.txt")s)"; return; }
  have_env "$env" || { say "SKIP $name (env '$env' missing)"; return; }
  # watchdog fixes: (1) 3dgs-mcmc's train.py REJECTS --port -> omit it for mcmc; (2) retry once on
  # any failure so a single crash never leaves a method un-timed with the GPU sitting idle.
  local portarg="--port $(( 6400 + RANDOM % 600 ))"; [ "$env" = "mcmc" ] && portarg=""
  local attempt
  for attempt in 1 2; do
    wait_alone; say "RUN(solo) $name env=$env attempt=$attempt"
    rm -rf "$mp"; mkdir -p "$mp"; local t0=$(date +%s)
    ( cd "$repo" && $C run --no-capture-output -n "$env" python "$entry" -s "$SRC" -m "$mp" -i images_4 $viewer $portarg "$@" ) >>"$LOG" 2>&1
    local rc=$? secs=$(( $(date +%s) - t0 ))
    if [ "$rc" = 0 ]; then echo "$secs">"$mp/wall_seconds.txt"; say "DONE $name ${secs}s ($((secs/60))min)"; return; fi
    say "FAIL $name rc=$rc (attempt $attempt)"
  done
  say "GAVE UP $name after 2 attempts (needs manual re-run)"; }
mvgs_run(){ local mp="$OUT/mvgs"; [ -f "$mp/wall_seconds.txt" ] && { say "SKIP mvgs"; return; }
  have_env mvgs || { say "SKIP mvgs (env)"; return; }
  local mv; for mv in 3 2 1; do wait_alone; rm -rf "$mp"; mkdir -p "$mp"; say "RUN(solo) mvgs --mv $mv"
    local t0=$(date +%s) port=$(( 6400 + RANDOM % 600 ))
    ( cd /data1/zhangao/repos/MVGS && $C run --no-capture-output -n mvgs python train.py -s "$SRC" -m "$mp" -i images_4 --port "$port" --eval --mv "$mv" ) >>"$LOG" 2>&1
    local rc=$? secs=$(( $(date +%s) - t0 ))
    if [ "$rc" = 0 ]; then echo "$secs">"$mp/wall_seconds.txt"; echo "$mv">"$mp/mv_used.txt"; say "DONE mvgs mv=$mv ${secs}s ($((secs/60))min)"; return; fi
    say "mvgs mv=$mv FAIL rc=$rc -> lower"; done; }
ours_run(){ local mp="$OUT/ours"; [ -f "$mp/wall_seconds.txt" ] && { say "SKIP ours"; return; }
  wait_alone; say "RUN(solo) ours GS-LSS bonsai@60 FAST"
  ( cd /data1/ZHANGAo_2024/3dgs-imc-codex && CUDA_VISIBLE_DEVICES=0 SCENES=bonsai ANGLES=60.0 DATASETS=mipnerf360 BIASES=angle \
      OUT_ROOT="$OUT/ours_root" DATA_ROOT=/data1/zhangao/DATA CONDA_EXE=$C ENV_NAME=3dgs-imc RUN_EVAL=1 bash run.sh ) >>"$LOG" 2>&1
  local tdir="$OUT/ours_root/mipnerf360/bonsai_angle_60.0_i30000_aa"
  local t=$(grep -oE '[0-9.]+seconds' "$tdir/training_time.txt" 2>/dev/null | tr -d 'seconds')
  mkdir -p "$mp"; [ -n "$t" ] && { echo "$t">"$mp/wall_seconds.txt"; say "DONE ours ${t}s ($(awk "BEGIN{printf \"%.1f\",${t}/60}")min)"; } || say "ours: no training_time (check log)"; }
say "===== Table-7 bonsai@60 timing (hust2 GPU0, solo) START ====="
run 3dgs     gaussian_splatting /data1/zhangao/repos/gaussian_splatting train.py "--disable_viewer" --eval
run hier     gaussian_splatting /data1/zhangao/repos/gaussian_splatting train.py "--disable_viewer" --eval -d dav2_4
run mh       mh                 /data1/zhangao/repos/MH-3DGS    train.py "--disable_viewer" --eval
run mcmc     mcmc               /data1/zhangao/repos/3dgs-mcmc  train.py ""                --eval --cap_max 1000000
run scaffold scaffold           /data1/zhangao/repos/Scaffold-GS train.py ""                --eval
run pgsr     pgsr               /data1/ZHANGAo_2024/PGSR        train.py ""                --eval
mvgs_run
ours_run
say "===== Table-7 bonsai timing DONE ====="
