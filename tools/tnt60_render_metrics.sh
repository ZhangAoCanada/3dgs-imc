#!/usr/bin/env bash
# Re-render T&T@60 (truck_resize + train_resize) from the SAVED best checkpoints
# (develop3_multi_angle2, iter_30000) and compute all 6 metrics with the current
# metrics.py (no-ref ON). Runs on hust2 GPU 2. AA toggled via env AA (1=--antialiasing).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
unset METRICS_SKIP_NOREF                 # ensure BRISQUE/NIQE/PIQE are computed
C=/opt/anaconda3/bin/conda
ENV=3dgs-imc
REPO=/data1/ZHANGAo_2024/3dgs-imc-codex
CKPT=/data1/zhangao/outputs/tnt60_ckpt
AA="${AA:-1}"; AA_ARG=(); [ "$AA" = "1" ] && AA_ARG=(--antialiasing)
LOG="$CKPT/_render_metrics_aa${AA}.log"; : >"$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
cd "$REPO" || { say "repo missing"; exit 1; }
say "===== T&T@60 re-render+metrics from checkpoints (AA=$AA) GPU2 ====="
for sc in truck_resize train_resize; do
  mp="$CKPT/$sc"
  [ -f "$mp/point_cloud/iteration_30000/point_cloud.ply" ] || { say "SKIP $sc (no ply)"; continue; }
  say "RENDER $sc"
  rm -rf "$mp/test"
  $C run --no-capture-output -n "$ENV" python render.py \
      --model_path "$mp" --iteration 30000 --skip_train "${AA_ARG[@]}" --quiet >>"$LOG" 2>&1 \
      && say "  renders=$(ls "$mp"/test/ours_30000/renders/*.png 2>/dev/null|wc -l) gt=$(ls "$mp"/test/ours_30000/gt/*.png 2>/dev/null|wc -l)" \
      || { say "  RENDER FAILED $sc (see log)"; continue; }
  say "METRICS $sc"
  $C run --no-capture-output -n "$ENV" python metrics.py \
      --model_paths "$mp" --metrics_save_path "$mp/metrics.csv" >>"$LOG" 2>&1 \
      && say "  $(tail -1 "$mp/metrics.csv")" || say "  METRICS FAILED $sc (see log)"
done
say "===== DONE (AA=$AA) ====="