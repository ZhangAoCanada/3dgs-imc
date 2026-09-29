#!/usr/bin/env bash
# Time each Table-7 baseline on truck_resize (hust2, GPU2 ONLY), external wall-clock. Sequential.
# Skips a method if its env is missing (run again later) or if it's already timed. --eval = standard
# split; training TIME for fixed 30k iters is ~independent of the holdout, so this is representative.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # let MVGS --mv 3 fit in the 4090's 24GB
export WANDB_MODE=disabled   # Scaffold-GS wandb.log()s every iter -> no-op it so the timing is pure training
C=/opt/anaconda3/bin/conda
SRC=/data2/zhangao/tandt_db/tandt/truck_resize
OUT=/data1/zhangao/outputs/baselines_t7_time; mkdir -p "$OUT"
LOG="$OUT/_t7_baselines.log"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
have_env(){ $C env list 2>/dev/null | grep -q "^$1 "; }
run(){  # name env repo entry viewer "extra-args..."   (viewer = "--disable_viewer" or "")
  local name=$1 env=$2 repo=$3 entry=$4 viewer=$5; shift 5
  local mp="$OUT/$name"
  [ -f "$mp/wall_seconds.txt" ] && { say "SKIP $name (already timed: $(cat "$mp/wall_seconds.txt")s)"; return; }
  have_env "$env" || { say "SKIP $name (env '$env' not built yet)"; return; }
  say "RUN $name env=$env repo=$repo viewer='$viewer'"
  mkdir -p "$mp"; local t0=$(date +%s)
  local port=$(( 6200 + RANDOM % 700 ))
  ( cd "$repo" && $C run --no-capture-output -n "$env" python "$entry" -s "$SRC" -m "$mp" $viewer --port "$port" "$@" ) >>"$LOG" 2>&1
  local rc=$? t1=$(date +%s) secs=$(( $(date +%s) - t0 ))
  if [ "$rc" = 0 ]; then echo "$secs" > "$mp/wall_seconds.txt"; say "DONE $name rc=0 time=${secs}s ($((secs/60))min)"
  else say "FAIL $name rc=$rc (see log)"; fi
}
# MVGS: time at its DEFAULT mv=3; on OOM fall back 3->2->1, record the closest-to-default that fits.
mvgs_ladder(){
  local mp="$OUT/mvgs"
  [ -f "$mp/wall_seconds.txt" ] && { say "SKIP mvgs (already timed mv=$(cat "$mp/mv_used.txt" 2>/dev/null) $(cat "$mp/wall_seconds.txt")s)"; return; }
  have_env mvgs || { say "SKIP mvgs (env not built)"; return; }
  local mv
  for mv in 3 2 1; do
    rm -rf "$mp"; mkdir -p "$mp"
    say "RUN mvgs --mv $mv (default-first ladder)"
    local t0=$(date +%s) port=$(( 6200 + RANDOM % 700 ))
    ( cd /data1/zhangao/repos/MVGS && $C run --no-capture-output -n mvgs python train.py -s "$SRC" -m "$mp" --port "$port" --eval --mv "$mv" ) >>"$LOG" 2>&1
    local rc=$? secs=$(( $(date +%s) - t0 ))
    if [ "$rc" = 0 ]; then echo "$secs" > "$mp/wall_seconds.txt"; echo "$mv" > "$mp/mv_used.txt"; say "DONE mvgs --mv $mv time=${secs}s ($((secs/60))min)"; return; fi
    say "mvgs --mv $mv FAILED rc=$rc (likely OOM) -> try lower mv"
  done
  say "mvgs failed at all mv levels"
}
say "===== Table-7 baseline timing (hust2 GPU2) START ====="
run pgsr     pgsr               /data1/ZHANGAo_2024/PGSR        train.py ""                --eval
run 3dgs     gaussian_splatting /data1/zhangao/repos/gaussian_splatting  train.py "--disable_viewer" --eval
run hier     gaussian_splatting /data1/zhangao/repos/gaussian_splatting  train.py "--disable_viewer" --eval -d dav2
run mcmc     mcmc               /data1/zhangao/repos/3dgs-mcmc  train.py ""                --eval --cap_max 2000000
run mh       mh                 /data1/zhangao/repos/MH-3DGS    train.py "--disable_viewer" --eval
run scaffold scaffold           /data1/zhangao/repos/Scaffold-GS train.py ""                --eval
mvgs_ladder
say "===== pass complete (re-run to pick up newly-built envs) ====="
