#!/usr/bin/env bash
# CLEAN solo re-time of all 7 Table-7 baselines on hust2 GPU2 (truck_resize, 30k iters).
# Each method runs ONLY when GPU2 is idle (>=22GB free of 24GB => no co-resident training),
# guaranteeing an alone measurement. Sequential. Writes to baselines_t7_clean/<name>/wall_seconds.txt
# so the previous (possibly-concurrent) baselines_t7_time/ is preserved for comparison.
# Single-instance via flock so a stray second copy can never run concurrently (the original bug).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
LOCK=/tmp/hust2_retime_clean.lock
exec 9>"$LOCK"; flock -n 9 || { echo "[retime] another instance holds the lock -> exit"; exit 0; }
C=/opt/anaconda3/bin/conda
SRC=/data2/zhangao/tandt_db/tandt/truck_resize
OUT=/data1/zhangao/outputs/baselines_t7_clean; mkdir -p "$OUT"
LOG="$OUT/_clean.log"
FREE_NEED=${FREE_NEED:-22000}    # >=22GB free of 24GB == GPU2 essentially idle (no other training)
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
have_env(){ $C env list 2>/dev/null | grep -q "^$1 "; }
wait_alone(){ while :; do
  local free; free=$(nvidia-smi --id=2 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|tr -d ' '||echo 0)
  [ "${free:-0}" -ge "$FREE_NEED" ] && { say "GPU2 ${free}MiB free -> run SOLO"; return; }
  say "GPU2 only ${free}MiB free (<${FREE_NEED}); something co-resident, wait 60s"; sleep 60; done; }
run(){ local name=$1 env=$2 repo=$3 entry=$4 viewer=$5; shift 5
  local mp="$OUT/$name"
  [ -f "$mp/wall_seconds.txt" ] && { say "SKIP $name (clean time exists: $(cat "$mp/wall_seconds.txt")s)"; return; }
  have_env "$env" || { say "SKIP $name (env '$env' missing)"; return; }
  wait_alone
  say "RUN(solo) $name env=$env repo=$repo"
  rm -rf "$mp"; mkdir -p "$mp"; local t0=$(date +%s) port=$(( 6300 + RANDOM % 600 ))
  local portarg="--port $port"; [ "$env" = "mcmc" ] && portarg=""   # 3dgs-mcmc train.py has no --port
  ( cd "$repo" && $C run --no-capture-output -n "$env" python "$entry" -s "$SRC" -m "$mp" $viewer $portarg "$@" ) >>"$LOG" 2>&1
  local rc=$? secs=$(( $(date +%s) - t0 ))
  if [ "$rc" = 0 ]; then echo "$secs" > "$mp/wall_seconds.txt"; say "DONE $name rc=0 time=${secs}s ($((secs/60))min)"
  else say "FAIL $name rc=$rc (see log)"; fi
}
mvgs_clean(){ local mp="$OUT/mvgs"
  [ -f "$mp/wall_seconds.txt" ] && { say "SKIP mvgs (clean time exists)"; return; }
  have_env mvgs || { say "SKIP mvgs (env missing)"; return; }
  local mv; for mv in 3 2 1; do
    wait_alone; rm -rf "$mp"; mkdir -p "$mp"; say "RUN(solo) mvgs --mv $mv (default-first)"
    local t0=$(date +%s) port=$(( 6300 + RANDOM % 600 ))
    ( cd /data1/zhangao/repos/MVGS && $C run --no-capture-output -n mvgs python train.py -s "$SRC" -m "$mp" --port "$port" --eval --mv "$mv" ) >>"$LOG" 2>&1
    local rc=$? secs=$(( $(date +%s) - t0 ))
    if [ "$rc" = 0 ]; then echo "$secs" > "$mp/wall_seconds.txt"; echo "$mv" > "$mp/mv_used.txt"; say "DONE mvgs --mv $mv time=${secs}s ($((secs/60))min)"; return; fi
    say "mvgs --mv $mv FAIL rc=$rc -> try lower mv"
  done; say "mvgs failed at all mv"; }
say "===== CLEAN solo re-time START (FREE_NEED=${FREE_NEED}, single-instance) ====="
run 3dgs     gaussian_splatting /data1/zhangao/repos/gaussian_splatting train.py "--disable_viewer" --eval
run hier     gaussian_splatting /data1/zhangao/repos/gaussian_splatting train.py "--disable_viewer" --eval -d dav2
run mh       mh                 /data1/zhangao/repos/MH-3DGS    train.py "--disable_viewer" --eval
run mcmc     mcmc               /data1/zhangao/repos/3dgs-mcmc  train.py ""                --eval --cap_max 2000000
run scaffold scaffold           /data1/zhangao/repos/Scaffold-GS train.py ""                --eval
run pgsr     pgsr               /data1/ZHANGAo_2024/PGSR        train.py ""                --eval
mvgs_clean
say "===== CLEAN solo re-time DONE -> compare baselines_t7_clean vs baselines_t7_time ====="
