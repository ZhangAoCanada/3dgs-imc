#!/usr/bin/env bash
# KEEPALIVE WATCHDOG — never let a GPU sit idle while Table-5/6 metric jobs remain.
# Root cause of earlier idle: each phase finished and nothing relaunched the next work.
# This supervisor TRACKS THE PIDS IT LAUNCHES (kill -0 on specific pids; NO pgrep => immune to
# self-matching status commands), keeps 2 claim-aware run_t56 workers alive on server GPU1 and 1 on
# hust2 GPU2 (after the mcmc clean re-time frees GPU2), relaunches any that die while jobs remain,
# and exits once every assigned job has metrics. Single-instance via flock.
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
OUT_ROOT=/data1/zhangao/outputs/tables567
LOG="$OUT_ROOT/_keepalive_watchdog.log"
INT=${INT:-90}
TARGET_SRV=${TARGET_SRV:-2}
LOCK=/tmp/keepalive_watchdog.lock
exec 9>"$LOCK"; flock -n 9 || { echo "[wd] another watchdog holds the lock -> exit"; exit 0; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Disjoint partitions (server fs and hust2 fs are SEPARATE => partition, not claim, dedups across machines)
# hust2's 4090 (sm_89) now RUNS ours: fixed by patching the CPF DBSCAN to take numpy directly (the
# cudf.DataFrame wrapper crashed via the broken cudf->cupy->rmm chain; numpy path = byte-identical labels).
# Balanced partition: server 2 slots (fast 5090) gets 9 jobs, hust2 1 slot (4090) gets 6 trials.
# TABLE 6 LAST (user 2026-06-21): each machine clears its Table-5 cells (jobs 6-11) BEFORE any trial
# (jobs 14-22) — run_t56 iterates the job table in index order, so Table-5 precedes trials automatically.
SRV_JOBS="6 7 8 9 14 15 16 17 18"   # server: Table5 {eps0.5_tr,eps0.75_t/tr,eps1.0_t} then trials 02-06
HUST2_JOBS="10 11 19 20 21 22"      # hust2: Table5 {eps1.0_tr, mlp_truck} then trials 07-10
declare -A JOBMAP=(
 [4]="T5_eps0.25_train train_resize" [5]="T5_eps0.5_truck truck_resize" [6]="T5_eps0.5_train train_resize"
 [7]="T5_eps0.75_truck truck_resize" [8]="T5_eps0.75_train train_resize" [9]="T5_eps1.0_truck truck_resize"
 [10]="T5_eps1.0_train train_resize" [11]="T5_mlp_truck truck_resize"
 [14]="T6_trial02_truck truck_resize" [15]="T6_trial03_truck truck_resize" [16]="T6_trial04_truck truck_resize"
 [17]="T6_trial05_truck truck_resize" [18]="T6_trial06_truck truck_resize" [19]="T6_trial07_truck truck_resize"
 [20]="T6_trial08_truck truck_resize" [21]="T6_trial09_truck truck_resize" [22]="T6_trial10_truck truck_resize"
)
undone_local(){ local n=0 i t s; for i in $1; do read -r t s <<<"${JOBMAP[$i]}"
  grep -q ours_ "$OUT_ROOT/tandt/${s}_discontinuous_${t}_i30000/metrics.csv" 2>/dev/null || n=$((n+1)); done; echo "$n"; }
undone_hust2(){ [ -z "$HUST2_JOBS" ] && { echo 0; return; }
  local list="" i t s; for i in $HUST2_JOBS; do read -r t s <<<"${JOBMAP[$i]}"; list+="$OUT_ROOT/tandt/${s}_discontinuous_${t}_i30000/metrics.csv "; done
  timeout 30 ssh hust2 "n=0; for f in $list; do grep -q ours_ \$f 2>/dev/null || n=\$((n+1)); done; echo \$n" 2>/dev/null || echo "?"; }
mcmc_clean_done(){ timeout 20 ssh hust2 'test -s /data1/zhangao/outputs/baselines_t7_clean/mcmc/wall_seconds.txt && echo yes || echo no' 2>/dev/null || echo no; }

# ---- server workers: track launched PIDs (robust, no pattern matching) ----
declare -a SRV_PIDS=()
prune_srv(){ local p; local -a keep=(); if [ "${#SRV_PIDS[@]}" -gt 0 ]; then
  for p in "${SRV_PIDS[@]}"; do kill -0 "$p" 2>/dev/null && keep+=("$p"); done; fi; SRV_PIDS=("${keep[@]}"); }
launch_srv(){ CUDA_VISIBLE_DEVICES=1 OUT_ROOT="$OUT_ROOT" CONDA_EXE=/opt/miniconda3/bin/conda ENV_NAME=3dgs-imc \
    DATA_ROOT=/data1/zhangao/DATA FREE_NEED=14000 JOBS="$SRV_JOBS" \
    nohup bash tools/run_t56.sh >>/tmp/run_t56_wd.log 2>&1 9>&- & local pid=$!; SRV_PIDS+=("$pid"); say "launched SERVER worker pid $pid"; }

# ---- hust2 worker: track its remote pid ----
HUST2_PID=""
hust2_alive(){ [ -n "$HUST2_PID" ] || { echo no; return; }; timeout 20 ssh hust2 "kill -0 $HUST2_PID 2>/dev/null && echo yes || echo no" 2>/dev/null || echo no; }
launch_hust2(){ HUST2_PID=$(timeout 40 ssh hust2 'cd /data1/ZHANGAo_2024/3dgs-imc-codex && CUDA_VISIBLE_DEVICES=2 \
    DATA_ROOT=/data2/zhangao/tandt_db OUT_ROOT=/data1/zhangao/outputs/tables567 CONDA_EXE=/opt/anaconda3/bin/conda \
    ENV_NAME=3dgs-imc FREE_NEED=16000 JOBS="'"$HUST2_JOBS"'" nohup bash tools/run_t56.sh >>/data1/zhangao/outputs/tables567/_run_t56_wd.log 2>&1 & echo $!' 2>/dev/null)
  say "launched HUST2 worker remote pid ${HUST2_PID:-FAIL}"; }

say "===== keepalive watchdog START (PID-tracked; server target=$TARGET_SRV, interval=${INT}s) ====="
while :; do
  prune_srv
  rem_s=$(undone_local "$SRV_JOBS"); rem_h=$(undone_hust2)
  ls=${#SRV_PIDS[@]}
  say "server: remaining=$rem_s live=$ls | hust2: remaining=$rem_h"
  if [ "$rem_s" -gt 0 ]; then while [ "$ls" -lt "$TARGET_SRV" ]; do launch_srv; ls=$((ls+1)); sleep 4; done; fi
  if [ "$rem_h" != "?" ] && [ "${rem_h:-0}" -gt 0 ]; then
    if [ "$(mcmc_clean_done)" = "yes" ]; then
      [ "$(hust2_alive)" = "yes" ] || launch_hust2
    else say "hust2: waiting for mcmc clean re-time to free GPU2 before ours trials"; fi
  fi
  if [ "$rem_s" -eq 0 ] && [ "${rem_h:-1}" = "0" ]; then say "ALL metric jobs DONE (server+hust2) -> watchdog exit"; break; fi
  sleep "$INT"
done
