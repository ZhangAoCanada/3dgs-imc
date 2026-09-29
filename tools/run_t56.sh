#!/usr/bin/env bash
# Run the Table-5/6 ours reruns (T&T discontinuous) on ONE machine/GPU. Partition-aware so the
# 22 jobs can be split across server-GPU1 + hust2-GPU2 + wsl. Each job -> diag_run.sh (SLOW pipeline,
# discontinuous split). Gates on free GPU memory so the shared server won't OOM another user's job.
#
# Env knobs:
#   CUDA_VISIBLE_DEVICES  GPU to use (server:1, hust2:2, wsl:0)
#   REPO_DIR              codex repo dir (default: this repo)
#   OUT_ROOT              output root (default /data1/zhangao/outputs/tables567)
#   DATA_ROOT             data root holding tandt/<scene> (default /data1/zhangao/DATA)
#   CONDA_EXE / ENV_NAME  conda binary + env (defaults = server)
#   JOBS                  which jobs: "ALL" or comma/space list of indices (1-22) or names
#   FREE_NEED             MiB free required before launching each job (default 12000)
set -uo pipefail
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export DATA_ROOT=${DATA_ROOT:-/data1/zhangao/DATA}
export OUT_ROOT=${OUT_ROOT:-/data1/zhangao/outputs/tables567}
export CONDA_EXE=${CONDA_EXE:-/data1/hewenxuan/miniconda3/bin/conda}
export ENV_NAME=${ENV_NAME:-3dgs-imc}
FREE_NEED=${FREE_NEED:-12000}
NVSMI=${NVSMI:-nvidia-smi}     # wsl override: /usr/lib/wsl/lib/nvidia-smi
JOBS=${JOBS:-ALL}
LOG="$OUT_ROOT/_run_t56_$(hostname)_g${CUDA_VISIBLE_DEVICES}.log"; mkdir -p "$OUT_ROOT"
say(){ echo "[$(date '+%m-%d %H:%M:%S') $(hostname):${CUDA_VISIBLE_DEVICES}] $*" | tee -a "$LOG"; }

# job table: idx  TAG  SCENE  CAP  EPS  NET  SLOW(1=full GS-LSS, 0=FAST/Eff for Table-7 ours timing)
JOBLIST=(
 "23 T7_ours_fast      truck_resize 2000000 0.25 sine 0"
 "1  T5_eps0.1_truck   truck_resize 2000000 0.1  sine 1"
 "2  T5_eps0.1_train   train_resize 1000000 0.1  sine 1"
 "3  T5_eps0.25_truck  truck_resize 2000000 0.25 sine 1"
 "4  T5_eps0.25_train  train_resize 1000000 0.25 sine 1"
 "5  T5_eps0.5_truck   truck_resize 2000000 0.5  sine 1"
 "6  T5_eps0.5_train   train_resize 1000000 0.5  sine 1"
 "7  T5_eps0.75_truck  truck_resize 2000000 0.75 sine 1"
 "8  T5_eps0.75_train  train_resize 1000000 0.75 sine 1"
 "9  T5_eps1.0_truck   truck_resize 2000000 1.0  sine 1"
 "10 T5_eps1.0_train   train_resize 1000000 1.0  sine 1"
 "11 T5_mlp_truck      truck_resize 2000000 0.25 relu 1"
 "12 T5_mlp_train      train_resize 1000000 0.25 relu 1"
 "13 T6_trial01_truck  truck_resize 2000000 0.25 sine 1"
 "14 T6_trial02_truck  truck_resize 2000000 0.25 sine 1"
 "15 T6_trial03_truck  truck_resize 2000000 0.25 sine 1"
 "16 T6_trial04_truck  truck_resize 2000000 0.25 sine 1"
 "17 T6_trial05_truck  truck_resize 2000000 0.25 sine 1"
 "18 T6_trial06_truck  truck_resize 2000000 0.25 sine 1"
 "19 T6_trial07_truck  truck_resize 2000000 0.25 sine 1"
 "20 T6_trial08_truck  truck_resize 2000000 0.25 sine 1"
 "21 T6_trial09_truck  truck_resize 2000000 0.25 sine 1"
 "22 T6_trial10_truck  truck_resize 2000000 0.25 sine 1"
)
want(){ [ "$JOBS" = "ALL" ] && return 0; case " $JOBS " in *" $1 "*) return 0;; *" $2 "*) return 0;; esac; return 1; }
wait_gpu(){ while :; do
  local free; free=$("$NVSMI" --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|tr -d ' '||echo 0)
  [ "${free:-0}" -ge "$FREE_NEED" ] && return 0; say "GPU ${free}MiB<${FREE_NEED}, wait 90s"; sleep 90; done; }

say "start JOBS=$JOBS OUT_ROOT=$OUT_ROOT env=$ENV_NAME"
for row in "${JOBLIST[@]}"; do
  read -r idx tag scene cap eps net slow <<<"$row"
  want "$idx" "$tag" || continue
  mp="$OUT_ROOT/tandt/${scene}_discontinuous_${tag}_i30000"
  if grep -q "ours_" "$mp/metrics.csv" 2>/dev/null; then say "SKIP $tag (already has metrics)"; continue; fi
  # Atomic per-job claim so concurrent workers (2 on a GPU, watchdog relaunches) never double-run
  # the same job. mkdir is atomic; a claim held by a LIVE worker pid => skip; a stale (dead-pid)
  # claim => take over. The claim is released on failure (no metrics) so the job stays retryable.
  mkdir -p "$OUT_ROOT/.claims" 2>/dev/null
  cdir="$OUT_ROOT/.claims/$tag"
  if mkdir "$cdir" 2>/dev/null; then echo "$$" > "$cdir/pid"; else
    cpid=$(cat "$cdir/pid" 2>/dev/null || echo "")
    if [ -n "$cpid" ] && kill -0 "$cpid" 2>/dev/null; then say "SKIP $tag (claimed by live $cpid)"; continue; fi
    say "$tag stale claim (pid '${cpid:-?}' dead) -> take over"; echo "$$" > "$cdir/pid"
  fi
  wait_gpu
  # Re-check after waiting: the job may have been completed meanwhile (e.g. by an orphaned training
  # whose worker-loop died). Avoid a wasteful duplicate re-run.
  if grep -q "ours_" "$mp/metrics.csv" 2>/dev/null; then say "SKIP $tag (completed while waiting)"; rm -rf "$cdir"; continue; fi
  # SLOW (Tables 5/6) uses refresh=0 (all views) to match the previous-best develop3 runs;
  # FAST (Table-7 timing) keeps the Eff default 16. (Server is headless -> all-views is safe.)
  refresh=16; [ "$slow" = "1" ] && refresh=0
  say "RUN [$idx] $tag scene=$scene cap=$cap eps=$eps net=$net slow=$slow refresh=$refresh"
  SCENE="$scene" DATASET=tandt SPLIT=discontinuous SLOW="$slow" ANGLE=0 \
    IMAGES=images DEPTHS=dav2 CAP_MAX="$cap" EPS="$eps" CPF_NET_TYPE="$net" \
    CPF_MAX_REFRESH_VIEWS="$refresh" \
    DEPTH_INIT=0.3 DEPTH_FINAL=0.003 ITER=30000 ITERATIONS=30000 \
    SAVE_ITERATIONS=30000 RUN_EVAL=1 TAG="$tag" \
    bash diag_run.sh >>"$LOG" 2>&1
  p=$(awk -F, 'NR>1 && $3 ~ /ours_/{print $5}' "$mp/metrics.csv" 2>/dev/null | tail -1)
  if [ -z "$p" ]; then rm -rf "$cdir"; say "DONE $tag PSNR=NA -> released claim (retryable)"; else say "DONE $tag PSNR=$p"; fi
done
say "worker finished (JOBS=$JOBS)"
