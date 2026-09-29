#!/usr/bin/env bash
# Stage-3 aerial-ground diagnostic (GPU-gated, polite). Isolates WHY GS-LSS underperforms vanilla
# 3DGS-MCMC (16.225) on MatrixCity (our best = 14.564). Two axes, both on the SLOW pipeline:
#   - depth regularization OFF (paper's own Table 3 shows depth/geo methods crater on aerial-ground)
#   - CPF mechanism OFF  (CPF_DISABLE=1 -> MCMC base; isolates whether CPF hurts the disparity case)
# Waits until GPU1 has enough free memory (won't collide with the other user's job), runs sequentially.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=1                 # server rule: GPU1 ONLY (GPU0 is off-limits)
GPU=1
FREE_NEED=${FREE_NEED:-22000}                 # MiB free on GPU1 required before launching a cap3M run
SCENE="blockA_fusion_small_aerial+somestreet"
OUTD=/data1/zhangao/outputs/diag/large_angle
LOG="$OUTD/_diagnostic.log"; mkdir -p "$OUTD"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

wait_gpu(){
  while :; do
    local free; free=$(nvidia-smi --id=$GPU --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo 0)
    if [ "${free:-0}" -ge "$FREE_NEED" ]; then say "GPU$GPU ${free}MiB free (>=$FREE_NEED) -> launch"; return 0; fi
    say "GPU$GPU ${free}MiB free (<$FREE_NEED); other user busy, waiting 120s..."; sleep 120
  done
}
harvest(){  # echo best ours_ PSNR from a model dir's metrics.csv
  python3 - "$1" <<'PY'
import csv,glob,sys,os
mp=sys.argv[1]; best=None
for f in glob.glob(os.path.join(mp,"metrics.csv")):
    try:
        for r in csv.DictReader(open(f)):
            if r.get("method","").startswith("ours_"):
                best=max(best or 0.0, float(r["PSNR"]))
    except Exception: pass
print(f"{best:.3f}" if best else "NA")
PY
}
run_job(){  # tag  extra-env-assignments
  local tag=$1; shift
  wait_gpu
  say "RUN $tag : $*"
  ( set -a
    SCENE="$SCENE"; DATASET=large_angle; SPLIT=traintest; SLOW=1; ANGLE=0
    IMAGES=input_cached; DEPTHS=dav2_cached; CAP_MAX=3000000; EPS=0.1; AA=0
    ITER=30000; ITERATIONS=30000; RUN_EVAL=1; CONDA_EXE=/data1/hewenxuan/miniconda3/bin/conda
    eval "$*"; TAG="$tag"
    bash diag_run.sh ) >>"$LOG" 2>&1
  local rc=$? mp="$OUTD/${SCENE}_traintest_${tag}_i30000"
  say "DONE $tag rc=$rc PSNR=$(harvest "$mp")  ($mp)"
}

say "===== Stage-3 MatrixCity diagnostic START (beat MCMC 16.225 / published 16.951; baseline ours=14.564) ====="
run_job "DG_depthoff"        'DEPTH_INIT=0 DEPTH_FINAL=0'                       # depth-reg hypothesis
run_job "DG_cpfoff"          'CPF_DISABLE=1 DEPTH_INIT=0.3 DEPTH_FINAL=0.003'  # CPF hypothesis
run_job "DG_cpfoff_depthoff" 'CPF_DISABLE=1 DEPTH_INIT=0 DEPTH_FINAL=0'        # both off -> MCMC-like base
say "===== diagnostic COMPLETE. Summary: ====="
for t in DG_depthoff DG_cpfoff DG_cpfoff_depthoff; do
  say "  $t -> PSNR $(harvest "$OUTD/${SCENE}_traintest_${t}_i30000")"
done
say "(baseline ours=14.564, MCMC=16.225, published=16.951)"
