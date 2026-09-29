#!/usr/bin/env bash
# Table-5 CPF ablation REDO on T&T@60 (truck_resize + train_resize), SLOW develop3 pipeline
# (matches the T&T@60 Table-1 main result). Varies ONLY EPS (DBSCAN) and CPF_NET_TYPE (sine/relu);
# everything else = the develop3_multi_angle2 config (AA on, images/dav2, min_samples 500, nview3,
# depth0.3, caps truck=2M/train=1M). GPU 1 only, serial. Each run -> its own MODEL_PATH, skip-if-done.
# ε=0.25/sine is run FIRST (doubles as the ε=0.25 row + the SIREN row + the reproduction check).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
C=${CONDA_EXE:-/data1/hewenxuan/miniconda3/bin/conda}
ENV=${ENV_NAME:-3dgs-imc}
ROOT=/data1/zhangao/outputs/tnt60_table5
LOG="$ROOT/_driver.log"; mkdir -p "$ROOT"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
# tag  EPS  NET   (ε=0.25/sine first: it is BOTH the ε=0.25 row AND the SIREN row AND the repro check)
CONFIGS=(
 "eps0.25_sine 0.25 sine"
 "eps0.1_sine  0.1  sine"
 "eps0.5_sine  0.5  sine"
 "eps0.75_sine 0.75 sine"
 "eps1.0_sine  1.0  sine"
 "mlp_eps0.25  0.25 relu"
)
declare -A CAP=( [truck_resize]=2000000 [train_resize]=1000000 )
say "===== T&T@60 Table-5 ablation START (SLOW, GPU1, AA=1) ====="
for row in "${CONFIGS[@]}"; do
  read -r tag eps net <<<"$row"
  for sc in truck_resize train_resize; do
    mp="$ROOT/${sc}_${tag}_aa1_i30000"
    if grep -q "ours_" "$mp/metrics.csv" 2>/dev/null; then
      say "SKIP $sc/$tag ($(awk -F, 'NR>1&&$3~/ours_/{print "PSNR="$5}' "$mp/metrics.csv"|tail -1))"; continue
    fi
    say "RUN $sc/$tag eps=$eps net=$net cap=${CAP[$sc]}"
    SCENE="$sc" ANGLE=60.0 CAP_MAX="${CAP[$sc]}" DATASET=tandt DATA_ROOT=/data1/zhangao/DATA \
      OUT_ROOT="$ROOT" MODEL_PATH="$mp" ITER=30000 AA=1 EPS="$eps" CPF_NET_TYPE="$net" \
      IMAGES=images DEPTHS=dav2 CONDA_EXE="$C" ENV_NAME="$ENV" \
      bash repro_develop3.sh >>"$LOG" 2>&1 \
      && say "  DONE $sc/$tag: $(awk -F, 'NR>1&&$3~/ours_/{print "SSIM="$4" PSNR="$5" LPIPS="$6" BRISQUE="$7" NIQE="$8" PIQE="$9}' "$mp/metrics.csv"|tail -1)" \
      || say "  FAILED $sc/$tag (see log)"
  done
done
say "===== T&T@60 Table-5 ablation DONE ====="