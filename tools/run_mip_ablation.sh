#!/usr/bin/env bash
# Table-5 CPF ablation on Mip-NeRF 360 coverage-gap 60 (matches Table-1 protocol via run.sh).
# Varies ONLY EPS (DBSCAN) and CPF_NET_TYPE (sine=SIREN / relu=MLP); everything else is the
# per-scene best config that run.sh's run_best sets (cap, AA, floater-prune, nview, images_4, dav2_4).
# Each config -> its own OUT_ROOT so runs don't overwrite. FAST/Eff pipeline (run_fast_lss_dev.sh).
#
# Env: SCENES (default "bonsai kitchen"), CUDA_VISIBLE_DEVICES, FREE_NEED, REPO_DIR, DATA_ROOT.
set -uo pipefail
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export DATA_ROOT=${DATA_ROOT:-/data1/zhangao/DATA}
ABL_ROOT=${ABL_ROOT:-/data1/zhangao/outputs/mip_ablation}
SCENES=${SCENES:-"bonsai kitchen"}
CONFIGS_FILTER=${CONFIGS_FILTER:-}   # space-separated config tags to run (empty = all 6)
ANGLE=${ANGLE:-60.0}
FREE_NEED=${FREE_NEED:-14000}
NVSMI=${NVSMI:-nvidia-smi}
LOG="$ABL_ROOT/_run_mip_ablation_$(hostname)_g${CUDA_VISIBLE_DEVICES}.log"; mkdir -p "$ABL_ROOT"
say(){ echo "[$(date '+%m-%d %H:%M:%S') $(hostname):${CUDA_VISIBLE_DEVICES}] $*" | tee -a "$LOG"; }
wait_gpu(){ while :; do
  local f; f=$("$NVSMI" --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|tr -d ' '||echo 0)
  [ "${f:-0}" -ge "$FREE_NEED" ] && return 0; say "GPU ${f}MiB<${FREE_NEED}, wait 90s"; sleep 90; done; }

# config table:  tag  EPS  NET
CONFIGS=(
 "eps0.1_sine   0.1  sine"
 "eps0.25_sine  0.25 sine"   # == SIREN-style row
 "eps0.5_sine   0.5  sine"
 "eps0.75_sine  0.75 sine"
 "eps1.0_sine   1.0  sine"
 "mlp_eps0.25   0.25 relu"   # == MLP row
)
say "start SCENES='$SCENES' ANGLE=$ANGLE ABL_ROOT=$ABL_ROOT"
for sc in $SCENES; do
  for row in "${CONFIGS[@]}"; do
    read -r tag eps net <<<"$row"
    [ -n "$CONFIGS_FILTER" ] && case " $CONFIGS_FILTER " in *" $tag "*) :;; *) continue;; esac
    out="$ABL_ROOT/${sc}_${tag}"
    mp="$out/mipnerf360/${sc}_angle_${ANGLE}_i30000_aa"
    if grep -q "ours_" "$mp/metrics.csv" 2>/dev/null; then say "SKIP ${sc}/${tag} (has metrics)"; continue; fi
    wait_gpu
    say "RUN ${sc}/${tag} eps=$eps net=$net -> $out"
    OUT_ROOT="$out" EPS="$eps" CPF_NET_TYPE="$net" MIN_SAMPLES=500 \
      SCENES="$sc" ANGLES="$ANGLE" DATASETS=mipnerf360 BIASES=angle RUN_EVAL=1 \
      bash run.sh >>"$LOG" 2>&1
    p=$(awk -F, 'NR>1 && $3~/ours_/{print $5}' "$mp/metrics.csv" 2>/dev/null | tail -1)
    say "DONE ${sc}/${tag} PSNR=${p:-NA}"
  done
done
say "ablation worker finished (SCENES='$SCENES')"
