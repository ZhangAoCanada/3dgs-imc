#!/usr/bin/env bash
# diag_run.sh — isolated single (scene,angle) run mirroring run.sh's BEST config,
# with selected knobs overridable from the caller's env, to root-cause the
# regression vs the develop3 (paper) GS-LSS runs. GPU 1 only.
set -uo pipefail
# default GPU 1 (dev server rule); workers on other machines override via env
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

SCENE=${SCENE:?set SCENE}
ANGLE=${ANGLE:-0}                         # required only for angle splits; traintest/discontinuous ignore it
TAG=${TAG:?set TAG}                       # output subdir tag, e.g. depth03
DATA_ROOT=${DATA_ROOT:-/data1/zhangao/DATA}
DATASET=${DATASET:-mipnerf360}
OUT_ROOT=${OUT_ROOT:-/data1/zhangao/outputs/diag}
SRC="${DATA_ROOT}/${DATASET}/${SCENE}"

# ---- run.sh BEST config baseline (overridable via env) ----
export CPF_PRIOR_SOURCE=${CPF_PRIOR_SOURCE:-render}
export CPF_MAX_POINTS=${CPF_MAX_POINTS:-100000}
export CPF_DRIFT_MAX_POINTS=${CPF_DRIFT_MAX_POINTS:-0}
export CPF_TRAIN_INTERVAL=${CPF_TRAIN_INTERVAL:-8}
export CPF_DERIVATIVE_INTERVAL=${CPF_DERIVATIVE_INTERVAL:-2}
export CPF_MAX_REFRESH_VIEWS=${CPF_MAX_REFRESH_VIEWS:-16}
export SCALE_REG=${SCALE_REG:-0.01}
export OPACITY_REG=${OPACITY_REG:-0.01}
export ANTIALIASING=${ANTIALIASING:-0}
export DEPTH_L1_WEIGHT_INIT=${DEPTH_L1_WEIGHT_INIT:-0}
export DEPTH_L1_WEIGHT_FINAL=${DEPTH_L1_WEIGHT_FINAL:-0}
export DEPTH_ALIGN_SFM_ON_DEMAND=${DEPTH_ALIGN_SFM_ON_DEMAND:-0}
# shipped redesign defaults (match run.sh): on-manifold + coverage-targeted noise
export CPF_NOISE_TANGENT=${CPF_NOISE_TANGENT:-1.0}
export CPF_NOISE_OBS_GATE=${CPF_NOISE_OBS_GATE:-1}
export CPF_MAX_NOISE_NORM=${CPF_MAX_NOISE_NORM:-0.005}
export CPF_NOISE_WEIGHT=${CPF_NOISE_WEIGHT:-1.0}
export CPF_NOISE_TRAIN_WEIGHT=${CPF_NOISE_TRAIN_WEIGHT:-0}
# prototype knobs (Design 3 floater-drift, Design 1 view-dir noise); default OFF
export CPF_PULL_WEIGHT=${CPF_PULL_WEIGHT:-0} CPF_PULL_OBS_GATE=${CPF_PULL_OBS_GATE:-0} CPF_PULL_PERSIST=${CPF_PULL_PERSIST:-0}
export CPF_NOISE_VIEWDIR=${CPF_NOISE_VIEWDIR:-0}
export CPF_FIELD_SFM_ANCHOR=0 CPF_FIELD_ALPHA_THRESH=0
export RANDOM_VIEW_NUM=${RANDOM_VIEW_NUM:-1}
export NN_START_ITER=${NN_START_ITER:-2500}
export NN_DERIVATIVES_ITER=${NN_DERIVATIVES_ITER:-3000}
export NN_UPDATE_INTERVAL=${NN_UPDATE_INTERVAL:-3000}
export PARTIAL_SCALING=${PARTIAL_SCALING:-1e0}
export SIGMA_SCALING=${SIGMA_SCALING:-1e-6}
export DENSIFY_FROM_ITER=${DENSIFY_FROM_ITER:-500}
export DENSIFY_UNTIL_ITER=${DENSIFY_UNTIL_ITER:-25000}
export DENSIFICATION_INTERVAL=${DENSIFICATION_INTERVAL:-200}
export ITERATIONS=${ITERATIONS:-30000}
export RUN_EVAL=${RUN_EVAL:-1}
export TIMING_BREAKDOWN=${TIMING_BREAKDOWN:-0}

export SCENE SOURCE_PATH="$SRC"
export CAP_MAX=${CAP_MAX:?set CAP_MAX}
export IMAGES=${IMAGES:-images_4}
export DEPTHS=${DEPTHS:-dav2_4}
export TEST_PATH=""
if [ "${SPLIT:-}" = "traintest" ]; then   # Table 3 aerial-ground: train on ${SRC}/train, evaluate on ${SRC}/test
    export SOURCE_PATH="${SRC}/train"
    export TEST_PATH="${SRC}/test"
    export TEST_SELECTION=""
    export LABEL="${SCENE}_traintest_${TAG}"
    export MODEL_PATH="${OUT_ROOT}/${DATASET}/${SCENE}_traintest_${TAG}_i${ITERATIONS}"
elif [ -n "${SPLIT:-}" ]; then    # SPLIT=discontinuous -> ${SRC}/discontinuous.txt (Table 2); ANGLE unused
    export TEST_SELECTION="${SRC}/${SPLIT}.txt"
    export LABEL="${SCENE}_${SPLIT}_${TAG}"
    export MODEL_PATH="${OUT_ROOT}/${DATASET}/${SCENE}_${SPLIT}_${TAG}_i${ITERATIONS}"
else
    export TEST_SELECTION="${SRC}/test_selection_angle_${ANGLE}.txt"
    export LABEL="${SCENE}_angle_${ANGLE}_${TAG}"
    export MODEL_PATH="${OUT_ROOT}/${DATASET}/${SCENE}_angle_${ANGLE}_${TAG}_i${ITERATIONS}"
fi

echo "======== DIAG ${SCENE}@${ANGLE} [${TAG}] cap=${CAP_MAX} depthW=${DEPTH_L1_WEIGHT_INIT} nview=${RANDOM_VIEW_NUM} prior=${CPF_PRIOR_SOURCE} slow=${SLOW:-0} -> ${MODEL_PATH}"
if [ "${SLOW:-0}" = "1" ]; then    # SLOW pipeline: original train_experimental_branch3.py + depth + nview3 (GS-LSS recipe)
    PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} bash repro_develop3.sh; rc=$?
else
    PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} bash run_fast_lss_dev.sh; rc=$?
fi
# Belt-and-suspenders: a run is only successful if it left an ours_ metric (i.e. a real
# checkpoint that rendered). A uvm-killed run leaves none -> propagate failure so the
# worker routes it to fail/ (retryable), instead of silently marking it done.
if ! grep -q "ours_" "${MODEL_PATH}/metrics.csv" 2>/dev/null; then rc=1; fi
echo "[diag_run] DONE (rc=${rc}) ${MODEL_PATH}"
exit "$rc"
