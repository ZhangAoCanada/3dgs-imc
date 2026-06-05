#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_DIR}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_PATH="outputs/monitor/counter_angle80_lss_refine_from18k_i24000_${STAMP}.log"

export SCENE=${SCENE:-counter}
export ITERATIONS=${ITERATIONS:-24000}
export LOAD_ITERATION=${LOAD_ITERATION:-18000}
export LOAD_MODEL_PATH=${LOAD_MODEL_PATH:-outputs/fast_lss_dev/counter_angle80_lss_sfmdepth_i18000_depth1_images4}
export CPF_REFRESH_STOP_ITER=${CPF_REFRESH_STOP_ITER:-18000}
export CPF_TRAIN_STOP_ITER=${CPF_TRAIN_STOP_ITER:-18000}
export CPF_STOP_ITER=${CPF_STOP_ITER:-18000}
export STOCHASTIC_STOP_ITER=${STOCHASTIC_STOP_ITER:-18000}
export CPF_UPDATE_LOSS_THRESHOLD=${CPF_UPDATE_LOSS_THRESHOLD:-0}
export RANDOM_VIEW_NUM=${RANDOM_VIEW_NUM:-1}
export DEPTH_L1_WEIGHT_INIT=${DEPTH_L1_WEIGHT_INIT:-1.0}
export DEPTH_L1_WEIGHT_FINAL=${DEPTH_L1_WEIGHT_FINAL:-0.01}
export DEPTH_L1_STOP_ITER=${DEPTH_L1_STOP_ITER:-18000}
export TIMING_BREAKDOWN=${TIMING_BREAKDOWN:-1}
export TIMING_SYNC=${TIMING_SYNC:-0}
export RUN_EVAL=${RUN_EVAL:-1}
export MODEL_PATH=${MODEL_PATH:-outputs/fast_lss_dev/${SCENE}_angle80_lss_sfmdepth_i${ITERATIONS}_depth1_refine_from${LOAD_ITERATION}_images4}
export LABEL=${LABEL:-counter_angle80_lss_refine_from${LOAD_ITERATION}_i${ITERATIONS}}

if [[ ! -f "${LOAD_MODEL_PATH}/point_cloud/iteration_${LOAD_ITERATION}/point_cloud.ply" ]]; then
    echo "missing load PLY: ${LOAD_MODEL_PATH}/point_cloud/iteration_${LOAD_ITERATION}/point_cloud.ply" >&2
    exit 2
fi

if [[ "${SKIP_DATA_PREFLIGHT:-0}" != "1" ]]; then
    python3 scripts/preflight_angle80_experiment.py --scenes "${SCENE}" --methods lss_refine
fi

"${SCRIPT_DIR}/launch_monitored_experiment.sh" "${LOG_PATH}" -- bash run_fast_lss_dev.sh
