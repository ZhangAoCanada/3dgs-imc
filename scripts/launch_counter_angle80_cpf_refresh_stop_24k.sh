#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_PATH=${LOG_PATH:-outputs/monitor/counter_angle80_lss_cpfrefreshstop_i24000_${STAMP}.log}

export GPU=${GPU:-1}
export SCENE=${SCENE:-counter}
export ITERATIONS=${ITERATIONS:-24000}
export CPF_REFRESH_STOP_ITER=${CPF_REFRESH_STOP_ITER:-18000}
export CPF_STOP_ITER=${CPF_STOP_ITER:-0}
export CPF_UPDATE_LOSS_THRESHOLD=${CPF_UPDATE_LOSS_THRESHOLD:-0}
export RANDOM_VIEW_NUM=${RANDOM_VIEW_NUM:-1}
export RUN_EVAL=${RUN_EVAL:-1}
export MODEL_ROOT=${MODEL_ROOT:-outputs/fast_lss_dev}
export OUTPUT_STALL_TIMEOUT=${OUTPUT_STALL_TIMEOUT:-900}
export STALL_TIMEOUT=${STALL_TIMEOUT:-900}
export REQUIRE_GPU_AFTER=${REQUIRE_GPU_AFTER:-300}
export MAX_RUNTIME=${MAX_RUNTIME:-2700}
export LABEL=${LABEL:-counter_angle80_lss_cpfrefreshstop_i24000}

if [[ "${SKIP_DATA_PREFLIGHT:-0}" != "1" ]]; then
    python3 scripts/preflight_angle80_experiment.py --scenes "${SCENE}" --methods lss_refreshstop
fi

"${SCRIPT_DIR}/launch_monitored_experiment.sh" "${LOG_PATH}" -- bash run_fast_lss_dev.sh
