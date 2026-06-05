#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 LOG_PATH -- COMMAND [ARGS...]" >&2
    exit 2
fi

LOG_PATH=$1
shift
if [[ ${1:-} == "--" ]]; then
    shift
fi
if [[ $# -eq 0 ]]; then
    echo "missing command" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$(dirname "${LOG_PATH}")"
: > "${LOG_PATH}"

GPU=${GPU:-0}
INTERVAL=${INTERVAL:-30}
WARMUP=${WARMUP:-300}
STALL_TIMEOUT=${STALL_TIMEOUT:-900}
OUTPUT_STALL_TIMEOUT=${OUTPUT_STALL_TIMEOUT:-900}
MAX_RUNTIME=${MAX_RUNTIME:-2700}
LABEL=${LABEL:-experiment_stall}
if [[ "${GPU}" == "none" || "${GPU}" == "disabled" || "${GPU}" == "off" ]]; then
    REQUIRE_GPU_AFTER=${REQUIRE_GPU_AFTER:-0}
else
    REQUIRE_GPU_AFTER=${REQUIRE_GPU_AFTER:-300}
fi

MONITOR_CMD=(
    python3 "${SCRIPT_DIR}/monitor_experiment.py"
    --log "${LOG_PATH}"
    --gpu "${GPU}"
    --interval "${INTERVAL}"
    --warmup "${WARMUP}"
    --stall-timeout "${STALL_TIMEOUT}"
    --output-stall-timeout "${OUTPUT_STALL_TIMEOUT}"
    --max-runtime "${MAX_RUNTIME}"
    --require-gpu-after "${REQUIRE_GPU_AFTER}"
    --label "${LABEL}"
    -- "$@"
)

if [[ "${BACKGROUND:-0}" == "1" ]]; then
    nohup "${MONITOR_CMD[@]}" > "${LOG_PATH}.console" 2>&1 &
    MONITOR_PID=$!
    echo "${MONITOR_PID}" > "${LOG_PATH}.monitor.pid"
    sleep 2
    if ! kill -0 "${MONITOR_PID}" 2>/dev/null; then
        set +e
        wait "${MONITOR_PID}"
        RC=$?
        set -e
        echo "monitor exited early rc=${RC}; see ${LOG_PATH}.console" >&2
        if [[ -s "${LOG_PATH}.console" ]]; then
            tail -n 20 "${LOG_PATH}.console" >&2 || true
        fi
        exit "${RC}"
    fi

    echo "monitor_pid=${MONITOR_PID}"
    echo "log=${LOG_PATH}"
    echo "heartbeat=${LOG_PATH}.heartbeat"
else
    echo "monitor_mode=foreground"
    echo "log=${LOG_PATH}"
    echo "heartbeat=${LOG_PATH}.heartbeat"
    set +e
    "${MONITOR_CMD[@]}" 2>&1 | tee "${LOG_PATH}.console"
    RC=${PIPESTATUS[0]}
    set -e
    exit "${RC}"
fi
