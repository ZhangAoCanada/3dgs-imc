#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
if [[ "${SKIP_UVM_PREFLIGHT:-0}" != "1" ]]; then
    python3 "${REPO_ROOT}/scripts/check_nvidia_uvm_state.py"
fi

CONDA_EXE=${CONDA_EXE:-/data1/hewenxuan/miniconda3/bin/conda}
ENV_NAME=${ENV_NAME:-3dgs-imc}
MH_REPO=${MH_REPO:-/data1/zhangao/repos/MH-3DGS}
SCENE=${SCENE:-counter}
SOURCE_PATH=${SOURCE_PATH:-/data1/zhangao/DATA/mipnerf360/${SCENE}}
TEST_SELECTION=${TEST_SELECTION:-${SOURCE_PATH}/test_selection_angle_80.0.txt}
IMAGES=${IMAGES:-images_4}
RESOLUTION=${RESOLUTION:-1}
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/outputs/mh_3dgs/${SCENE}_angle80_i30000_${IMAGES}_aa_fix}
PORT=${PORT:-12345}

mkdir -p "${MODEL_PATH}"
cd "${MH_REPO}"

/usr/bin/time -f "Total training time: %e seconds" -o "${MODEL_PATH}/training_time.txt" \
"${CONDA_EXE}" run --no-capture-output -n "${ENV_NAME}" python train.py \
    --source_path "${SOURCE_PATH}" \
    --model_path "${MODEL_PATH}" \
    --test_selection_path "${TEST_SELECTION}" \
    --images "${IMAGES}" \
    --resolution "${RESOLUTION}" \
    --data_device cpu \
    --port "${PORT}"

"${CONDA_EXE}" run --no-capture-output -n "${ENV_NAME}" python render.py \
    --model_path "${MODEL_PATH}" \
    --iteration -1 \
    --skip_train

"${CONDA_EXE}" run --no-capture-output -n "${ENV_NAME}" python metrics.py \
    --model_paths "${MODEL_PATH}" \
    --metrics_save_path "${MODEL_PATH}/metrics.csv"
