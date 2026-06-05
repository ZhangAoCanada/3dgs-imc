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
MCMC_REPO=${MCMC_REPO:-/data1/zhangao/repos/3dgs-mcmc}
SCENE=${SCENE:-bonsai}
SOURCE_PATH=${SOURCE_PATH:-/data1/zhangao/DATA/mipnerf360/${SCENE}}
TEST_SELECTION=${TEST_SELECTION:-${SOURCE_PATH}/test_selection_angle_80.0.txt}
IMAGES=${IMAGES:-images_4}
RESOLUTION=${RESOLUTION:-1}
CAP_MAX=${CAP_MAX:-1300000}
MODEL_PATH=${MODEL_PATH:-/data1/zhangao/repos/3dgs-imc-codex/outputs/3dgs_mcmc/${SCENE}_angle80_default_${IMAGES}}

cd "${MCMC_REPO}"

mkdir -p "${MODEL_PATH}"
/usr/bin/time -f "Total training time: %e seconds" -o "${MODEL_PATH}/training_time.txt" \
"${CONDA_EXE}" run --no-capture-output -n "${ENV_NAME}" python train.py \
    --source_path "${SOURCE_PATH}" \
    --model_path "${MODEL_PATH}" \
    --images "${IMAGES}" \
    --test_selection_path "${TEST_SELECTION}" \
    --cap_max "${CAP_MAX}" \
    --resolution "${RESOLUTION}" \
    --data_device cpu \
    --init_type sfm

"${CONDA_EXE}" run --no-capture-output -n "${ENV_NAME}" python render.py \
    --model_path "${MODEL_PATH}" \
    --iteration -1 \
    --skip_train

"${CONDA_EXE}" run --no-capture-output -n "${ENV_NAME}" python metrics.py \
    --model_paths "${MODEL_PATH}" \
    --metrics_save_path "${MODEL_PATH}/metrics.csv"
