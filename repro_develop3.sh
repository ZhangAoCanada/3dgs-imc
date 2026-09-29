#!/usr/bin/env bash
# repro_develop3.sh — faithfully reproduce a develop3 (paper-good) GS-LSS run using
# the ORIGINAL train_experimental_branch3.py, to confirm the target is reachable and
# whether the train_experimental_branch3_mc.py refactor is the regression. GPU 1 only.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}   # respect caller (server GPU1 / WSL GPU0)
CONDA_EXE=${CONDA_EXE:-/data1/hewenxuan/miniconda3/bin/conda}
ENV_NAME=${ENV_NAME:-3dgs-imc}

SCENE=${SCENE:?}; ANGLE=${ANGLE:-0}; CAP_MAX=${CAP_MAX:?}   # ANGLE only feeds the test_selection default; traintest (Table 3) sets TEST_PATH instead
DATASET=${DATASET:-mipnerf360}
SRC=${SOURCE_PATH:-${DATA_ROOT:-/data1/zhangao/DATA}/${DATASET}/${SCENE}}   # portable: diag_run exports SOURCE_PATH per-machine (server /data1, WSL /mnt/c)
OUT_ROOT=${OUT_ROOT:-/data1/zhangao/outputs/repro}
ITER=${ITER:-30000}
AA=${AA:-0}                                  # 0 = woAA (target 22.38), 1 = AA
MODEL_PATH=${MODEL_PATH:-${OUT_ROOT}/${DATASET}/${SCENE}_angle_${ANGLE}_develop3repro_aa${AA}_i${ITER}}

AA_ARG=(); [ "$AA" = "1" ] && AA_ARG=(--antialiasing)
# Table 3 (aerial-ground): when TEST_PATH (a test/ dir) is set, evaluate on it via --test_path;
# else use the angle/discontinuous test_selection file. IMAGES/DEPTHS overridable (aerial uses full 'images').
TEST_ARG=(--test_selection_path "${TEST_SELECTION:-${SRC}/test_selection_angle_${ANGLE}.txt}")
[ -n "${TEST_PATH:-}" ] && TEST_ARG=(--test_path "${TEST_PATH}")

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${CONDA_EXE}" run -n "${ENV_NAME}" python train_experimental_branch3.py \
    --source_path "${SRC}" \
    "${TEST_ARG[@]}" \
    --model_path "${MODEL_PATH}" \
    --images "${IMAGES:-images_4}" \
    --depths "${DEPTHS:-dav2_4}" \
    --resolution 1 \
    --data_device cpu \
    --init_type sfm \
    --cap_max "${CAP_MAX}" \
    --scale_reg 0.01 \
    --opacity_reg 0.01 \
    --noise_lr 5e5 \
    --densify_from_iter 500 \
    --densification_interval 200 \
    --depth_l1_weight_init "${DEPTH_INIT:-0.3}" \
    --depth_l1_weight_final "${DEPTH_FINAL:-0.003}" \
    --nn_type allviews \
    --bound_type local \
    --partial_scaling "${PARTIAL_SCALING:-1e0}" \
    --sigma_scaling "${SIGMA_SCALING:-1e-6}" \
    --eps "${EPS:-0.25}" \
    --min_samples 500 \
    --range_scale 1 \
    --minmax wholeonly \
    --noise_method sigma-detach \
    --random_view_num 3 \
    --nn_start_iter 2000 \
    --nn_derivatives_iter 2500 \
    --nn_update_interval 500 \
    --cpf_max_refresh_views "${CPF_MAX_REFRESH_VIEWS:-0}" \
    --dvoxel_size 0.01 \
    --iterations "${ITER}" \
    --save_iterations ${SAVE_ITERATIONS:-${ITER}} \
    --test_iterations 99999999 \
    "${AA_ARG[@]}" \
    --disable_viewer --quiet

for _it in ${SAVE_ITERATIONS:-${ITER}}; do   # render every saved checkpoint (dual-ckpt early-stop)
    "${CONDA_EXE}" run -n "${ENV_NAME}" python render.py \
        --model_path "${MODEL_PATH}" --iteration "${_it}" --skip_train "${AA_ARG[@]}" --quiet
done

"${CONDA_EXE}" run -n "${ENV_NAME}" python metrics.py \
    --model_paths "${MODEL_PATH}" --metrics_save_path "${MODEL_PATH}/metrics.csv"
# fail LOUDLY if the run produced no metrics (else seq_runner marks it done-without-metrics silently)
grep -q "ours_" "${MODEL_PATH}/metrics.csv" 2>/dev/null || { echo "[repro] FAILED: no ours_ metrics for ${MODEL_PATH}"; exit 1; }
echo "[repro] DONE ${MODEL_PATH}"
