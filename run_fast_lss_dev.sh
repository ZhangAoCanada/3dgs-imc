#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}


if [[ "${SKIP_UVM_PREFLIGHT:-0}" != "1" ]]; then
    python3 scripts/check_nvidia_uvm_state.py
fi

CONDA_EXE=${CONDA_EXE:-/data1/hewenxuan/miniconda3/bin/conda}
ENV_NAME=${ENV_NAME:-3dgs-imc}

SCENE=${SCENE:-counter}
SOURCE_PATH=${SOURCE_PATH:-/data1/zhangao/DATA/mipnerf360/${SCENE}}
TEST_SELECTION=${TEST_SELECTION:-${SOURCE_PATH}/test_selection_angle_80.0.txt}
MODEL_ROOT=${MODEL_ROOT:-outputs/fast_lss_dev}
ITERATIONS=${ITERATIONS:-18000}
DENSIFY_FROM_ITER=${DENSIFY_FROM_ITER:-500}
DENSIFY_UNTIL_ITER=${DENSIFY_UNTIL_ITER:-12000}
DENSIFICATION_INTERVAL=${DENSIFICATION_INTERVAL:-400}
DEPTH_INIT_POINTS=${DEPTH_INIT_POINTS:-0}
DEPTH_INIT_STRIDE=${DEPTH_INIT_STRIDE:-32}
DEPTH_INIT_VIEWS=${DEPTH_INIT_VIEWS:-0}
DEPTH_L1_WEIGHT_INIT=${DEPTH_L1_WEIGHT_INIT:-1.0}
DEPTH_L1_WEIGHT_FINAL=${DEPTH_L1_WEIGHT_FINAL:-0.01}
DEPTH_L1_STOP_ITER=${DEPTH_L1_STOP_ITER:-0}
DEPTH_ALIGN_SFM_ON_DEMAND=${DEPTH_ALIGN_SFM_ON_DEMAND:-1}
RANDOM_VIEW_NUM=${RANDOM_VIEW_NUM:-1}
CPF_NOISE_WEIGHT=${CPF_NOISE_WEIGHT:-0.5}
CPF_MAX_DRIFT_NORM=${CPF_MAX_DRIFT_NORM:-0.0005}
CPF_MAX_NOISE_NORM=${CPF_MAX_NOISE_NORM:-0.0005}
CPF_STOP_ITER=${CPF_STOP_ITER:-0}
CPF_REFRESH_STOP_ITER=${CPF_REFRESH_STOP_ITER:-0}
CPF_TRAIN_STOP_ITER=${CPF_TRAIN_STOP_ITER:-0}
CPF_UPDATE_LOSS_THRESHOLD=${CPF_UPDATE_LOSS_THRESHOLD:-0}
CPF_LOSS_WEIGHT=${CPF_LOSS_WEIGHT:-1.0}
CPF_LOSS_CAP=${CPF_LOSS_CAP:-0}
CPF_RELIABILITY_GATE=${CPF_RELIABILITY_GATE:-0}
CPF_RELIABILITY_LOSS_RATIO=${CPF_RELIABILITY_LOSS_RATIO:-4.0}
CPF_RELIABILITY_MIN_LOSS=${CPF_RELIABILITY_MIN_LOSS:-30.0}
CPF_RELIABILITY_EMA_DECAY=${CPF_RELIABILITY_EMA_DECAY:-0.95}
CPF_PRIOR_SOURCE=${CPF_PRIOR_SOURCE:-sfm_depth}
CPF_MAX_POINTS=${CPF_MAX_POINTS:-25000}
CPF_MAX_REFRESH_VIEWS=${CPF_MAX_REFRESH_VIEWS:-16}
CPF_UNIQUE_EVERY=${CPF_UNIQUE_EVERY:-8}
CPF_TRAIN_INTERVAL=${CPF_TRAIN_INTERVAL:-8}
CPF_DERIVATIVE_INTERVAL=${CPF_DERIVATIVE_INTERVAL:-8}
CPF_SFM_ALIGN_MAX_POINTS=${CPF_SFM_ALIGN_MAX_POINTS:-20000}
NN_START_ITER=${NN_START_ITER:-2500}
NN_DERIVATIVES_ITER=${NN_DERIVATIVES_ITER:-3000}
NN_UPDATE_INTERVAL=${NN_UPDATE_INTERVAL:-3000}
PARTIAL_SCALING=${PARTIAL_SCALING:-1e0}
SIGMA_SCALING=${SIGMA_SCALING:-1e-6}
CPF_DENSIFY=${CPF_DENSIFY:-0}
CPF_DENSIFY_START_ITER=${CPF_DENSIFY_START_ITER:-3000}
CPF_DENSIFY_UNTIL_ITER=${CPF_DENSIFY_UNTIL_ITER:-12000}
CPF_DENSIFY_POOL=${CPF_DENSIFY_POOL:-32768}
CPF_DENSIFY_DENSITY_POINTS=${CPF_DENSIFY_DENSITY_POINTS:-16384}
CPF_DENSIFY_SIGMA_WEIGHT=${CPF_DENSIFY_SIGMA_WEIGHT:-1.0}
CPF_DENSIFY_GAP_WEIGHT=${CPF_DENSIFY_GAP_WEIGHT:-0.5}
CPF_DENSIFY_RELOCATE=${CPF_DENSIFY_RELOCATE:-1}
CPF_DENSIFY_BIRTH=${CPF_DENSIFY_BIRTH:-1}
CPF_DENSIFY_BIRTH_RATIO=${CPF_DENSIFY_BIRTH_RATIO:-1.0}
CPF_DENSIFY_SIGMA_MIN=${CPF_DENSIFY_SIGMA_MIN:-0.0}
CPF_DENSIFY_GAP_MIN=${CPF_DENSIFY_GAP_MIN:-0.0}
CPF_DENSIFY_USE_RESIDUAL=${CPF_DENSIFY_USE_RESIDUAL:-0}
CPF_DENSIFY_GUIDED_CLONE=${CPF_DENSIFY_GUIDED_CLONE:-0}
CPF_DENSIFY_GUIDED_RATIO=${CPF_DENSIFY_GUIDED_RATIO:-1.0}
CPF_DENSIFY_GUIDED_GRAD_WEIGHT=${CPF_DENSIFY_GUIDED_GRAD_WEIGHT:-1.0}
CPF_DENSIFY_GUIDED_CPF_WEIGHT=${CPF_DENSIFY_GUIDED_CPF_WEIGHT:-0.75}
CPF_DENSIFY_GUIDED_DENSITY_WEIGHT=${CPF_DENSIFY_GUIDED_DENSITY_WEIGHT:-0.5}
CPF_DENSIFY_GUIDED_COARSE_RATIO=${CPF_DENSIFY_GUIDED_COARSE_RATIO:-0.25}
CPF_DENSIFY_GUIDED_FINE_SCALE=${CPF_DENSIFY_GUIDED_FINE_SCALE:-1.0}
CPF_DENSIFY_GUIDED_COARSE_SCALE=${CPF_DENSIFY_GUIDED_COARSE_SCALE:-5.0}
CPF_RESIDUAL_CACHE=${CPF_RESIDUAL_CACHE:-0}
CPF_RESIDUAL_CACHE_START_ITER=${CPF_RESIDUAL_CACHE_START_ITER:-1000}
CPF_RESIDUAL_CACHE_UNTIL_ITER=${CPF_RESIDUAL_CACHE_UNTIL_ITER:-12000}
CPF_RESIDUAL_CACHE_INTERVAL=${CPF_RESIDUAL_CACHE_INTERVAL:-200}
CPF_RESIDUAL_CACHE_PIXELS=${CPF_RESIDUAL_CACHE_PIXELS:-1024}
CPF_RESIDUAL_CACHE_MAX_POINTS=${CPF_RESIDUAL_CACHE_MAX_POINTS:-32768}
TIMING_BREAKDOWN=${TIMING_BREAKDOWN:-1}
TIMING_SYNC=${TIMING_SYNC:-0}
ANTIALIASING=${ANTIALIASING:-0}
NOISE_LR=${NOISE_LR:-5e5}
STOCHASTIC_STOP_ITER=${STOCHASTIC_STOP_ITER:-0}
CAP_MAX=${CAP_MAX:-1000000}
LAMBDA_DSSIM=${LAMBDA_DSSIM:-0.2}
SCALE_REG=${SCALE_REG:-0.0}
OPACITY_REG=${OPACITY_REG:-0.0}
LOAD_ITERATION=${LOAD_ITERATION:-0}
LOAD_MODEL_PATH=${LOAD_MODEL_PATH:-}

MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/${SCENE}_angle80_lss_sfmdepth_i${ITERATIONS}_depth1_aligndepth_best_images4}

LOAD_ARGS=()
if [[ "${LOAD_ITERATION}" != "0" ]]; then
    LOAD_ARGS+=(--load_iteration "${LOAD_ITERATION}")
fi
if [[ -n "${LOAD_MODEL_PATH}" ]]; then
    LOAD_ARGS+=(--load_model_path "${LOAD_MODEL_PATH}")
fi

CPF_DENSIFY_ARGS=()
if [[ "${CPF_DENSIFY}" == "1" ]]; then
    CPF_DENSIFY_ARGS+=(--cpf_densify)
fi

TIMING_ARGS=()
if [[ "${TIMING_BREAKDOWN}" == "1" ]]; then
    TIMING_ARGS+=(--timing_breakdown)
fi
if [[ "${TIMING_SYNC}" == "1" ]]; then
    TIMING_ARGS+=(--timing_sync)
fi

PIPELINE_ARGS=()
if [[ "${ANTIALIASING}" == "1" ]]; then
    PIPELINE_ARGS+=(--antialiasing)
fi

"${CONDA_EXE}" run -n "${ENV_NAME}" python train_experimental_branch3_mc.py \
    --source_path "${SOURCE_PATH}" \
    --test_selection_path "${TEST_SELECTION}" \
    --model_path "${MODEL_PATH}" \
    "${LOAD_ARGS[@]}" \
    --images images_4 \
    --depths dav2_4 \
    --resolution 1 \
    --data_device cpu \
    --init_type sfm \
    --cap_max "${CAP_MAX}" \
    --depth_init_points "${DEPTH_INIT_POINTS}" \
    --depth_init_stride "${DEPTH_INIT_STRIDE}" \
    --depth_init_views "${DEPTH_INIT_VIEWS}" \
    --iterations "${ITERATIONS}" \
    --position_lr_max_steps "${ITERATIONS}" \
    --densify_from_iter "${DENSIFY_FROM_ITER}" \
    --densify_until_iter "${DENSIFY_UNTIL_ITER}" \
    --densification_interval "${DENSIFICATION_INTERVAL}" \
    --depth_l1_weight_init "${DEPTH_L1_WEIGHT_INIT}" \
    --depth_l1_weight_final "${DEPTH_L1_WEIGHT_FINAL}" \
    --depth_l1_stop_iter "${DEPTH_L1_STOP_ITER}" \
    --depth_align_sfm_on_demand "${DEPTH_ALIGN_SFM_ON_DEMAND}" \
    --scale_reg "${SCALE_REG}" \
    --opacity_reg "${OPACITY_REG}" \
    --lambda_dssim "${LAMBDA_DSSIM}" \
    --noise_lr "${NOISE_LR}" \
    --stochastic_stop_iter "${STOCHASTIC_STOP_ITER}" \
    --nn_type allviews \
    --bound_type local \
    --partial_scaling "${PARTIAL_SCALING}" \
    --sigma_scaling "${SIGMA_SCALING}" \
    --minmax wholeonly \
    --range_scale 1 \
    --noise_method sigma-detach \
    --random_view_num "${RANDOM_VIEW_NUM}" \
    --nn_start_iter "${NN_START_ITER}" \
    --nn_derivatives_iter "${NN_DERIVATIVES_ITER}" \
    --nn_update_interval "${NN_UPDATE_INTERVAL}" \
    --cpf_prior_source "${CPF_PRIOR_SOURCE}" \
    --cpf_max_refresh_views "${CPF_MAX_REFRESH_VIEWS}" \
    --cpf_unique_every "${CPF_UNIQUE_EVERY}" \
    --cpf_max_points "${CPF_MAX_POINTS}" \
    --cpf_sfm_align_max_points "${CPF_SFM_ALIGN_MAX_POINTS}" \
    --cpf_train_interval "${CPF_TRAIN_INTERVAL}" \
    --cpf_derivative_interval "${CPF_DERIVATIVE_INTERVAL}" \
    --cpf_stop_iter "${CPF_STOP_ITER}" \
    --cpf_refresh_stop_iter "${CPF_REFRESH_STOP_ITER}" \
    --cpf_train_stop_iter "${CPF_TRAIN_STOP_ITER}" \
    --cpf_update_loss_threshold "${CPF_UPDATE_LOSS_THRESHOLD}" \
    --cpf_loss_weight "${CPF_LOSS_WEIGHT}" \
    --cpf_loss_cap "${CPF_LOSS_CAP}" \
    --cpf_reliability_gate "${CPF_RELIABILITY_GATE}" \
    --cpf_reliability_loss_ratio "${CPF_RELIABILITY_LOSS_RATIO}" \
    --cpf_reliability_min_loss "${CPF_RELIABILITY_MIN_LOSS}" \
    --cpf_reliability_ema_decay "${CPF_RELIABILITY_EMA_DECAY}" \
    "${CPF_DENSIFY_ARGS[@]}" \
    --cpf_densify_start_iter "${CPF_DENSIFY_START_ITER}" \
    --cpf_densify_until_iter "${CPF_DENSIFY_UNTIL_ITER}" \
    --cpf_densify_pool "${CPF_DENSIFY_POOL}" \
    --cpf_densify_density_points "${CPF_DENSIFY_DENSITY_POINTS}" \
    --cpf_densify_sigma_weight "${CPF_DENSIFY_SIGMA_WEIGHT}" \
    --cpf_densify_gap_weight "${CPF_DENSIFY_GAP_WEIGHT}" \
    --cpf_densify_relocate "${CPF_DENSIFY_RELOCATE}" \
    --cpf_densify_birth "${CPF_DENSIFY_BIRTH}" \
    --cpf_densify_birth_ratio "${CPF_DENSIFY_BIRTH_RATIO}" \
    --cpf_densify_sigma_min "${CPF_DENSIFY_SIGMA_MIN}" \
    --cpf_densify_gap_min "${CPF_DENSIFY_GAP_MIN}" \
    --cpf_densify_use_residual "${CPF_DENSIFY_USE_RESIDUAL}" \
    --cpf_densify_guided_clone "${CPF_DENSIFY_GUIDED_CLONE}" \
    --cpf_densify_guided_ratio "${CPF_DENSIFY_GUIDED_RATIO}" \
    --cpf_densify_guided_grad_weight "${CPF_DENSIFY_GUIDED_GRAD_WEIGHT}" \
    --cpf_densify_guided_cpf_weight "${CPF_DENSIFY_GUIDED_CPF_WEIGHT}" \
    --cpf_densify_guided_density_weight "${CPF_DENSIFY_GUIDED_DENSITY_WEIGHT}" \
    --cpf_densify_guided_coarse_ratio "${CPF_DENSIFY_GUIDED_COARSE_RATIO}" \
    --cpf_densify_guided_fine_scale "${CPF_DENSIFY_GUIDED_FINE_SCALE}" \
    --cpf_densify_guided_coarse_scale "${CPF_DENSIFY_GUIDED_COARSE_SCALE}" \
    --cpf_residual_cache "${CPF_RESIDUAL_CACHE}" \
    --cpf_residual_cache_start_iter "${CPF_RESIDUAL_CACHE_START_ITER}" \
    --cpf_residual_cache_until_iter "${CPF_RESIDUAL_CACHE_UNTIL_ITER}" \
    --cpf_residual_cache_interval "${CPF_RESIDUAL_CACHE_INTERVAL}" \
    --cpf_residual_cache_pixels "${CPF_RESIDUAL_CACHE_PIXELS}" \
    --cpf_residual_cache_max_points "${CPF_RESIDUAL_CACHE_MAX_POINTS}" \
    --cpf_noise_weight "${CPF_NOISE_WEIGHT}" \
    --cpf_max_drift_norm "${CPF_MAX_DRIFT_NORM}" \
    --cpf_max_noise_norm "${CPF_MAX_NOISE_NORM}" \
    --dds_ratio 2 \
    --dvoxel_size 0.002 \
    --eps 0.25 \
    --min_samples 500 \
    --test_iterations 99999999 \
    --save_iterations "${ITERATIONS}" \
    "${TIMING_ARGS[@]}" \
    "${PIPELINE_ARGS[@]}" \
    --disable_viewer \
    --quiet

if [[ "${RUN_EVAL:-1}" == "1" ]]; then
    "${CONDA_EXE}" run -n "${ENV_NAME}" python render.py \
        --model_path "${MODEL_PATH}" \
        --iteration "${ITERATIONS}" \
        --skip_train \
        "${PIPELINE_ARGS[@]}" \
        --quiet

    "${CONDA_EXE}" run -n "${ENV_NAME}" python metrics.py \
        --model_paths "${MODEL_PATH}" \
        --metrics_save_path "${MODEL_PATH}/metrics.csv"
fi
