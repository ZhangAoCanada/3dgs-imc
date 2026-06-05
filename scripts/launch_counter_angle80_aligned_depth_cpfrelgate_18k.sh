#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=counter_angle80_lss_aligned_depth_cpfrelgate_i18000
MODEL_PATH=outputs/fast_lss_dev/counter_angle80_lss_sfmdepth_i18000_depth1_aligndepth_cpfrelgate_images4

exec scripts/launch_monitored_experiment.sh "${RUN_NAME}" \
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    SCENE=counter \
    ITERATIONS=18000 \
    MODEL_PATH="${MODEL_PATH}" \
    DEPTH_L1_WEIGHT_INIT=1.0 \
    DEPTH_L1_WEIGHT_FINAL=0.01 \
    DEPTH_ALIGN_SFM_ON_DEMAND=1 \
    CPF_RELIABILITY_GATE=1 \
    CPF_RELIABILITY_LOSS_RATIO=4.0 \
    CPF_RELIABILITY_MIN_LOSS=30.0 \
    CPF_RELIABILITY_EMA_DECAY=0.95 \
    TIMING_BREAKDOWN=1 \
    bash run_fast_lss_dev.sh
