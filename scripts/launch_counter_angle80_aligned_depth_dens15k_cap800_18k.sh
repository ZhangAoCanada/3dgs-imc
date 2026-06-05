#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=counter_angle80_lss_aligned_depth_dens15k_cap800_i18000
MODEL_PATH=outputs/fast_lss_dev/counter_angle80_lss_sfmdepth_i18000_depth1_aligndepth_dens15k_cap800_images4

exec scripts/launch_monitored_experiment.sh "${RUN_NAME}" \
    env SCENE=counter \
    ITERATIONS=18000 \
    MODEL_PATH="${MODEL_PATH}" \
    DEPTH_L1_WEIGHT_INIT=1.0 \
    DEPTH_L1_WEIGHT_FINAL=0.01 \
    DEPTH_ALIGN_SFM_ON_DEMAND=1 \
    DENSIFY_UNTIL_ITER=15000 \
    CAP_MAX=800000 \
    TIMING_BREAKDOWN=1 \
    bash run_fast_lss_dev.sh
