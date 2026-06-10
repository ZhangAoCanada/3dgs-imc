#!/usr/bin/env bash
# Round-3 probes (room@60): test the "gap-edge under-consolidation" hypothesis from the
# opacity-mass forensics (fast model has 2.5x less mass at train-vis<=2 than old slow model).
#   R3a noise-light : revert CPF noise to pre-redesign strength (tangent off, obs-gate off,
#                     clamp 5e-4, every 8 iters, weight 0.5) -> does removing the constant
#                     churn on low-obs Gaussians let the gap edge consolidate?
#   R3b nview3      : RANDOM_VIEW_NUM=3 alone (no depth) -> the paper's random-batch gradient
#                     de-biasing, never tested alone in the fast pipeline (~2.5x runtime).
# Refs: room@60 locked=15.27 (v1) / 15.35 (AA,nofloater) / 15.61 (AA0,floater0); old slow 17.27-17.73.
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex

( export SCENE=room ANGLE=60.0 TAG=noiselight CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1 \
         CPF_NOISE_TANGENT=0 CPF_NOISE_OBS_GATE=0 CPF_MAX_NOISE_NORM=0.0005 CPF_DERIVATIVE_INTERVAL=8 CPF_NOISE_WEIGHT=0.5; \
  bash diag_run.sh ) || echo "[probe] R3a failed"
( export SCENE=room ANGLE=60.0 TAG=nview3 CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1 RANDOM_VIEW_NUM=3; \
  bash diag_run.sh ) || echo "[probe] R3b failed"

echo "================ ROUND-3 PROBE RESULTS ================"
for run in room_angle_60.0_noiselight room_angle_60.0_nview3; do
    f="/data1/zhangao/outputs/diag/mipnerf360/${run}_i30000/metrics.csv"
    echo "== ${run}"
    cat "$f" 2>/dev/null || echo "  (missing)"
done
