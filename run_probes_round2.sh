#!/usr/bin/env bash
# Round-2 probes: CPF freshness (the largest remaining fast-vs-slow config delta) on room@60.
#   R2a room@60 locked + NN_UPDATE_INTERVAL=1000            -> field 3x fresher
#   R2b room@60 locked + NN_UPDATE_INTERVAL=500 + 32 views  -> old-pipeline cadence mimic
#   R2c room@60 locked + EPS=0.5                            -> close out the last untested per-scene HP
# Refs: room@60 locked=15.27 (v1), round-1 best (AA0+floater0)=15.61, old slow-pipeline best=17.73.
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex

( export SCENE=room ANGLE=60.0 TAG=nnup1000 CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1 NN_UPDATE_INTERVAL=1000; bash diag_run.sh ) || echo "[probe] R2a failed"
( export SCENE=room ANGLE=60.0 TAG=nnup500_rv32 CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1 NN_UPDATE_INTERVAL=500 CPF_MAX_REFRESH_VIEWS=32; bash diag_run.sh ) || echo "[probe] R2b failed"
( export SCENE=room ANGLE=60.0 TAG=eps05 CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1 EPS=0.5; bash diag_run.sh ) || echo "[probe] R2c failed"

echo "================ ROUND-2 PROBE RESULTS ================"
for run in room_angle_60.0_nnup1000 room_angle_60.0_nnup500_rv32 room_angle_60.0_eps05; do
    f="/data1/zhangao/outputs/diag/mipnerf360/${run}_i30000/metrics.csv"
    echo "== ${run}"
    cat "$f" 2>/dev/null || echo "  (missing)"
done
