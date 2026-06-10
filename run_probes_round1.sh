#!/usr/bin/env bash
# Round-1 design-level probes (GPU 1, sequential). Each probe isolates one suspected
# design blocker vs the locked best-stage config:
#   P1 room@60   outdoor-treatment (AA=0, floater=0)  -> is the indoor gate wrong for cluttered room?
#   P2 room@60   AA=1, floater=0                      -> bisect: floater harm vs AA harm
#   P3 bonsai@100 locked + CPF_STOP_ITER=25000        -> does annealing CPF stochasticity lift the mean?
#   P4 bonsai@100 locked + CPF_NOISE_UNIT_STD=1       -> does removing the frozen random std head help?
# References: room@60 locked=15.27 (v1), old-best 17.73; bonsai@100 locked=17.42/16.18 (v1/v2), old-best 18.05.
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex

( export SCENE=room  ANGLE=60.0  TAG=noaa_nofloater        CAP_MAX=1200000 ANTIALIASING=0 CPF_FIELD_OPACITY_REG=0;   bash diag_run.sh ) || echo "[probe] P1 failed"
( export SCENE=room  ANGLE=60.0  TAG=aa_nofloater          CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0;   bash diag_run.sh ) || echo "[probe] P2 failed"
( export SCENE=bonsai ANGLE=100.0 TAG=aa_floater_cpfstop25k CAP_MAX=1000000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1 CPF_STOP_ITER=25000;    bash diag_run.sh ) || echo "[probe] P3 failed"
( export SCENE=bonsai ANGLE=100.0 TAG=aa_floater_unitstd    CAP_MAX=1000000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1 CPF_NOISE_UNIT_STD=1;   bash diag_run.sh ) || echo "[probe] P4 failed"

echo "================ ROUND-1 PROBE RESULTS ================"
for run in room_angle_60.0_noaa_nofloater room_angle_60.0_aa_nofloater bonsai_angle_100.0_aa_floater_cpfstop25k bonsai_angle_100.0_aa_floater_unitstd; do
    f="/data1/zhangao/outputs/diag/mipnerf360/${run}_i30000/metrics.csv"
    echo "== ${run}"
    cat "$f" 2>/dev/null || echo "  (missing)"
done
