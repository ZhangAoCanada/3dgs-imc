#!/usr/bin/env bash
# =============================================================================
# run.sh — train ONLY the BEST GS-LSS (depth-free CPF) model across ALL
#          datasets, scenes, and BIAS SETUPS (same coverage as the old run.sh):
#            - angle / coverage-gap splits   (test_selection_angle_<A>.txt)
#            - discontinuous split           (discontinuous.txt)
#            - aerial-ground split           (train/ + test/ directory pair)
#
# VERIFIED best config beats the REAL 3dgs-mcmc baseline (counter angle-80, no-AA,
# matched cap 1M): GS-LSS 20.99 @18k vs 3dgs-mcmc 20.53 @30k = +0.46 PSNR, fewer iters.
# Method = depth-free rendered-depth CPF prior + 3DGS-MCMC compactness regs;
# every explored extra lever was within-noise or HURT -> all pinned OFF below.
# Each run does train -> render.py -> metrics.py (writes metrics.csv).
#
# Defaults: ALL datasets / scenes / bias setups. Restrict / extend, e.g.:
#   BIASES=angle SCENES=counter ANGLES=80.0 bash run.sh        # one scene, one split
#   BIASES=traintest DATASETS=large_angle bash run.sh          # only aerial-ground
#   BIASES=discontinuous DATASETS=mipnerf360 bash run.sh
#   ANTIALIASING=1 BIASES=angle SCENES=counter ANGLES=80.0 bash run.sh   # test AA (-> *_aa/)
#   CUDA_VISIBLE_DEVICES=0  ITERATIONS=18000  bash run.sh
# (legacy hand-tuned per-setup sweeps live in git history before this commit.)
# =============================================================================
set -uo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

DATA_ROOT=${DATA_ROOT:-/data1/zhangao/DATA}
DATASETS=${DATASETS:-"mipnerf360 tandt large_angle"}
SCENES=${SCENES:-}                                       # empty => auto-discover per dataset
BIASES=${BIASES:-"angle discontinuous traintest"}        # which bias setups to run
ANGLES=${ANGLES:-"10.0 20.0 40.0 60.0 80.0 100.0 120.0"} # angle splits (for the 'angle' bias)
export ITERATIONS=${ITERATIONS:-18000}
OUT_ROOT=${OUT_ROOT:-/data1/zhangao/outputs/new_dev}
CAP_OVERRIDE=${CAP_MAX:-}

# ---- BEST CPF config (scene-agnostic; do not change without re-verifying vs real 3dgs-mcmc) ----
export CPF_PRIOR_SOURCE=render            # depth-free rendered-depth prior (NOT DAV2)
export CPF_MAX_POINTS=100000              # per-train CPF budget (the 25k cap cost ~0.13 dB)
export CPF_DRIFT_MAX_POINTS=0             # 100k coverage; broadening it hurts clean scenes
export CPF_TRAIN_INTERVAL=8
export CPF_DERIVATIVE_INTERVAL=8
export CPF_MAX_REFRESH_VIEWS=16
export SCALE_REG=0.01                     # 3DGS-MCMC compactness regs — biggest quality lever
export OPACITY_REG=0.01
export ANTIALIASING=${ANTIALIASING:-0}    # default no-AA (AA hurt GS-LSS); ANTIALIASING=1 to test AA
export DEPTH_L1_WEIGHT_INIT=0             # depth-free
export DEPTH_L1_WEIGHT_FINAL=0
export DEPTH_ALIGN_SFM_ON_DEMAND=0
export CPF_PULL_WEIGHT=0 CPF_PULL_OBS_GATE=0 CPF_PULL_PERSIST=0
export CPF_FIELD_SFM_ANCHOR=0 CPF_FIELD_ALPHA_THRESH=0 CPF_NOISE_OBS_GATE=0
export RANDOM_VIEW_NUM=${RANDOM_VIEW_NUM:-1}
export DENSIFY_FROM_ITER=${DENSIFY_FROM_ITER:-500}
export DENSIFY_UNTIL_ITER=${DENSIFY_UNTIL_ITER:-12000}
export DENSIFICATION_INTERVAL=${DENSIFICATION_INTERVAL:-400}
export RUN_EVAL=${RUN_EVAL:-1}            # train -> render.py -> metrics.py
export TIMING_BREAKDOWN=${TIMING_BREAKDOWN:-1}

AA_TAG=""; [ "${ANTIALIASING}" = "1" ] && AA_TAG="_aa"

# per-scene Gaussian cap (3DGS-MCMC tuned counts); fallback for other datasets' scenes
scene_cap() { case "$1" in
    bicycle) echo 5000000 ;; stump) echo 4200000 ;; garden) echo 4000000 ;;
    treehill) echo 3300000 ;; flowers) echo 3000000 ;; kitchen) echo 1500000 ;;
    room) echo 1200000 ;; bonsai|counter) echo 1000000 ;;
    *) echo "${DEFAULT_CAP:-3000000}" ;;
esac ; }

# run_best  dataset scene bias source_path test_selection test_path
run_best() {
    local ds=$1 sc=$2 bias=$3 srcp=$4 tsel=$5 tpath=$6
    local cap="${CAP_OVERRIDE:-$(scene_cap "$sc")}"
    local imgs=images_4; [ -d "$srcp/images_4" ] || imgs=images   # auto-detect resolution dir
    local dpth=dav2_4;   [ -d "$srcp/dav2_4"  ] || dpth=""         # depth-free anyway; load only if present
    export SCENE="$sc" SOURCE_PATH="$srcp" CAP_MAX="$cap" IMAGES="$imgs" DEPTHS="$dpth"
    export TEST_SELECTION="$tsel" TEST_PATH="$tpath"
    export LABEL="${ds}_${sc}_${bias}_cpf_best_i${ITERATIONS}${AA_TAG}"
    export MODEL_PATH="${OUT_ROOT}/${ds}/${sc}_${bias}_i${ITERATIONS}${AA_TAG}"
    echo "======== BEST GS-LSS: ${ds}/${sc} [${bias}]  (cap ${cap}, img ${imgs}, AA=${ANTIALIASING}) ========"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash run_fast_lss_dev.sh \
        || echo "[run.sh] WARN: ${ds}/${sc} [${bias}] failed; continuing"
}

for dataset in $DATASETS; do
    droot="${DATA_ROOT}/${dataset}"
    [ -d "$droot" ] || { echo "[run.sh] skip dataset (missing): $droot"; continue; }
    scenes="$SCENES"
    [ -z "$scenes" ] && scenes=$(for d in "$droot"/*/; do [ -d "$d" ] && basename "$d"; done)
    for scene in $scenes; do
        src="${droot}/${scene}"
        [ -d "$src" ] || continue
        for bias in $BIASES; do
            case "$bias" in
                angle)
                    for a in $ANGLES; do
                        sp="${src}/test_selection_angle_${a}.txt"
                        [ -f "$sp" ] && run_best "$dataset" "$scene" "angle_${a}" "$src" "$sp" ""
                    done ;;
                discontinuous)
                    sp="${src}/discontinuous.txt"
                    [ -f "$sp" ] && run_best "$dataset" "$scene" "discontinuous" "$src" "$sp" "" ;;
                traintest)
                    [ -d "$src/train" ] && [ -d "$src/test" ] \
                        && run_best "$dataset" "$scene" "traintest" "$src/train" "" "$src/test" ;;
                *) echo "[run.sh] unknown bias '$bias' (use: angle discontinuous traintest)" ;;
            esac
        done
    done
done
echo "[run.sh] all requested (dataset x scene x bias) best-model runs complete."
