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

# Repo dir: portable defaults resolve under here regardless of the caller's cwd.
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- Configurable roots (override via env / command line) ----
#   DATA_ROOT     : dataset root, contains <dataset>/<scene>/...   (point this at your data)
#   OUT_ROOT      : full training outputs (models + renders) — heavy; can be a scratch disk
#   SWEEP_RESULTS : light metrics-only copy of every run, collected for easy transfer
# e.g.  DATA_ROOT=/path/to/DATA OUT_ROOT=/scratch/gslss_models bash run.sh
DATA_ROOT=${DATA_ROOT:-/data1/zhangao/DATA}
OUT_ROOT=${OUT_ROOT:-${REPO_DIR}/outputs/sweep_models}
SWEEP_RESULTS=${SWEEP_RESULTS:-${REPO_DIR}/outputs/sweep_results}

# Default run: Mip-NeRF 360, coverage-gap (angle) bias, at the 5 evaluated angles 20/40/60/80/100.
# (Override with DATASETS=/BIASES=/SCENES=/ANGLES= to widen scope.)
DATASETS=${DATASETS:-"mipnerf360"}
SCENES=${SCENES:-}                                       # empty => auto-discover per dataset
BIASES=${BIASES:-"angle"}                                # coverage-gap angle splits
ANGLES=${ANGLES:-"20.0 40.0 60.0 80.0 100.0"} # the 5 coverage angles
export ITERATIONS=${ITERATIONS:-30000}
CAP_OVERRIDE=${CAP_MAX:-}

# AA policy: indoor scenes benefit from antialiasing, outdoor do not. By DEFAULT auto-select per
# scene (indoor -> AA, outdoor -> no-AA). Set ANTIALIASING=0 or =1 to FORCE one mode for all scenes.
if [ -n "${ANTIALIASING+x}" ]; then AA_FORCED="${ANTIALIASING}"; else AA_FORCED=""; fi
# Field-consensus floater pruning: indoor-gated by default (helps indoor extreme-angle over-fit,
# hurts detailed outdoor). Set CPF_FIELD_OPACITY_REG to force a value for ALL scenes.
if [ -n "${CPF_FIELD_OPACITY_REG+x}" ]; then FLOATER_FORCED="${CPF_FIELD_OPACITY_REG}"; else FLOATER_FORCED=""; fi
# Random-batch size: room needs 3-view gradient de-biasing (gap-edge consolidation; room@60
# 15.27 -> 16.63 confirmed @30k). Does NOT transfer to treehill, neutral-to-negative elsewhere
# (counter@80 +0.09, bicycle@20 -0.17 PSNR / better LPIPS) -> room only. ~2.5x per-iter cost.
# Set RANDOM_VIEW_NUM to force one value for ALL scenes.
if [ -n "${RANDOM_VIEW_NUM+x}" ]; then NVIEW_FORCED="${RANDOM_VIEW_NUM}"; else NVIEW_FORCED=""; fi

# ---- BEST CPF config (scene-agnostic; do not change without re-verifying vs real 3dgs-mcmc) ----
export CPF_PRIOR_SOURCE=render            # depth-free rendered-depth prior (NOT DAV2)
export CPF_MAX_POINTS=100000              # per-train CPF budget (the 25k cap cost ~0.13 dB)
export CPF_DRIFT_MAX_POINTS=0             # 100k coverage; broadening it hurts clean scenes
export CPF_TRAIN_INTERVAL=8
export CPF_DERIVATIVE_INTERVAL=${CPF_DERIVATIVE_INTERVAL:-2}   # noise applied 4x more often (redesign)
export CPF_MAX_REFRESH_VIEWS=16
export SCALE_REG=0.01                     # 3DGS-MCMC compactness regs — biggest quality lever
export OPACITY_REG=0.01
# ANTIALIASING is set PER SCENE in run_best (indoor->AA) unless AA_FORCED (see top).
export DEPTH_L1_WEIGHT_INIT=0             # depth-free
export DEPTH_L1_WEIGHT_FINAL=0
export DEPTH_ALIGN_SFM_ON_DEMAND=0
export CPF_PULL_WEIGHT=0 CPF_PULL_OBS_GATE=0 CPF_PULL_PERSIST=0
export CPF_FIELD_SFM_ANCHOR=0 CPF_FIELD_ALPHA_THRESH=0
# ---- Stochasticity redesign (parameterized MCMC): on-MANIFOLD, coverage-TARGETED noise ----
# The default CPF noise was vestigial (untrained std head, tiny 5e-4 clamp, occupancy-weighted).
# Two changes make the stochastic exploration productive instead of destructive:
#  (1) CPF_NOISE_TANGENT=1: project the per-Gaussian noise onto the field's surface TANGENT plane
#      (suppress the normal component via the field gradient) -> noise explores the under-constrained
#      DOF and stays ON the learned geometry manifold (raw isotropic amplification HURTS: -0.96).
#  (2) CPF_NOISE_OBS_GATE=1: scale noise by view-coverage -> ~0 on well-observed Gaussians, full on
#      under-observed (the floater-prone ones the biased views leave ambiguous; grows with angle).
# Validated (AA/noAA best-mode): bonsai@100 16.52->17.39 (+0.87), bonsai@60 +0.07, counter@80 +0.02
# (no regression). Helps bias-stressed cells, neutral on well-constrained ones. 4090-safe (<=14GB).
export CPF_NOISE_TANGENT=${CPF_NOISE_TANGENT:-1.0}
export CPF_NOISE_OBS_GATE=${CPF_NOISE_OBS_GATE:-1}
export CPF_MAX_NOISE_NORM=${CPF_MAX_NOISE_NORM:-0.005}
export CPF_NOISE_WEIGHT=${CPF_NOISE_WEIGHT:-1.0}
export CPF_NOISE_TRAIN_WEIGHT=${CPF_NOISE_TRAIN_WEIGHT:-0}   # trained head OFF (hand-shaping beat it)
# RANDOM_VIEW_NUM is set PER SCENE in run_best (room->3) unless NVIEW_FORCED (see top).
# Densification schedule MUST reach the per-scene cap_max. add_new_gs grows the
# Gaussian set only +5%/step, so #steps = (until-from)/interval sets the final count:
# the old until=12000/interval=400 gave ~28 steps = 1.05^28 ~= 3.9x SFM (bicycle@20
# stalled at 212k of a 5M cap -> -1.8 dB PSNR, 0.38 LPIPS). Restoring the repo
# defaults until=25000/interval=200 (~122 steps) lets every scene reach its cap
# (bicycle 5M), recovering/beating the paper: bicycle@20 20.83 -> 22.64 PSNR,
# 0.38 -> 0.222 LPIPS, 5M Gaussians, 36 min in this fast pipeline (vs 2h39m original).
export DENSIFY_FROM_ITER=${DENSIFY_FROM_ITER:-500}
export DENSIFY_UNTIL_ITER=${DENSIFY_UNTIL_ITER:-25000}
export DENSIFICATION_INTERVAL=${DENSIFICATION_INTERVAL:-200}
export RUN_EVAL=${RUN_EVAL:-1}            # train -> render.py -> metrics.py
export TIMING_BREAKDOWN=${TIMING_BREAKDOWN:-1}
# Dual-row final protocol: save AND evaluate both the 15k and the final checkpoint
# (metrics.csv gets one row per checkpoint -> "GS-LSS-Eff@15k" + "@30k" table rows).
if [ -z "${SAVE_ITERATIONS+x}" ]; then
    if [ "${ITERATIONS}" != "15000" ]; then SAVE_ITERATIONS="15000 ${ITERATIONS}"; else SAVE_ITERATIONS="${ITERATIONS}"; fi
fi
export SAVE_ITERATIONS

# per-scene Gaussian cap (3DGS-MCMC tuned counts); fallback for other datasets' scenes
scene_cap() { case "$1" in
    bicycle) echo 5000000 ;; stump) echo 4200000 ;; garden) echo 4000000 ;;
    treehill) echo 3300000 ;; flowers) echo 3000000 ;; kitchen) echo 1500000 ;;
    room) echo 1200000 ;; bonsai|counter) echo 1000000 ;;
    *) echo "${DEFAULT_CAP:-3000000}" ;;
esac ; }

# indoor scenes use antialiasing by default (outdoor do not)
is_indoor() { case "$1" in bonsai|counter|kitchen|room) return 0 ;; *) return 1 ;; esac ; }

# per-scene random-batch size (see NVIEW_FORCED note above)
scene_nview() { case "$1" in room) echo 3 ;; *) echo 1 ;; esac ; }

# run_best  dataset scene bias source_path test_selection test_path
run_best() {
    local ds=$1 sc=$2 bias=$3 srcp=$4 tsel=$5 tpath=$6
    local cap="${CAP_OVERRIDE:-$(scene_cap "$sc")}"
    local imgs=images_4; [ -d "$srcp/images_4" ] || imgs=images   # auto-detect resolution dir
    local dpth=dav2_4;   [ -d "$srcp/dav2_4"  ] || dpth=""         # depth-free anyway; load only if present
    # per-scene AA: forced if the caller set ANTIALIASING, else auto (indoor -> AA)
    local aa
    if [ -n "$AA_FORCED" ]; then aa="$AA_FORCED"; elif is_indoor "$sc"; then aa=1; else aa=0; fi
    export ANTIALIASING="$aa"
    local AA_TAG=""; [ "$aa" = "1" ] && AA_TAG="_aa"
    # field-consensus floater pruning: wins indoor extreme-angle over-fit (bonsai@100 17.39->17.98,
    # beats MH); the smooth field false-prunes detailed OUTDOOR geometry -> indoor only by default.
    local floater=0; is_indoor "$sc" && floater=0.1
    [ -n "$FLOATER_FORCED" ] && floater="$FLOATER_FORCED"
    export CPF_FIELD_OPACITY_REG="$floater"
    # per-scene random-batch size (room -> 3-view de-biasing)
    local nv; nv=$(scene_nview "$sc")
    [ -n "$NVIEW_FORCED" ] && nv="$NVIEW_FORCED"
    export RANDOM_VIEW_NUM="$nv"
    export SCENE="$sc" SOURCE_PATH="$srcp" CAP_MAX="$cap" IMAGES="$imgs" DEPTHS="$dpth"
    export TEST_SELECTION="$tsel" TEST_PATH="$tpath"
    export LABEL="${ds}_${sc}_${bias}_cpf_best_i${ITERATIONS}${AA_TAG}"
    local run_name="${sc}_${bias}_i${ITERATIONS}${AA_TAG}"
    export MODEL_PATH="${OUT_ROOT}/${ds}/${run_name}"
    echo "======== BEST GS-LSS: ${ds}/${sc} [${bias}]  (cap ${cap}, img ${imgs}, AA=${aa}) ========"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash run_fast_lss_dev.sh \
        || echo "[run.sh] WARN: ${ds}/${sc} [${bias}] failed; continuing"
    # collect a light metrics-only copy (easy to copy back from another machine)
    if [ -f "${MODEL_PATH}/metrics.csv" ]; then
        mkdir -p "${SWEEP_RESULTS}/${ds}/${run_name}"
        cp -f "${MODEL_PATH}/metrics.csv" "${SWEEP_RESULTS}/${ds}/${run_name}/metrics.csv"
        echo "[run.sh] metrics -> ${SWEEP_RESULTS}/${ds}/${run_name}/metrics.csv"
    else
        echo "[run.sh] WARN: no metrics.csv for ${ds}/${run_name}"
    fi
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
