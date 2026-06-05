#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

SCENES=${SCENES:-counter room kitchen}
METHODS=${METHODS:-lss_refine mcmc mh}
FORCE=${FORCE:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

export GPU=${GPU:-1}
export OUTPUT_STALL_TIMEOUT=${OUTPUT_STALL_TIMEOUT:-900}
export STALL_TIMEOUT=${STALL_TIMEOUT:-900}
export REQUIRE_GPU_AFTER=${REQUIRE_GPU_AFTER:-300}
export MAX_RUNTIME=${MAX_RUNTIME:-2700}

if [[ "${SKIP_DATA_PREFLIGHT:-0}" != "1" ]]; then
    python3 scripts/preflight_angle80_experiment.py --scenes ${SCENES} --methods ${METHODS}
fi

run_monitored() {
    local label=$1
    shift
    local log_path="outputs/monitor/${label}_${STAMP}.log"
    LABEL="${label}" "${SCRIPT_DIR}/launch_monitored_experiment.sh" "${log_path}" -- "$@"
}

for scene in ${SCENES}; do
    for method in ${METHODS}; do
        case "${method}" in
            lss_refine)
                model_path="outputs/fast_lss_dev/${scene}_angle80_lss_sfmdepth_i24000_depth1_refine18000_images4"
                if [[ "${FORCE}" != "1" && -f "${model_path}/metrics.csv" ]]; then
                    echo "skip existing ${method} ${scene}: ${model_path}/metrics.csv"
                    continue
                fi
                run_monitored "${scene}_angle80_lss_i24000_refine18000" \
                    env SCENE="${scene}" MODEL_PATH="${model_path}" ITERATIONS=24000 \
                    DEPTH_L1_WEIGHT_INIT=1.0 DEPTH_L1_WEIGHT_FINAL=0.01 DEPTH_L1_STOP_ITER=18000 \
                    CPF_REFRESH_STOP_ITER=18000 CPF_TRAIN_STOP_ITER=18000 \
                    CPF_STOP_ITER=18000 STOCHASTIC_STOP_ITER=18000 \
                    CPF_UPDATE_LOSS_THRESHOLD=0 RANDOM_VIEW_NUM=1 \
                    TIMING_BREAKDOWN=1 TIMING_SYNC=0 RUN_EVAL=1 bash run_fast_lss_dev.sh
                ;;
            lss_freeze)
                model_path="outputs/fast_lss_dev/${scene}_angle80_lss_sfmdepth_i24000_depth1_cpffreeze18000_images4"
                if [[ "${FORCE}" != "1" && -f "${model_path}/metrics.csv" ]]; then
                    echo "skip existing ${method} ${scene}: ${model_path}/metrics.csv"
                    continue
                fi
                run_monitored "${scene}_angle80_lss_i24000_cpffreeze18000" \
                    env SCENE="${scene}" MODEL_PATH="${model_path}" ITERATIONS=24000 \
                    DEPTH_L1_WEIGHT_INIT=1.0 DEPTH_L1_WEIGHT_FINAL=0.01 \
                    CPF_REFRESH_STOP_ITER=18000 CPF_TRAIN_STOP_ITER=18000 \
                    CPF_STOP_ITER=0 CPF_UPDATE_LOSS_THRESHOLD=0 \
                    RANDOM_VIEW_NUM=1 TIMING_BREAKDOWN=1 TIMING_SYNC=0 \
                    RUN_EVAL=1 bash run_fast_lss_dev.sh
                ;;
            lss_refreshstop)
                model_path="outputs/fast_lss_dev/${scene}_angle80_lss_sfmdepth_i24000_depth1_refreshstop18000_images4"
                if [[ "${FORCE}" != "1" && -f "${model_path}/metrics.csv" ]]; then
                    echo "skip existing ${method} ${scene}: ${model_path}/metrics.csv"
                    continue
                fi
                run_monitored "${scene}_angle80_lss_i24000_refreshstop18000" \
                    env SCENE="${scene}" MODEL_PATH="${model_path}" ITERATIONS=24000 \
                    DEPTH_L1_WEIGHT_INIT=1.0 DEPTH_L1_WEIGHT_FINAL=0.01 \
                    CPF_REFRESH_STOP_ITER=18000 CPF_TRAIN_STOP_ITER=0 \
                    CPF_STOP_ITER=0 CPF_UPDATE_LOSS_THRESHOLD=0 \
                    RANDOM_VIEW_NUM=1 TIMING_BREAKDOWN=1 TIMING_SYNC=0 \
                    RUN_EVAL=1 bash run_fast_lss_dev.sh
                ;;
            mcmc)
                model_path="outputs/3dgs_mcmc/${scene}_angle80_default_images_4"
                if [[ "${FORCE}" != "1" && -f "${model_path}/metrics.csv" ]]; then
                    echo "skip existing ${method} ${scene}: ${model_path}/metrics.csv"
                    continue
                fi
                run_monitored "${scene}_angle80_3dgs_mcmc_default" \
                    env SCENE="${scene}" MODEL_PATH="${REPO_ROOT}/${model_path}" IMAGES=images_4 \
                    bash scripts/run_3dgs_mcmc_angle80.sh
                ;;
            mh)
                model_path="outputs/mh_3dgs/${scene}_angle80_i30000_images_4_aa_fix"
                if [[ "${FORCE}" != "1" && -f "${model_path}/metrics.csv" ]]; then
                    echo "skip existing ${method} ${scene}: ${model_path}/metrics.csv"
                    continue
                fi
                run_monitored "${scene}_angle80_mh3dgs_i30000" \
                    env SCENE="${scene}" MODEL_PATH="${REPO_ROOT}/${model_path}" IMAGES=images_4 \
                    bash scripts/run_mh3dgs_angle80.sh
                ;;
            *)
                echo "unknown method: ${method}" >&2
                exit 2
                ;;
        esac
    done
done

python3 scripts/collect_experiment_results.py --out outputs/experiment_results_table.md
