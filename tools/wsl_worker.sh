#!/usr/bin/env bash
# =============================================================================
# wsl_worker.sh — run this ON YOUR WSL machine (inside tmux/screen).
# Pull-based worker: the remote dev server cannot reach your WSL, so this loop
#   1. rsyncs the latest training source FROM the server (only .py/.sh/config dirs),
#   2. pulls job files from the server queue  (outputs/wsl_queue/pending/*.job),
#   3. runs each job on the LOCAL GPU (diag probe or run.sh final cell),
#   4. rsyncs results + logs BACK to the server (outputs/wsl_results/),
#   5. sleeps and repeats.
# One-time setup on WSL:
#   - passwordless ssh to the server:  ssh ${SERVER} echo ok        (must print ok)
#   - adjust the four paths below if they differ, then:  bash wsl_worker.sh
# =============================================================================
set -uo pipefail

# ---- EDIT THESE (defaults guessed from your previous WSL runs) ----
SERVER=${SERVER:?set SERVER, e.g. zhangao@<server-ip> (how you ssh to the dev server)}
SREPO=${SREPO:-/data1/zhangao/repos/3dgs-imc-codex}          # repo on the server
LREPO=${LREPO:-$HOME/Documents/wsl_repos/3dgs-imc-codex}     # repo on this WSL machine
LDATA=${LDATA:-$HOME/Documents/wsl_repos/DATA}               # mipnerf360 parent dir on WSL
LOUT=${LOUT:-$HOME/Documents/wsl_repos/outputs/wsl_lane}     # local outputs for lane work
POLL=${POLL:-120}                                            # queue poll seconds

mkdir -p "$LOUT" "$LREPO/outputs"

sync_code() {  # server -> WSL: only the source that changes during development
    rsync -az --timeout=60 \
        --include='*.py' --include='*.sh' \
        --include='scene_experimental_branch3/***' --include='scene_experimental/***' \
        --include='arguments/***' --include='utils/***' --include='gaussian_renderer/***' \
        --include='scripts/***' --include='lpipsPyTorch/***' \
        --exclude='*' \
        "$SERVER:$SREPO/" "$LREPO/" || echo "[worker] WARN code sync failed"
}

run_job() {  # $1 = local job file
    local jf=$1 name; name=$(basename "$jf" .job)
    echo "[worker $(date +%H:%M)] running $name"
    (
        set -a
        # shellcheck disable=SC1090
        source "$jf"            # provides KIND + all run env (SCENE/ANGLE/TAG/CAP_MAX/...)
        set +a
        cd "$LREPO"
        export CUDA_VISIBLE_DEVICES=${WSL_GPU:-0}
        export DATA_ROOT="$LDATA"
        if [ "${KIND:-diag}" = "run" ]; then
            # final-sweep cell via run.sh (dual checkpoints etc. per job env)
            OUT_ROOT="$LOUT/sweep_models" SWEEP_RESULTS="$LOUT/sweep_results" bash run.sh
        else
            OUT_ROOT="$LOUT/diag" bash diag_run.sh
        fi
    ) > "$LOUT/${name}.log" 2>&1
    local rc=$?
    # push results (metrics + light files only) and mark the job done/failed on the server
    rsync -az --timeout=120 --include='*/' --include='metrics.csv' --include='*.log' --exclude='*' \
        "$LOUT/" "$SERVER:$SREPO/outputs/wsl_results/" || echo "[worker] WARN result sync failed"
    if [ $rc -eq 0 ]; then
        ssh "$SERVER" "mv '$SREPO/outputs/wsl_queue/pending/${name}.job' '$SREPO/outputs/wsl_queue/done/' 2>/dev/null"
    else
        ssh "$SERVER" "mv '$SREPO/outputs/wsl_queue/pending/${name}.job' '$SREPO/outputs/wsl_queue/failed/' 2>/dev/null"
    fi
    echo "[worker $(date +%H:%M)] $name finished rc=$rc"
}

echo "[worker] starting; polling $SERVER:$SREPO/outputs/wsl_queue/pending every ${POLL}s"
while true; do
    sync_code
    mkdir -p "$LOUT/jobs"
    rsync -az --timeout=60 "$SERVER:$SREPO/outputs/wsl_queue/pending/" "$LOUT/jobs/" 2>/dev/null || true
    found=0
    for jf in "$LOUT"/jobs/*.job; do
        [ -e "$jf" ] || continue
        found=1
        run_job "$jf"
        rm -f "$jf"
    done
    [ "$found" = "0" ] && sleep "$POLL"
done
