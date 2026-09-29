#!/usr/bin/env bash
# Autonomous sweep sentinel for the WSL lane. Every cycle it:
#   1. rsyncs WSL metrics.csv -> server mirror (so per-cell analysis always has fresh data)
#   2. re-queues WSL failed/ cells (transient D-state hangs) up to RETRY_MAX, state kept server-side
#   3. detects a D-state render hang (job running but GPU idle ~18min) and pkills the stuck python
#      (worker then moves it to failed/, step 2 re-queues it next cycle)
#   4. restarts the WSL probe_worker if it died while work remains
# Exits (waking the agent) when the WSL lane is fully drained. The SERVER lane is tracked
# independently by the seq_runner background task. Touches ONLY the WSL machine's own GPU 0
# and READS server GPU 1 via the queue — never the server's GPU 0.
set -uo pipefail
REPO=/data1/zhangao/repos/3dgs-imc-codex
SVR_Q=/data1/zhangao/outputs/probe_queue
WSL_REPO=/mnt/c/Users/Aooooo/Documents/wsl_repos/3dgs-imc-codex
WSL_Q=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/dev_queue
WSL_OUT=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/diag
WSL_DIAG="$WSL_OUT/mipnerf360"
MIRROR="$REPO/outputs/wsl_results/diag/mipnerf360"
STATE=/data1/zhangao/outputs/sentinel_state
LOG=/data1/zhangao/outputs/sweep_sentinel.log
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 wsl"
RETRY_MAX=${RETRY_MAX:-3}
PERIOD=${PERIOD:-180}
IDLE_KILL=${IDLE_KILL:-6}   # consecutive idle cycles (~PERIOD*6 s) before declaring a D-state hang
mkdir -p "$STATE/retries" "$MIRROR"
say(){ echo "[sentinel $(date '+%F %T')] $*" >> "$LOG"; }
say "start pid=$$ period=${PERIOD}s retry_max=$RETRY_MAX idle_kill=$IDLE_KILL"

idle=0
while true; do
    # 1. mirror sync (metrics.csv only, preserve per-cell dir structure)
    rsync -azq -e "ssh -o BatchMode=yes" --include='*/' --include='metrics.csv' --exclude='*' \
        "wsl:$WSL_DIAG/" "$MIRROR/" 2>/dev/null || say "rsync transient fail"

    # 2. WSL state in one ssh round-trip (KEY=int lines, eval-safe after grep filter)
    wsl_state=$($SSH "echo PEND=\$(ls $WSL_Q/pending/*.job 2>/dev/null|wc -l); \
echo RUN=\$(ls $WSL_Q/running/*.job 2>/dev/null|wc -l); \
echo UTIL=\$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null|head -1|tr -dc 0-9); \
echo WORKER=\$(pgrep -fc probe_worker.sh 2>/dev/null||echo 0)" 2>/dev/null)
    PEND=0; RUN=0; UTIL=0; WORKER=0
    eval "$(printf '%s\n' "$wsl_state" | grep -E '^(PEND|RUN|UTIL|WORKER)=[0-9]*$')" 2>/dev/null || true
    wpend=${PEND:-0}; wrun=${RUN:-0}; wutil=${UTIL:-0}; nworker=${WORKER:-0}

    # 3. re-queue WSL failed cells (transient hangs), retry counter kept server-side
    for nm in $($SSH "cd $WSL_Q/failed 2>/dev/null && ls *.job 2>/dev/null | sed 's/\.job$//'" 2>/dev/null); do
        n=$(cat "$STATE/retries/$nm" 2>/dev/null || echo 0)
        if [ "${n:-0}" -lt "$RETRY_MAX" ]; then
            if $SSH "mv $WSL_Q/failed/$nm.job $WSL_Q/pending/ 2>/dev/null"; then
                echo $((n+1)) > "$STATE/retries/$nm"; say "requeue WSL $nm (try $((n+1))/$RETRY_MAX)"
            fi
        else
            say "WSL $nm exhausted $RETRY_MAX retries — leaving in failed/"
        fi
    done

    # 4. D-state hang: running but GPU idle for IDLE_KILL cycles -> kill python (worker -> failed -> requeued)
    if [ "$wrun" -gt 0 ] && [ "${wutil:-0}" -lt 3 ]; then idle=$((idle+1)); else idle=0; fi
    if [ "$idle" -ge "$IDLE_KILL" ]; then
        say "WSL D-STATE HANG (util ${wutil}% running=$wrun ~$((PERIOD*IDLE_KILL/60))min) -> pkill train"
        $SSH 'pkill -9 -f train_experimental_branch3_mc.py' 2>/dev/null || true
        idle=0
    fi

    # 5. restart dead worker if work remains
    if [ "${nworker:-0}" -lt 1 ] && [ $((wpend+wrun)) -gt 0 ]; then
        say "WSL worker dead, work remains (p$wpend r$wrun) -> restart"
        $SSH "cd $WSL_REPO && QDIR=$WSL_Q OUT_ROOT=$WSL_OUT WORKER_GPU=0 NVSMI=/usr/lib/wsl/lib/nvidia-smi GPU_FREE_MIN=16000 RAM_MIN=8 nohup setsid bash tools/probe_worker.sh >/dev/null 2>&1 < /dev/null &" 2>/dev/null || true
    fi

    # 6. server queue (read-only, for the heartbeat)
    spend=$(ls "$SVR_Q"/pending/*.job 2>/dev/null | wc -l)
    srun=$(ls "$SVR_Q"/running/*.job 2>/dev/null | wc -l)
    mc=$(find "$MIRROR" -name metrics.csv 2>/dev/null | wc -l)
    say "svr(p$spend r$srun) wsl(p$wpend r$wrun u${wutil}% w$nworker) idle=$idle mirror=$mc"

    # 7. exit -> wake agent when the WSL lane is fully drained
    if [ $((wpend+wrun)) -eq 0 ]; then say "WSL LANE DRAINED -> exit (final pull done)"; exit 0; fi
    sleep "$PERIOD"
done
