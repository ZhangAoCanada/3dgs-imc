#!/usr/bin/env bash
# Sweep sentinel v2: v1 healing + WORK-STEALING. Every cycle:
#   1. rsync WSL metrics.csv -> server mirror
#   2. re-queue WSL failed/ cells (D-state hang casualties), retry cap server-side
#   3. kill WSL D-state render hangs (running but GPU idle ~18min)
#   4. restart the WSL probe_worker if it died with work remaining
#   5. TOP-UP STEAL: WSL queue empty + server pending>1 -> move 2 server jobs to WSL
#      (atomic: mv to staging first so seq_runner can't double-run them)
#   6. exit (waking the agent) only when BOTH lanes are fully drained
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
IDLE_KILL=${IDLE_KILL:-6}
mkdir -p "$STATE/retries" "$MIRROR" "$SVR_Q/staging"
say(){ echo "[sentinel2 $(date '+%F %T')] $*" >> "$LOG"; }
# single-instance pidfile: if another sentinel is already live, exit (prevents racing duplicates
# from my pgrep-self-matching liveness checks). Robust check elsewhere = kill -0 $(cat sentinel.pid).
SPID=/data1/zhangao/outputs/sentinel.pid
if [ -f "$SPID" ] && kill -0 "$(cat "$SPID" 2>/dev/null)" 2>/dev/null && [ "$(cat "$SPID")" != "$$" ]; then
    echo "[sentinel2 $(date '+%F %T')] another sentinel ($(cat "$SPID")) alive — exiting" >> "$LOG"; exit 0
fi
echo $$ > "$SPID"; trap 'rm -f "$SPID"' EXIT
say "start pid=$$ period=${PERIOD}s retry_max=$RETRY_MAX idle_kill=$IDLE_KILL steal=on"

idle=0
while true; do
    rsync -azq -e "ssh -o BatchMode=yes" --include='*/' --include='metrics.csv' --exclude='*' \
        "wsl:$WSL_DIAG/" "$MIRROR/" 2>/dev/null || say "rsync transient fail"

    wsl_state=$($SSH "echo PEND=\$(ls $WSL_Q/pending/*.job 2>/dev/null|wc -l); \
echo RUN=\$(ls $WSL_Q/running/*.job 2>/dev/null|wc -l); \
echo UTIL=\$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null|head -1|tr -dc 0-9); \
echo WORKER=\$(pgrep -c -f 'bash tools/probe_worker.sh' 2>/dev/null||echo 0)" 2>/dev/null)
    PEND=0; RUN=0; UTIL=0; WORKER=0
    eval "$(printf '%s\n' "$wsl_state" | grep -E '^(PEND|RUN|UTIL|WORKER)=[0-9]*$')" 2>/dev/null || true
    wpend=${PEND:-0}; wrun=${RUN:-0}; wutil=${UTIL:-0}; nworker=${WORKER:-0}
    # WSL reachability: empty fetch = tunnel/host down. Skip WSL ops so we don't mistake
    # "unreachable" for "empty" (which would spam-steal server jobs to a dead WSL).
    WSL_OK=1; printf '%s' "$wsl_state" | grep -q '^PEND=' || WSL_OK=0
    if [ "$WSL_OK" = 0 ]; then say "WSL UNREACHABLE (tunnel/host down) — server-only this cycle"; fi

    for nm in $($SSH "cd $WSL_Q/failed 2>/dev/null && ls *.job 2>/dev/null | sed 's/\.job$//'" 2>/dev/null); do
        n=$(cat "$STATE/retries/$nm" 2>/dev/null || echo 0)
        if [ "${n:-0}" -lt "$RETRY_MAX" ]; then
            if $SSH "mv $WSL_Q/failed/$nm.job $WSL_Q/pending/ 2>/dev/null"; then
                echo $((n+1)) > "$STATE/retries/$nm"; say "requeue WSL $nm (try $((n+1))/$RETRY_MAX)"
            fi
        else say "WSL $nm exhausted $RETRY_MAX retries — leaving in failed/"; fi
    done

    if [ "$wrun" -gt 0 ] && [ "${wutil:-0}" -lt 3 ]; then idle=$((idle+1)); else idle=0; fi
    if [ "$idle" -ge "$IDLE_KILL" ]; then
        say "WSL D-STATE HANG (util ${wutil}% running=$wrun) -> pkill train"
        $SSH 'pkill -9 -f train_experimental_branch3_mc.py' 2>/dev/null || true
        idle=0
    fi

    if [ "${nworker:-0}" -lt 1 ] && [ $((wpend+wrun)) -gt 0 ]; then
        say "WSL worker dead, work remains (p$wpend r$wrun) -> restart"
        $SSH "cd $WSL_REPO && DATA_ROOT=/mnt/c/Users/Aooooo/Documents/DATA CONDA_EXE=/home/ooooo/miniconda3/bin/conda ENV_NAME=3dgs-imc QDIR=$WSL_Q OUT_ROOT=$WSL_OUT WORKER_GPU=0 NVSMI=/usr/lib/wsl/lib/nvidia-smi MAX_SLOTS=1 GPU_FREE_MIN=16000 RAM_MIN=8 nohup setsid bash tools/probe_worker.sh >/dev/null 2>&1 < /dev/null &" 2>/dev/null || true
    fi

    spend=$(ls "$SVR_Q"/pending/*.job 2>/dev/null | wc -l)
    srun=$(ls "$SVR_Q"/running/*.job 2>/dev/null | wc -l)

    # 4b. SERVER LANE SUPERVISION: restart seq_runner if it died with work pending.
    # (harness run_in_background tasks do NOT survive session events; nohup setsid does.)
    # pidfile + kill-0 (robust): pgrep -f self-matches orphaned launcher-wrapper cmdlines and
    # falsely reports seq_runner "alive" — that left GPU1 idle 2h after a drain.
    nseq=0; sp=$(cat "$SVR_Q/seq.pid" 2>/dev/null); [ -n "$sp" ] && kill -0 "$sp" 2>/dev/null && nseq=1
    ntrain=$(pgrep -c -f 'train_experimental_branch3_mc.py' 2>/dev/null || echo 0)
    if [ "${nseq:-0}" -lt 1 ] && [ "$spend" -gt 0 ]; then
        if [ "${ntrain:-0}" -ge 1 ]; then
            say "WARN seq_runner dead but $ntrain train proc(s) alive on server — NOT auto-restarting"
        else
            for rj in "$SVR_Q"/running/*.job; do
                [ -e "$rj" ] && mv "$rj" "$SVR_Q/pending/" 2>/dev/null && say "recovered stuck server job $(basename "$rj")"
            done
            rm -f "$SVR_Q/STOP"
            ( cd "$REPO" && QDIR="$SVR_Q" CUDA_VISIBLE_DEVICES=1 setsid bash tools/seq_runner.sh \
                > /data1/zhangao/outputs/seq_runner.log 2>&1 < /dev/null & )
            spend=$(ls "$SVR_Q"/pending/*.job 2>/dev/null | wc -l)
            srun=$(ls "$SVR_Q"/running/*.job 2>/dev/null | wc -l)
            say "server seq_runner RESTARTED (pending=$spend)"
        fi
    fi

    # 4c. WATCHDOG SUPERVISION (crash-safety): restart either resource_watchdog if it died.
    # pidfile + kill-0 is self-match-proof (pgrep -f can match the probe shell's own cmdline).
    swd=$(cat /data1/zhangao/outputs/watchdog.pid 2>/dev/null || echo "")
    if [ -z "$swd" ] || ! kill -0 "$swd" 2>/dev/null; then
        say "SERVER GPU1 watchdog DEAD -> restart"
        ( cd "$REPO" && WATCH_GPU=1 nohup setsid bash tools/resource_watchdog.sh >/dev/null 2>&1 </dev/null & )
    fi
    # explicit ALIVE/DEAD so an ssh blip (empty) does NOT spawn a racing 2nd watchdog
    # ALIVE only if the pid is up AND the heartbeat is fresh (<60s) — a stuck-but-alive watchdog
    # (blocked on a hung nvidia-smi during a uvm event) reads DEAD so we restart it.
    wwd=$($SSH 'p=$(cat /mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/watchdog.pid 2>/dev/null); hb=$(cat /mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/watchdog.heartbeat 2>/dev/null); age=$(( $(date +%s) - ${hb:-0} )); if [ -n "$p" ] && kill -0 "$p" 2>/dev/null && [ "${hb:-0}" -gt 0 ] && [ "$age" -lt 60 ]; then echo ALIVE; else echo DEAD; fi' 2>/dev/null)
    if [ "$wwd" = "DEAD" ]; then
        say "WSL GPU0 watchdog DEAD -> restart"
        # kill a stuck-but-alive instance first (by pidfile) so we don't leave a wedged one running
        $SSH 'p=$(cat /mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/watchdog.pid 2>/dev/null); [ -n "$p" ] && kill "$p" 2>/dev/null; true' 2>/dev/null || true
        $SSH "cd $WSL_REPO && WATCH_GPU=0 INTERVAL=10 DSTATE_CHECKS=2 HB_EVERY=18 \
LOG=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/watchdog.log \
PIDFILE=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/watchdog.pid \
HEARTBEAT=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/watchdog.heartbeat \
NVSMI=/usr/lib/wsl/lib/nvidia-smi GPU_PROC_MAX=24000 GPU_TOTAL_MAX=28000 \
RAM_FLOOR_GB=6 RAM_PANIC_GB=3 PROC_RSS_GB=24 LOAD_MAX=40 \
nohup setsid bash tools/resource_watchdog.sh >/dev/null 2>&1 </dev/null &" 2>/dev/null || true
    fi

    # 5. top-up steal: keep WSL fed while the server has a backlog. Skips >=5M-cap jobs
    # (bicycle 5M would OOM WSL's 32GB card) — those stay on the server. Only when WSL is reachable.
    if [ "$WSL_OK" = 1 ] && [ $((wpend+wrun)) -eq 0 ] && [ "$spend" -gt 1 ]; then
        moved=0
        for jf in $(ls -1 "$SVR_Q"/pending/*.job 2>/dev/null | sort); do
            [ "$moved" -ge 2 ] && break
            cap=$(grep -E '^CAP_MAX=' "$jf" 2>/dev/null | head -1 | cut -d= -f2)
            [ "${cap:-0}" -ge 4000000 ] && continue   # keep >=4M (bicycle/stump/garden) on server;
                                                       # 4M cells OOM'd WSL when paired 2-wide. treehill 3.3M is OK.
            nm=$(basename "$jf")
            if mv "$jf" "$SVR_Q/staging/$nm" 2>/dev/null; then
                if scp -q -o BatchMode=yes "$SVR_Q/staging/$nm" "wsl:$WSL_Q/pending/$nm" 2>/dev/null; then
                    rm -f "$SVR_Q/staging/$nm"; moved=$((moved+1)); say "STEAL $nm -> WSL"
                else mv "$SVR_Q/staging/$nm" "$SVR_Q/pending/$nm"; say "steal $nm failed, returned"; fi
            fi
        done
        spend=$(ls "$SVR_Q"/pending/*.job 2>/dev/null | wc -l)
    fi

    # 5b. REVERSE steal: when the (1-wide) server is fully idle but WSL has a backlog,
    # pull up to 2 WSL pending jobs onto the idle server (it has all scene data; indoor 2M
    # fits GPU1 32GB) and (re)launch seq_runner. Prevents GPU1 idling while WSL grinds.
    if [ "$spend" -eq 0 ] && [ "$srun" -eq 0 ] && [ "$wpend" -ge 1 ]; then
        for nm in $($SSH "cd $WSL_Q/pending 2>/dev/null && ls *.job 2>/dev/null | sort | head -2" 2>/dev/null); do
            if scp -q -o BatchMode=yes "wsl:$WSL_Q/pending/$nm" "$SVR_Q/pending/$nm" 2>/dev/null; then
                $SSH "rm -f $WSL_Q/pending/$nm" 2>/dev/null && say "REVERSE-STEAL $nm WSL -> server"
            fi
        done
        spend=$(ls "$SVR_Q"/pending/*.job 2>/dev/null | wc -l)
        nseq2=0; sp2=$(cat "$SVR_Q/seq.pid" 2>/dev/null); [ -n "$sp2" ] && kill -0 "$sp2" 2>/dev/null && nseq2=1
        if [ "${nseq2:-0}" -lt 1 ] && [ "$spend" -gt 0 ]; then
            rm -f "$SVR_Q/STOP"
            ( cd "$REPO" && QDIR="$SVR_Q" CUDA_VISIBLE_DEVICES=1 setsid bash tools/seq_runner.sh \
                > /data1/zhangao/outputs/seq_runner.log 2>&1 < /dev/null & )
            say "seq_runner relaunched after reverse-steal (pending=$spend)"
        fi
    fi

    mc=$(find "$MIRROR" -name metrics.csv 2>/dev/null | wc -l)
    say "svr(p$spend r$srun) wsl(p$wpend r$wrun u${wutil}% w$nworker) idle=$idle mirror=$mc"

    # 6. exit -> wake agent only when BOTH lanes are drained
    if [ $((wpend+wrun)) -eq 0 ] && [ $((spend+srun)) -eq 0 ]; then
        rsync -azq -e "ssh -o BatchMode=yes" --include='*/' --include='metrics.csv' --exclude='*' \
            "wsl:$WSL_DIAG/" "$MIRROR/" 2>/dev/null
        say "BOTH LANES DRAINED -> exit (final pull done)"; exit 0
    fi
    sleep "$PERIOD"
done
