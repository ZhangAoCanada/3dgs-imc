#!/usr/bin/env bash
# Minimal ROBUST keeper — replaces the fragile sweep_sentinel2. Keeps the crash-protection
# watchdogs + the WSL worker alive, and relaunches the server seq_runner whenever the server
# queue has pending work. Design rules that fix the past failures:
#   - liveness via PIDFILE + kill-0 (+ heartbeat for watchdogs); NEVER pgrep (it self-matches
#     orphaned launcher wrappers and the calling shell -> false "alive" -> 2h idle GPU).
#   - NEVER exits on drain (the sentinel exited at drain, so new work sat unsupervised).
#   - single-instance via own pidfile (stops the duplicate-keeper spiral).
#   - no bidirectional steal / no mirror sync (the complexity that broke); route work explicitly.
set -uo pipefail
REPO=/data1/zhangao/repos/3dgs-imc-codex
OUT=/data1/zhangao/outputs
SVR_Q=$OUT/probe_queue
KPID=$OUT/keeper.pid
# timeout wraps the whole ssh: a slow/hung remote (or a backgrounded relaunch that holds the
# channel open) must NOT block the keeper loop — that hung the first version in do_wait forever.
SSH="timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=15 wsl"
WR=/mnt/c/Users/Aooooo/Documents/wsl_repos
log(){ echo "[keeper $(date '+%F %T')] $*" >> "$OUT/keeper.log"; }

# single instance
if [ -f "$KPID" ] && kill -0 "$(cat "$KPID" 2>/dev/null)" 2>/dev/null && [ "$(cat "$KPID")" != "$$" ]; then
    log "another keeper ($(cat "$KPID")) alive — exiting"; exit 0
fi
echo $$ > "$KPID"; trap 'rm -f "$KPID"' EXIT
log "start pid=$$"

alive(){ local p; p=$(cat "$1" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

while true; do
    # ---- SERVER watchdog (GPU1) ----
    hb=$(cat "$OUT/watchdog.heartbeat" 2>/dev/null||echo 0); age=$(( $(date +%s) - hb ))
    if ! alive "$OUT/watchdog.pid" || { [ "$hb" -gt 0 ] && [ "$age" -ge 90 ]; }; then
        op=$(cat "$OUT/watchdog.pid" 2>/dev/null); [ -n "$op" ] && kill "$op" 2>/dev/null
        # SERVER (GPU1 = 24G card sized for 24G jobs): keep ABSOLUTE caps so we never
        # kill a healthy server job. GPU_TOTAL_MAX/PROC>cardmax => intentionally inert;
        # RAM floor stays absolute 16G (server has huge RAM shared with other users).
        ( cd "$REPO" && WATCH_GPU=1 INTERVAL=10 DSTATE_CHECKS=6 HB_EVERY=18 \
            LOG=$OUT/watchdog.log PIDFILE=$OUT/watchdog.pid HEARTBEAT=$OUT/watchdog.heartbeat \
            GPU_PROC_MAX=26000 GPU_TOTAL_MAX=30000 RAM_AVAIL_FLOOR_GB=16 RAM_PANIC_GB=6 PROC_RSS_GB=45 LOAD_MAX=100 \
            nohup setsid bash tools/resource_watchdog.sh >/dev/null 2>&1 </dev/null & )
        log "server watchdog restarted"
    fi

    # ---- SERVER seq_runner (GPU1 ONLY — GPU0 is reserved for other users, do NOT touch) ----
    for spec in "1:$SVR_Q:$OUT/seq_runner.log"; do
        gpu=${spec%%:*}; rest=${spec#*:}; q=${rest%%:*}; slog=${rest#*:}
        sp=$(ls "$q"/pending/*.job 2>/dev/null | wc -l)
        if [ "$sp" -gt 0 ] && ! alive "$q/seq.pid"; then
            for j in "$q"/running/*.job; do [ -e "$j" ] && mv "$j" "$q/pending/" 2>/dev/null; done
            rm -f "$q/STOP"
            ( cd "$REPO" && QDIR="$q" CUDA_VISIBLE_DEVICES="$gpu" nohup setsid bash tools/seq_runner.sh > "$slog" 2>&1 </dev/null & )
            log "seq_runner(GPU$gpu) restarted (pending=$sp)"
        fi
    done

    # ---- WSL worker + watchdog (one ssh round-trip; all pidfile/heartbeat) ----
    $SSH "set -uo pipefail; d=$WR/outputs/dev_queue
        wp=\$(cat \$d/worker.pid 2>/dev/null)
        if ! { [ -n \"\$wp\" ] && kill -0 \"\$wp\" 2>/dev/null; }; then
            cd $WR/3dgs-imc-codex && DATA_ROOT=/mnt/c/Users/Aooooo/Documents/DATA CONDA_EXE=/home/ooooo/miniconda3/bin/conda ENV_NAME=3dgs-imc \
              QDIR=\$d OUT_ROOT=$WR/outputs/diag WORKER_GPU=0 NVSMI=/usr/lib/wsl/lib/nvidia-smi MAX_SLOTS=1 GPU_FREE_MIN=16000 RAM_MIN=8 \
              nohup setsid bash tools/probe_worker.sh >/dev/null 2>&1 </dev/null & echo WORKER_RESTARTED
        fi
        p=\$(cat $WR/outputs/watchdog.pid 2>/dev/null); hb=\$(cat $WR/outputs/watchdog.heartbeat 2>/dev/null||echo 0); age=\$(( \$(date +%s) - hb ))
        if ! { [ -n \"\$p\" ] && kill -0 \"\$p\" 2>/dev/null && [ \"\$hb\" -gt 0 ] && [ \$age -lt 60 ]; }; then
            [ -n \"\$p\" ] && kill \"\$p\" 2>/dev/null
            cd $WR/3dgs-imc-codex && WATCH_GPU=0 INTERVAL=8 DSTATE_CHECKS=6 HB_EVERY=18 \
              LOG=$WR/outputs/watchdog.log PIDFILE=$WR/outputs/watchdog.pid HEARTBEAT=$WR/outputs/watchdog.heartbeat \
              NVSMI=/usr/lib/wsl/lib/nvidia-smi GPU_PCT=80 RAM_PCT=80 RAM_PANIC_GB=3 LOAD_MAX=40 \
              nohup setsid bash tools/resource_watchdog.sh >/dev/null 2>&1 </dev/null & echo WD_RESTARTED
        fi" 2>/dev/null | while read -r m; do log "WSL $m"; done

    # ---- retry failed jobs (genuine uvm hang / transient): fail/ -> pending, capped at 3 tries ----
    for jf in "$SVR_Q"/fail/*.job; do
        [ -e "$jf" ] || continue; n=$(basename "$jf" .job); mkdir -p "$SVR_Q/.retries"
        t=$(( $(cat "$SVR_Q/.retries/$n" 2>/dev/null || echo 0) + 1 )); echo "$t" > "$SVR_Q/.retries/$n"
        if [ "$t" -le 3 ]; then mv "$jf" "$SVR_Q/pending/" && log "RETRY $n (attempt $t)"; else log "GIVEUP $n after $((t-1)) fails"; fi
    done
    $SSH "d=$WR/outputs/dev_queue; mkdir -p \$d/.retries
        for jf in \$d/fail/*.job; do [ -e \"\$jf\" ] || continue; n=\$(basename \"\$jf\" .job)
            t=\$(( \$(cat \$d/.retries/\$n 2>/dev/null || echo 0) + 1 )); echo \$t > \$d/.retries/\$n
            if [ \$t -le 3 ]; then mv \"\$jf\" \$d/pending/ && echo \"RETRY \$n (try \$t)\"; else echo \"GIVEUP \$n\"; fi
        done" 2>/dev/null | while read -r m; do log "WSL $m"; done

    sleep 60
done
