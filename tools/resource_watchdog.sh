#!/usr/bin/env bash
# =============================================================================
# resource_watchdog.sh — protect the machine from our own training jobs.
# Watches RAM / GPU / CPU / D-state every INTERVAL seconds and kills OUR
# offending process immediately (only $USER's train/render/metrics processes;
# never touches other users). All actions logged to $LOG.
#   start:  nohup setsid bash tools/resource_watchdog.sh >/dev/null 2>&1 &
#   stop :  kill $(cat /data1/zhangao/outputs/watchdog.pid)
# Thresholds (env-overridable):
#   GPU_PROC_MAX=26000 MiB   per-process GPU memory cap
#   GPU_TOTAL_MAX=30000 MiB  total GPU-1 cap (kill largest of ours)
#   RAM_FLOOR_GB=16          MemAvailable floor (kill largest-RSS of ours)
#   RAM_PANIC_GB=6           panic floor (kill ALL ours)
#   PROC_RSS_GB=45           per-process RSS cap (leak tripwire)
#   LOAD_MAX=100             1-min loadavg cap, sustained 3 checks (112 cores)
#   DSTATE_CHECKS=3          consecutive D-state checks before kill+alert
# =============================================================================
set -uo pipefail
LOG=${LOG:-/data1/zhangao/outputs/watchdog.log}
PIDFILE=${PIDFILE:-/data1/zhangao/outputs/watchdog.pid}
INTERVAL=${INTERVAL:-15}
GPU_PROC_MAX=${GPU_PROC_MAX:-26000}
GPU_TOTAL_MAX=${GPU_TOTAL_MAX:-30000}
RAM_FLOOR_GB=${RAM_FLOOR_GB:-16}
RAM_PANIC_GB=${RAM_PANIC_GB:-6}
PROC_RSS_GB=${PROC_RSS_GB:-45}
LOAD_MAX=${LOAD_MAX:-100}
DSTATE_CHECKS=${DSTATE_CHECKS:-3}
PATTERN='train_experimental_branch3_mc.py|render\.py|metrics\.py'
# nvidia-smi is not on PATH in non-interactive WSL shells
NVSMI=${NVSMI:-$(command -v nvidia-smi 2>/dev/null || echo /usr/lib/wsl/lib/nvidia-smi)}

echo $$ > "$PIDFILE"
log() { echo "[watchdog $(date '+%F %T')] $*" >> "$LOG"; }
log "started pid=$$ interval=${INTERVAL}s gpu_proc=${GPU_PROC_MAX} gpu_total=${GPU_TOTAL_MAX} ram_floor=${RAM_FLOOR_GB}G panic=${RAM_PANIC_GB}G rss=${PROC_RSS_GB}G load=${LOAD_MAX}"

my_procs() {  # pid rss_kb cpu% state cmd  (only $USER + our training pattern)
    ps -u "$USER" -o pid=,rss=,pcpu=,stat=,cmd= | grep -E "$PATTERN" | grep -v grep | grep -v watchdog || true
}

kill_proc() { # pid reason
    local pid=$1 reason=$2
    local cmd; cmd=$(ps -o cmd= -p "$pid" 2>/dev/null | cut -c1-160)
    log "KILL pid=$pid reason='$reason' cmd='$cmd'"
    kill "$pid" 2>/dev/null; sleep 3
    kill -0 "$pid" 2>/dev/null && { kill -9 "$pid" 2>/dev/null; log "KILL -9 pid=$pid (TERM ignored)"; }
}

declare -A dstate_count load_strikes=0
load_strikes=0
while true; do
    procs=$(my_procs)

    # ---- GPU: per-process and total caps ----
    if [ -n "$procs" ]; then
        while IFS=, read -r gpid gmem; do
            gpid=$(echo "$gpid" | tr -d ' '); gmem=$(echo "$gmem" | tr -d ' ')
            [ -n "$gpid" ] || continue
            if echo "$procs" | awk '{print $1}' | grep -qx "$gpid" && [ "${gmem:-0}" -gt "$GPU_PROC_MAX" ]; then
                kill_proc "$gpid" "GPU proc mem ${gmem}MiB > ${GPU_PROC_MAX}MiB"
            fi
        done < <("$NVSMI" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
        gtot=$("$NVSMI" --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
        if [ "${gtot:-0}" -gt "$GPU_TOTAL_MAX" ]; then
            big=$("$NVSMI" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
                  | tr -d ' ' | sort -t, -k2 -nr | awk -F, '{print $1}' \
                  | while read -r p; do echo "$procs" | awk '{print $1}' | grep -qx "$p" && { echo "$p"; break; }; done)
            [ -n "${big:-}" ] && kill_proc "$big" "GPU1 total ${gtot}MiB > ${GPU_TOTAL_MAX}MiB"
        fi
    fi

    # ---- RAM: floors and per-process RSS ----
    avail_gb=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$avail_gb" -lt "$RAM_PANIC_GB" ] && [ -n "$procs" ]; then
        log "PANIC MemAvailable=${avail_gb}G < ${RAM_PANIC_GB}G — killing ALL our trainings"
        echo "$procs" | awk '{print $1}' | while read -r p; do kill_proc "$p" "RAM panic"; done
    elif [ "$avail_gb" -lt "$RAM_FLOOR_GB" ] && [ -n "$procs" ]; then
        big=$(echo "$procs" | sort -k2 -nr | head -1 | awk '{print $1}')
        [ -n "$big" ] && kill_proc "$big" "MemAvailable ${avail_gb}G < ${RAM_FLOOR_GB}G (largest RSS)"
    fi
    if [ -n "$procs" ]; then
        echo "$procs" | while read -r pid rss _cpu _stat _cmd; do
            if [ "${rss:-0}" -gt $((PROC_RSS_GB * 1048576)) ]; then
                kill_proc "$pid" "RSS $((rss/1048576))G > ${PROC_RSS_GB}G"
            fi
        done
    fi

    # ---- CPU: sustained extreme load ----
    load=$(awk '{print int($1)}' /proc/loadavg)
    if [ "$load" -gt "$LOAD_MAX" ]; then
        load_strikes=$((load_strikes + 1))
        if [ "$load_strikes" -ge 3 ] && [ -n "$procs" ]; then
            big=$(echo "$procs" | sort -k3 -nr | head -1 | awk '{print $1}')
            [ -n "$big" ] && kill_proc "$big" "loadavg ${load} > ${LOAD_MAX} x3 (top CPU)"
            load_strikes=0
        fi
    else
        load_strikes=0
    fi

    # ---- D-state (nvidia-uvm hang — this box's known crash mode) ----
    if [ -n "$procs" ]; then
        while read -r pid _rss _cpu stat _cmd; do
            case "$stat" in
                D*) dstate_count[$pid]=$(( ${dstate_count[$pid]:-0} + 1 ))
                    if [ "${dstate_count[$pid]}" -ge "$DSTATE_CHECKS" ]; then
                        log "CRITICAL pid=$pid in D-state x${dstate_count[$pid]} (possible nvidia-uvm hang) — attempting kill"
                        kill_proc "$pid" "D-state x${dstate_count[$pid]}"
                        dstate_count[$pid]=0
                    fi ;;
                *) dstate_count[$pid]=0 ;;
            esac
        done <<< "$procs"
    fi

    sleep "$INTERVAL"
done
