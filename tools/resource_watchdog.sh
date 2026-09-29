#!/usr/bin/env bash
# =============================================================================
# resource_watchdog.sh — protect the machine from our own training jobs.
# Watches RAM / GPU / CPU / D-state every INTERVAL seconds and kills OUR
# offending process immediately (only $USER's train/render/metrics processes;
# never touches other users). All actions logged to $LOG.
#   start:  nohup setsid bash tools/resource_watchdog.sh >/dev/null 2>&1 &
#   stop :  kill $(cat /data1/zhangao/outputs/watchdog.pid)
#
# PRIMARY RULE (user mandate, do NOT let the box crash):
#   * kill OUR largest job the instant TOTAL GPU memory used >= GPU_PCT% (default 80)
#   * kill OUR largest job the instant TOTAL RAM used      >= RAM_PCT% (default 80)
#   * kill any single OUR process using >= GPU_PCT% of GPU mem, or >= RAM_PCT% of RAM
# Percentage caps are derived from the GPU's reported memory.total and /proc MemTotal
# at startup, so they self-adapt to the 24G server card and the 32G WSL card alike.
# Absolute *_MAX/*_GB env overrides still win if set explicitly.
#
# Thresholds (env-overridable):
#   GPU_PCT=80               kill when GPU mem used (total OR per-proc) >= this %% of card total
#   RAM_PCT=80               kill when RAM used (total OR per-proc RSS) >= this %% of MemTotal
#   GPU_PROC_MAX / GPU_TOTAL_MAX (MiB)  explicit overrides; default = GPU_PCT%% of card total
#   RAM_USED_MAX_GB / PROC_RSS_GB (GB)  explicit overrides; default = RAM_PCT%% of MemTotal
#   RAM_PANIC_GB=4           hard panic: MemAvailable below this -> kill ALL ours
#   LOAD_MAX=100             1-min loadavg cap, sustained 3 checks (112 cores)
#   DSTATE_CHECKS=2          consecutive D-state checks before kill+alert (nvidia-uvm hang)
# =============================================================================
set -uo pipefail
LOG=${LOG:-/data1/zhangao/outputs/watchdog.log}
PIDFILE=${PIDFILE:-/data1/zhangao/outputs/watchdog.pid}
INTERVAL=${INTERVAL:-10}
WATCH_GPU=${WATCH_GPU:-1}        # GPU index for the total-memory cap (server=1; WSL=0, single-GPU)
GPU_PCT=${GPU_PCT:-80}
RAM_PCT=${RAM_PCT:-80}
HEARTBEAT=${HEARTBEAT:-${LOG%.log}.heartbeat}  # liveness file: holds epoch-seconds of last loop
HB_EVERY=${HB_EVERY:-20}         # log a heartbeat line every HB_EVERY loops
RAM_PANIC_GB=${RAM_PANIC_GB:-4}
LOAD_MAX=${LOAD_MAX:-100}
DSTATE_CHECKS=${DSTATE_CHECKS:-2}
# MATCH BOTH PIPELINES: train_experimental_branch3.py (slow) AND ..._mc.py (fast),
# plus any train*.py / render.py / metrics.py. The old pattern only matched _mc.py,
# so SLOW repro jobs were INVISIBLE to the watchdog and could wedge the box uncapped.
PATTERN='train_experimental_branch3[._]|train[A-Za-z0-9_]*\.py|render\.py|metrics\.py'
# nvidia-smi is not on PATH in non-interactive WSL shells
NVSMI=${NVSMI:-$(command -v nvidia-smi 2>/dev/null || echo /usr/lib/wsl/lib/nvidia-smi)}

# ---- derive percentage-based caps from the actual hardware (once) ----
GPU_TOTAL_MiB=$("$NVSMI" --query-gpu=memory.total --format=csv,noheader,nounits -i "$WATCH_GPU" 2>/dev/null | head -1 | tr -d ' ')
[ -n "${GPU_TOTAL_MiB:-}" ] && [ "${GPU_TOTAL_MiB:-0}" -gt 0 ] 2>/dev/null || GPU_TOTAL_MiB=24000
RAM_TOTAL_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo)
RAM_TOTAL_GB=$((RAM_TOTAL_KB / 1048576))
GPU_TOTAL_MAX=${GPU_TOTAL_MAX:-$(( GPU_TOTAL_MiB * GPU_PCT / 100 ))}   # total-GPU cap (MiB)
GPU_PROC_MAX=${GPU_PROC_MAX:-$(( GPU_TOTAL_MiB * GPU_PCT / 100 ))}     # per-proc GPU cap (MiB)
RAM_USED_MAX_GB=${RAM_USED_MAX_GB:-$(( RAM_TOTAL_GB * RAM_PCT / 100 ))}   # total-RAM-used cap (GB)
PROC_RSS_GB=${PROC_RSS_GB:-$(( RAM_TOTAL_GB * RAM_PCT / 100 ))}           # per-proc RSS cap (GB)
RAM_AVAIL_FLOOR_GB=${RAM_AVAIL_FLOOR_GB:-$(( RAM_TOTAL_GB - RAM_USED_MAX_GB ))}   # avail floor == kill when below (server passes absolute 16)

echo $$ > "$PIDFILE"
log() { echo "[watchdog $(date '+%F %T')] $*" >> "$LOG"; }
log "started pid=$$ interval=${INTERVAL}s gpu=${WATCH_GPU} GPU_total=${GPU_TOTAL_MiB}MiB pct=${GPU_PCT}% -> gpu_proc=${GPU_PROC_MAX} gpu_total=${GPU_TOTAL_MAX}; RAM_total=${RAM_TOTAL_GB}G pct=${RAM_PCT}% -> used_max=${RAM_USED_MAX_GB}G avail_floor=${RAM_AVAIL_FLOOR_GB}G rss=${PROC_RSS_GB}G panic=${RAM_PANIC_GB}G load=${LOAD_MAX}"

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

declare -A dstate_count io_prev; load_strikes=0
load_strikes=0
hb=0
while true; do
    procs=$(my_procs)
    # heartbeat: prove the loop is live (file = last-iteration epoch) + periodic status line
    hb=$((hb + 1)); date +%s > "$HEARTBEAT" 2>/dev/null
    if [ $((hb % HB_EVERY)) -eq 0 ]; then
        hbmem=$("$NVSMI" --query-gpu=memory.used --format=csv,noheader,nounits -i "$WATCH_GPU" 2>/dev/null | head -1)
        log "heartbeat gpu${WATCH_GPU}=${hbmem:-?}MiB/${GPU_TOTAL_MAX} our_procs=$(printf '%s' "$procs" | grep -c .)"
    fi

    # ---- GPU: total cap (PRIMARY machine-protection rule) ----
    gtot=$("$NVSMI" --query-gpu=memory.used --format=csv,noheader,nounits -i "$WATCH_GPU" 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$procs" ] && [ "${gtot:-0}" -gt "$GPU_TOTAL_MAX" ]; then
        # WSL's nvidia-smi reports per-process used_memory as [N/A], so we identify
        # the offender by the LARGEST RSS among our matched procs (the live trainer,
        # not the tiny conda-run wrapper). On the server this also picks the trainer.
        big=$(echo "$procs" | sort -k2 -nr | head -1 | awk '{print $1}')
        [ -n "${big:-}" ] && kill_proc "$big" "GPU${WATCH_GPU} total ${gtot}MiB >= ${GPU_TOTAL_MAX}MiB (${GPU_PCT}% of ${GPU_TOTAL_MiB}MiB) — largest-RSS of ours"
    fi
    # ---- GPU: per-process cap (server reports real per-proc mem; WSL [N/A] -> skipped) ----
    if [ -n "$procs" ]; then
        while IFS=, read -r gpid gmem; do
            gpid=$(echo "$gpid" | tr -d ' '); gmem=$(echo "$gmem" | tr -d ' ')
            [ -n "$gpid" ] || continue
            case "$gmem" in ''|*[!0-9]*) continue ;; esac   # skip [N/A] / non-numeric (WSL)
            if echo "$procs" | awk '{print $1}' | grep -qx "$gpid" && [ "$gmem" -gt "$GPU_PROC_MAX" ]; then
                kill_proc "$gpid" "GPU proc mem ${gmem}MiB >= ${GPU_PROC_MAX}MiB"
            fi
        done < <("$NVSMI" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
    fi

    # ---- RAM: total-used cap (PRIMARY) + panic + per-process RSS ----
    avail_gb=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$avail_gb" -lt "$RAM_PANIC_GB" ] && [ -n "$procs" ]; then
        log "PANIC MemAvailable=${avail_gb}G < ${RAM_PANIC_GB}G — killing ALL our trainings"
        echo "$procs" | awk '{print $1}' | while read -r p; do kill_proc "$p" "RAM panic"; done
    elif [ "$avail_gb" -lt "$RAM_AVAIL_FLOOR_GB" ] && [ -n "$procs" ]; then
        big=$(echo "$procs" | sort -k2 -nr | head -1 | awk '{print $1}')
        [ -n "$big" ] && kill_proc "$big" "RAM used >= ${RAM_PCT}% (avail ${avail_gb}G < floor ${RAM_AVAIL_FLOOR_GB}G of ${RAM_TOTAL_GB}G) — largest RSS"
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
    # A REAL uvm hang sits in D-state forever with FROZEN disk I/O. Slow 9p /mnt/c I/O
    # (loading full-res images, writing a big .ply) also shows D-state but its
    # /proc/<pid>/io KEEPS ADVANCING. So we only count toward a kill when D-state AND
    # no disk progress — never false-killing a healthy job mid-I/O, however slow.
    if [ -n "$procs" ]; then
        while read -r pid _rss _cpu stat _cmd; do
            case "$stat" in
                D*) io_now=$(awk '/^(read|write)_bytes:/{s+=$2} END{print s+0}' "/proc/$pid/io" 2>/dev/null)
                    if [ -n "${io_prev[$pid]:-}" ] && [ "${io_now:-0}" != "${io_prev[$pid]}" ]; then
                        dstate_count[$pid]=0          # disk advancing -> slow I/O, not a hang
                    else
                        dstate_count[$pid]=$(( ${dstate_count[$pid]:-0} + 1 ))   # D + no disk progress
                    fi
                    io_prev[$pid]=${io_now:-0}
                    if [ "${dstate_count[$pid]}" -ge "$DSTATE_CHECKS" ]; then
                        log "CRITICAL pid=$pid D-state x${dstate_count[$pid]} + frozen disk I/O (nvidia-uvm hang) — attempting kill"
                        kill_proc "$pid" "D-state hang x${dstate_count[$pid]}"
                        dstate_count[$pid]=0
                    fi ;;
                *) dstate_count[$pid]=0; io_prev[$pid]="" ;;
            esac
        done <<< "$procs"
    fi

    sleep "$INTERVAL"
done
