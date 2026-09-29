#!/usr/bin/env bash
# WSL crash-guard watchdog. Polls GPU memory % and system RAM %; if either exceeds its cap for
# 2 consecutive checks, kills the GS-LSS training process to keep WSL from crashing (the nvidia-uvm
# OOM hang). Conservative + hysteresis so a brief spike doesn't kill a healthy run.
NVSMI=${NVSMI:-/usr/lib/wsl/lib/nvidia-smi}
GPU_CAP=${GPU_CAP:-94}      # % of GPU memory
RAM_CAP=${RAM_CAP:-92}      # % of system RAM
INT=${INT:-15}             # seconds between checks
STRIKES=${STRIKES:-2}      # consecutive over-cap checks before killing
LOG=${LOG:-/home/ooooo/wsl_watchdog.log}
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
say "wsl watchdog START GPU_CAP=${GPU_CAP}% RAM_CAP=${RAM_CAP}% int=${INT}s strikes=${STRIKES}"
strk=0
while :; do
  line=$("$NVSMI" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')
  used=$(awk '{print $1+0}' <<<"$line"); total=$(awk '{print $2+0}' <<<"$line")
  gpu=0; [ "${total:-0}" -gt 0 ] && gpu=$(( used*100/total ))
  ram=$(free | awk '/Mem:/{printf "%d", ($2-$7)*100/$2}')
  if [ "${gpu:-0}" -ge "$GPU_CAP" ] || [ "${ram:-0}" -ge "$RAM_CAP" ]; then
    strk=$(( strk + 1 ))
    say "OVER-CAP #${strk}: GPU=${gpu}% (${used}/${total}MiB) RAM=${ram}%"
    if [ "$strk" -ge "$STRIKES" ]; then
      say "!! KILLING training (GPU=${gpu}% RAM=${ram}%) to prevent crash"
      pkill -9 -f train_experimental_branch3 2>/dev/null
      pkill -9 -f "python render.py" 2>/dev/null
      strk=0; sleep 30
    fi
  else
    strk=0
  fi
  sleep "$INT"
done
