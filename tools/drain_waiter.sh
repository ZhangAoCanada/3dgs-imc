#!/usr/bin/env bash
# Thin wake-signal: blocks until BOTH lanes (server probe_queue + WSL dev_queue) are
# drained for 2 consecutive checks, then exits so the (harness-tracked) launcher notifies
# the agent. The actual compute + healing is the durable nohup-setsid sentinel; this only wakes us.
set -uo pipefail
SVR_Q=/data1/zhangao/outputs/probe_queue
WSL_Q=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/dev_queue
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 wsl"
drained=0
while true; do
    s=$(( $(ls "$SVR_Q"/pending/*.job 2>/dev/null | wc -l) + $(ls "$SVR_Q"/running/*.job 2>/dev/null | wc -l) ))
    w=$($SSH "echo \$(( \$(ls $WSL_Q/pending/*.job 2>/dev/null | wc -l) + \$(ls $WSL_Q/running/*.job 2>/dev/null | wc -l) ))" 2>/dev/null)
    w=${w:-99}
    if [ "${s:-99}" -eq 0 ] && [ "$w" -eq 0 ]; then drained=$((drained+1)); else drained=0; fi
    [ "$drained" -ge 2 ] && break
    sleep 120
done
echo "BOTH LANES DRAINED at $(date '+%F %T') — 15k A/B + tuning sweep complete"
