#!/usr/bin/env bash
# Dead-simple SEQUENTIAL queue drainer (no daemon, no concurrency, no race). Drains QDIR/pending
# one job at a time, blocking, on CUDA_VISIBLE_DEVICES. Use for the server's big-cap scenes that
# must run 1-wide anyway. Launch via Bash run_in_background; it exits when the queue is empty.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDIR=${QDIR:-/data1/zhangao/outputs/probe_queue}
PEND="$QDIR/pending"; RUN="$QDIR/running"; DONE="$QDIR/done"; LOGS="$QDIR/logs"
mkdir -p "$RUN" "$DONE" "$LOGS"
echo $$ > "$QDIR/seq.pid"; trap 'rm -f "$QDIR/seq.pid"' EXIT   # liveness pidfile (sentinel checks kill -0; robust vs pgrep self-match)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
echo "[seq $(date +%T)] start, gpu=$CUDA_VISIBLE_DEVICES pending=$(ls "$PEND"/*.job 2>/dev/null|wc -l)"
while [ ! -f "$QDIR/STOP" ]; do
    jf=$(ls -1 "$PEND"/*.job 2>/dev/null | sort | head -1)
    [ -z "$jf" ] && break
    name=$(basename "$jf" .job)
    mv "$jf" "$RUN/$name.job" 2>/dev/null || continue
    echo "[seq $(date +%T)] RUN $name"
    ( set -a; source "$RUN/$name.job"; set +a; export TAG="$name"; bash diag_run.sh ) > "$LOGS/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then mv "$RUN/$name.job" "$DONE/" 2>/dev/null; else mv "$RUN/$name.job" "$QDIR/failed/" 2>/dev/null; mkdir -p "$QDIR/failed"; fi
    echo "[seq $(date +%T)] DONE $name rc=$rc"
done
echo "[seq $(date +%T)] queue drained"
