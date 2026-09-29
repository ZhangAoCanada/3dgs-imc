#!/usr/bin/env bash
# Block until the Stage-3 diagnostic has something actionable, then exit (harness re-invokes me).
# Exits when: a DG_* run produced metrics.csv | the launcher logged COMPLETE | the launcher died.
# Bounded to ~2h so it returns for re-assessment even if the GPU stays blocked.
OUTD=/data1/zhangao/outputs/diag/large_angle
LOG="$OUTD/_diagnostic.log"
for i in $(seq 1 40); do          # 40 * 180s = 2h cap
  if ls "$OUTD"/*DG_*/metrics.csv >/dev/null 2>&1; then echo "RESULT_READY"; exit 0; fi
  if grep -q "diagnostic COMPLETE" "$LOG" 2>/dev/null; then echo "DIAG_COMPLETE"; exit 0; fi
  if ! pgrep -f run_table3_diagnostic >/dev/null 2>&1; then echo "LAUNCHER_GONE"; exit 0; fi
  sleep 180
done
echo "STILL_WAITING_GPU"; exit 0
