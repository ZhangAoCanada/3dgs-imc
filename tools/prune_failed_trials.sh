#!/usr/bin/env bash
# Delete ONLY the failed/non-best trial dirs listed in _cleanup/REMOVE_dirs.txt, with per-path guards.
# Re-validates EACH path at delete time: must be under an allowed root, must NOT be in KEEP_dirs.txt,
# must NOT contain an aerial-ground marker. DRY-RUN by default; pass --apply to actually delete.
set -euo pipefail
CL=/data1/zhangao/outputs/_cleanup
REMOVE="$CL/REMOVE_dirs.txt"; KEEP="$CL/KEEP_dirs.txt"
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1

ALLOW_RE='^(/data1/zhangao/outputs/diag/|/data1/zhangao/outputs/develop3_multi_angle|/data1/zhangao/outputs/develop3_discontinuous|/data1/zhangao/repos/3dgs-imc-codex/outputs/wsl_results/diag/)'
AERIAL_RE='large_angle|[Zz]eche|[Mm]atrix|[Bb]lock[Aa]|aerial|traintest|somestreet|fusion|_0[Zz]_'

# load KEEP set into an associative array for O(1) membership
declare -A KEEPSET
while IFS=$'\t' read -r _sz path; do [ -n "${path:-}" ] && KEEPSET["$path"]=1; done < "$KEEP"

freed=0; ndel=0; nskip=0
while IFS=$'\t' read -r sz path; do
  [ -n "${path:-}" ] || continue
  # GUARDS
  if ! [[ "$path" =~ $ALLOW_RE ]];        then echo "SKIP(out-of-scope): $path"; nskip=$((nskip+1)); continue; fi
  if [[ "$path" =~ $AERIAL_RE ]];          then echo "SKIP(aerial):       $path"; nskip=$((nskip+1)); continue; fi
  if [ -n "${KEEPSET[$path]:-}" ];         then echo "SKIP(in-KEEP):      $path"; nskip=$((nskip+1)); continue; fi
  if [ ! -d "$path" ];                     then nskip=$((nskip+1)); continue; fi
  # path must be deep enough (never a top-level tree)
  depth=$(awk -F/ '{print NF}' <<<"$path"); if [ "$depth" -lt 6 ]; then echo "SKIP(too-shallow):  $path"; nskip=$((nskip+1)); continue; fi
  if [ "$APPLY" = "1" ]; then rm -rf -- "$path"; fi
  freed=$((freed + sz)); ndel=$((ndel+1))
done < "$REMOVE"

GB=$((1024*1024*1024))
echo "------------------------------------------------------------"
[ "$APPLY" = "1" ] && echo "DELETED $ndel dirs, freed $((freed/GB)) GB (skipped $nskip)" \
                   || echo "DRY-RUN: would delete $ndel dirs, free $((freed/GB)) GB (skipped $nskip). Re-run with --apply."
