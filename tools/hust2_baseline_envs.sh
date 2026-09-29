#!/usr/bin/env bash
# Build the 4 missing baseline conda envs on hust2 (clone gaussian_splatting + pip-build each repo's
# submodules with --no-build-isolation). Sequential to avoid conda lock contention.
set -uo pipefail
C=/opt/anaconda3/bin/conda
declare -A REPOS=( [mcmc]=/data1/zhangao/repos/3dgs-mcmc [mh]=/data1/zhangao/repos/MH-3DGS
                   [scaffold]=/data1/zhangao/repos/Scaffold-GS [mvgs]=/data1/zhangao/repos/MVGS )
for env in mcmc mh scaffold mvgs; do
  repo=${REPOS[$env]}
  echo "[$env $(date +%T)] start repo=$repo"
  $C env list | grep -q "^$env " || $C create -y --clone gaussian_splatting -n "$env"
  for sm in "$repo"/submodules/*/; do
    [ -d "$sm" ] || continue
    echo "[$env] build $(basename "$sm")"
    $C run -n "$env" pip install --no-build-isolation "$sm" 2>&1 | grep -iE "Successfully built|error:" | tail -1
  done
  $C run -n "$env" python -c "import torch,diff_gaussian_rasterization,simple_knn._C; print('${env}_ENV_OK')" 2>&1 | tail -1
  echo "[$env $(date +%T)] done"
done
echo ALL_BASELINE_ENVS_DONE
