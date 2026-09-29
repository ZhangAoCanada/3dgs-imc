#!/usr/bin/env bash
# Fix B preprocessing v2: train-only PatchMatch-MVS prior per (scene, angle) on WSL.
# Leak-proof via explicit cfg files (no image_deleter, no __auto__ source selection):
#   per scene  : undistort the FULL model once (cached base workspace, --max_image_size 800)
#   per angle  : copy base -> workspace, install train-only patch-match.cfg + fusion.cfg
#                (generated server-side by tools/make_mvs_cfgs.py), run stereo + fusion.
#   tools/build_mvs_prior.sh <scene> <angle> [<scene> <angle> ...]
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WDATA=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360
WWORK=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/mvs_priors
SOUT=$REPO/outputs/mvs_priors
PY=/home/zhangao/.conda/envs/3dgs-imc/bin/python
mkdir -p "$SOUT"

while [ $# -ge 2 ]; do
    SC=$1; ANG=$2; shift 2
    TAG="${SC}_angle_${ANG}"
    echo "===== MVS prior: $TAG ====="
    mkdir -p "$SOUT/$TAG/cfg"
    "$PY" "$REPO/tools/make_mvs_cfgs.py" "$SC" "$ANG" "$SOUT/$TAG/cfg"
    ssh -o BatchMode=yes wsl "mkdir -p '$WWORK/$TAG/cfg'"
    rsync -az "$SOUT/$TAG/cfg/" "wsl:$WWORK/$TAG/cfg/"
    # shellcheck disable=SC2087
    ssh -o BatchMode=yes wsl bash -s <<EOF
set -euo pipefail
BASE="$WWORK/${SC}_dense_base"
W="$WWORK/$TAG"
if [ -f "\$W/fused.ply" ]; then echo "[mvs] fused.ply exists, skip"; exit 0; fi
# Undistort at NATIVE resolution (consistent cameras+images); patch_match downsamples both
# together via PatchMatchStereo.max_image_size. (undistorter --max_image_size resized only the
# JPEGs here, leaving cameras at full res -> width-mismatch abort.)
if [ ! -d "\$BASE/images" ]; then
    echo "[mvs] building base undistorted workspace for $SC (native res)"
    colmap image_undistorter --image_path "$WDATA/$SC/images" --input_path "$WDATA/$SC/sparse/0" \
        --output_path "\$BASE" > "\$BASE.log" 2>&1
fi
rm -rf "\$W/dense"; mkdir -p "\$W/dense/stereo"
cp -r "\$BASE/images" "\$BASE/sparse" "\$W/dense/"
mkdir -p "\$W/dense/stereo/depth_maps" "\$W/dense/stereo/normal_maps" "\$W/dense/stereo/consistency_graphs"
cp "\$W/cfg/patch-match.cfg" "\$W/cfg/fusion.cfg" "\$W/dense/stereo/"
colmap patch_match_stereo --workspace_path "\$W/dense" \
    --PatchMatchStereo.gpu_index 0 --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.max_image_size 1000 --PatchMatchStereo.cache_size 32 > "\$W/pm.log" 2>&1
colmap stereo_fusion --workspace_path "\$W/dense" --output_path "\$W/fused.ply" > "\$W/fuse.log" 2>&1
ls -la "\$W/fused.ply"
EOF
    rsync -az "wsl:$WWORK/$TAG/fused.ply" "$SOUT/$TAG/fused.ply"
    echo "[mvs] pulled -> $SOUT/$TAG/fused.ply ($(du -h "$SOUT/$TAG/fused.ply" | cut -f1))"
done
echo "[mvs] all requested priors built"
