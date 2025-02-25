images=input_cached
init_type=sfm
model_path=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/new3dgs_exposure_sfm_10000000_5e5_scale0.01_opacity0.01_densification-500-400
echo "[rendering] rendering with ${model_path}"
CUDA_VISIBLE_DEVICES=1 python render.py \
    --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
    --depths dav2_cached \
    --model_path ${model_path} \
    --images ${images} \
    --resolution -1 \
    --init_type ${init_type} \
    --antialiasing \
    # --data_device cpu \