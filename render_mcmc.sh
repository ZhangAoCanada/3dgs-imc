images=input_cached
init_type=sfm
noise_lr=5e5
cap_max=10000000
scale_reg=0.01
opacity_reg=0.01
densify_from_iter=500
densification_interval=400
depth_l1_weight_init=0.01
depth_l1_weight_final=0.0001
model_path=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/new3dgs_exposure_sfm_${cap_max}_${noise_lr}_scale${scale_reg}_opacity0.01_densification-${densify_from_iter}-${densification_interval}
echo "[rendering] rendering with ${model_path}"
CUDA_VISIBLE_DEVICES=1 python train.py \
    --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
    --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
    --model_path ${model_path} \
    --images ${images} \
    --resolution -1 \
    --init_type ${init_type} \
    --antialiasing \
    --port 19999
    # --cap_max ${cap_max} \
    # --data_device cpu \
    # --scale_reg ${scale_reg} \
    # --opacity_reg ${opacity_reg} \
    # --noise_lr ${noise_lr} \
    # --densify_from_iter ${densify_from_iter} \
    # --densification_interval ${densification_interval} \
    # --depth_l1_weight_init ${depth_l1_weight_init} \
    # --depth_l1_weight_final ${depth_l1_weight_final} \