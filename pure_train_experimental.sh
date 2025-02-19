images=input_cached
init_type=sfm
noise_lr=5e5
cap_max=10000000
scale_reg=0.01
# scale_reg=0.0
opacity_reg=0.01
# opacity_reg=0.0
densify_from_iter=500
densification_interval=400
# densification_interval=300
depth_l1_weight_init=0.01
depth_l1_weight_final=0.0001
port=12331
echo "[training] training with cap_max=${cap_max}, noise_lr=${noise_lr}, scale_reg=${scale_reg}, densify_from_iter=${densify_from_iter}, densification_interval=${densification_interval}"
CUDA_VISIBLE_DEVICES=3 python train_experimental.py \
    --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
    --depths dav2_cached \
    --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
    --model_path outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_sfm_fixnum_nn2ndseparatestdalltry_dynamicsingle_-11 \
    --images ${images} \
    --resolution -1 \
    --init_type ${init_type} \
    --cap_max ${cap_max} \
    --data_device cpu \
    --scale_reg ${scale_reg} \
    --opacity_reg ${opacity_reg} \
    --noise_lr ${noise_lr} \
    --densify_from_iter ${densify_from_iter} \
    --densification_interval ${densification_interval} \
    --depth_l1_weight_init ${depth_l1_weight_init} \
    --depth_l1_weight_final ${depth_l1_weight_final} \
    --antialiasing \
    --port $port 