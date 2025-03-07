images=input_cached
init_type=sfm
noise_lr=5e5
# cap_max=10000000
cap_max=3000000
scale_reg=0.01
opacity_reg=0.01
densify_from_iter=500
densification_interval=400
depth_l1_weight_init=0.01
depth_l1_weight_final=0.0001
# depth_l1_weight_final=$(echo "scale=10; $depth_l1_weight_init * 0.001" | bc)
port=19888
echo "[training] training with cap_max=${cap_max}, noise_lr=${noise_lr}, scale_reg=${scale_reg}, densify_from_iter=${densify_from_iter}, densification_interval=${densification_interval}, depth_l1_weight_init=${depth_l1_weight_init}, depth_l1_weight_final=${depth_l1_weight_final}"
CUDA_VISIBLE_DEVICES=2 python train.py \
    --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
    --depths dav2_cached \
    --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
    --model_path outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/mcmc_exposure_${init_type}_${cap_max}_${noise_lr}_${depth_l1_weight_init}-${depth_l1_weight_final} \
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