for partial_scaling in 1e0 1e-1 1e-2 1e-3 1e-4 1e-5 1e1
do
    images=input_cached
    init_type=sfm
    # init_type=random
    noise_lr=5e5
    cap_max=3500000
    scale_reg=0.01
    # scale_reg=0.0
    opacity_reg=0.01
    # opacity_reg=0.0
    densify_from_iter=500
    densification_interval=400
    # densification_interval=300
    depth_l1_weight_init=0.1
    depth_l1_weight_final=0.001
    port=12311
    echo "[training] training with cap_max=${cap_max}, noise_lr=${noise_lr}, scale_reg=${scale_reg}, densify_from_iter=${densify_from_iter}, densification_interval=${densification_interval}"
    CUDA_VISIBLE_DEVICES=1 python train_experimental_branch2_singleview.py \
        --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
        --depths dav2_cached \
        --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
        --model_path outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch2_${init_type}_fixnum_singleview_unaligned_partial${partial_scaling}_sized_boundxyz \
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
        --partial_scaling ${partial_scaling} \
        --antialiasing \
        --port $port 
done