for depth_l1_weight_init in 0.01
do
    for depth_l1_weight_final in 0.001 0.0001
    do
        images=input_cached
        init_type=sfm
        # init_type=random
        noise_lr=5e5
        cap_max=3000000
        scale_reg=0.01
        opacity_reg=0.01
        densify_from_iter=500
        densification_interval=400
        # depth_l1_weight_init=0.1
        # depth_l1_weight_final=0.001
        nn_type=allviews
        bound_type=local
        # bound_type=global
        partial_scaling=1e0
        eps=0.05
        min_samples=200
        port=12331
        echo "[training] training with cap_max=${cap_max}, noise_lr=${noise_lr}, scale_reg=${scale_reg}, densify_from_iter=${densify_from_iter}, densification_interval=${densification_interval}"
        CUDA_VISIBLE_DEVICES=3 python train_experimental_branch2.py \
            --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
            --depths dav2_cached \
            --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
            --model_path outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch2_${init_type}_${nn_type}_${bound_type}_partial${partial_scaling}_${eps}_${min_samples}_${depth_l1_weight_init}_${depth_l1_weight_final} \
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
            --nn_type ${nn_type} \
            --bound_type ${bound_type} \
            --partial_scaling ${partial_scaling} \
            --eps ${eps} \
            --min_samples ${min_samples} \
            --antialiasing \
            --port $port 
    done
done