for partial_scaling in 1e0 1e-1 1e-2
do
    for sigma_scaling in 1e0 1e-1 1e-2 1e-3
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
        depth_l1_weight_init=0.1
        depth_l1_weight_final=0.001
        nn_type=allviews
        # bound_type=global
        bound_type=local
        eps=0.05
        min_samples=300
        port=12331
        minmax=wholescale
        range_scale=4
        iteration=30_000
        echo "[training] training with cap_max=${cap_max}, noise_lr=${noise_lr}, scale_reg=${scale_reg}, densify_from_iter=${densify_from_iter}, densification_interval=${densification_interval}"
        CUDA_VISIBLE_DEVICES=3 python train_experimental_branch3.py \
            --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
            --depths dav2_cached \
            --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
            --model_path outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_${init_type}_${nn_type}_${bound_type}_partial${partial_scaling}_${eps}_${min_samples}_${depth_l1_weight_init}_${depth_l1_weight_final}_separate_perturbpnts_mcmc_${minmax}_${range_scale}_all_wonetscale_newxyzscale_${sigma_scaling} \
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
            --sigma_scaling ${sigma_scaling} \
            --eps ${eps} \
            --min_samples ${min_samples} \
            --minmax ${minmax} \
            --range_scale ${range_scale} \
            --iteration ${iteration} \
            --antialiasing \
            --port $port 
    done
done