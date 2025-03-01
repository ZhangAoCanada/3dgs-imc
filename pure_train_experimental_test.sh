for depth_l1_weight_init in 0.1 0.01
do
    for depth_l1_weight_final in 0.01 0.001 0.0001
    do
        images=input_cached
        init_type=sfm
        # init_type=random
        noise_lr=5e5
        cap_max=3500000
        scale_reg=0.01
        opacity_reg=0.01
        densify_from_iter=500
        densification_interval=400
        # depth_l1_weight_init=0.1
        # depth_l1_weight_final=0.01
        experimental_schedule_gamma=0.9
        experimental_randomview_num=5
        port=12321
        echo "[training] training with cap_max=${cap_max}, noise_lr=${noise_lr}, scale_reg=${scale_reg}, densify_from_iter=${densify_from_iter}, densification_interval=${densification_interval}"

        CUDA_VISIBLE_DEVICES=2 python train_experimental2.py \
            --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
            --depths dav2_cached \
            --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
            --model_path outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_${init_type}_350w_combine+std_randomview${experimental_randomview_num}_lr5_dynamicbound_fft_depthnormalize_${depth_l1_weight_init}_${depth_l1_weight_final} \
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
            --experimental_schedule_gamma ${experimental_schedule_gamma} \
            --antialiasing \
            --depthnorm \
            --port $port 
    done
done
