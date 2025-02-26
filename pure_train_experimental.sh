for experimental_randomview_num in 1 2 3 4 5 6 7 8 9 10
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
    depth_l1_weight_init=0.01
    depth_l1_weight_final=0.0001
    experimental_schedule_gamma=0.9
    port=12321
    echo "[training] training with cap_max=${cap_max}, noise_lr=${noise_lr}, scale_reg=${scale_reg}, densify_from_iter=${densify_from_iter}, densification_interval=${densification_interval}"

    CUDA_VISIBLE_DEVICES=2 python train_experimental2.py \
        --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
        --depths dav2_cached \
        --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
        --model_path outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_${init_type}_350w_combine+std_randomview${experimental_randomview_num}_lr5_dynamicbound_fft \
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
        --port $port 
done