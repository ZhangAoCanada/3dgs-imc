# images=input_cached
# init_type=sfm
# noise_lr=5e5
# cap_max=3000000
# scale_reg=0.01
# opacity_reg=0.01
# densify_from_iter=500
# densification_interval=400
# depth_l1_weight_init=0.1
# depth_l1_weight_final=0.001
# nn_type=allviews
# bound_type=local
# eps=0.05
# min_samples=300
# port=12321
# minmax=wholescale
# range_scale=4
# partial_scaling=1e0
# sigma_scaling=1e0
# # noise_method=opacity-sigma-detach
# noise_method=sigma-detach
# random_view_num=3
# echo "[training] training with cap_max=${cap_max}, noise_lr=${noise_lr}, scale_reg=${scale_reg}, densify_from_iter=${densify_from_iter}, densification_interval=${densification_interval}"
# CUDA_VISIBLE_DEVICES=2 python train_experimental_branch3.py \
#     --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
#     --depths dav2_cached \
#     --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
#     --model_path outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_${init_type}_${nn_type}_${bound_type}_${eps}_${min_samples}_${depth_l1_weight_init}_${depth_l1_weight_final}_${minmax}_${range_scale}_${partial_scaling}_${sigma_scaling}_${noise_method}_${random_view_num}_${scale_reg}_${opacity_reg} \
#     --images ${images} \
#     --resolution -1 \
#     --init_type ${init_type} \
#     --cap_max ${cap_max} \
#     --data_device cpu \
#     --scale_reg ${scale_reg} \
#     --opacity_reg ${opacity_reg} \
#     --noise_lr ${noise_lr} \
#     --densify_from_iter ${densify_from_iter} \
#     --densification_interval ${densification_interval} \
#     --depth_l1_weight_init ${depth_l1_weight_init} \
#     --depth_l1_weight_final ${depth_l1_weight_final} \
#     --nn_type ${nn_type} \
#     --bound_type ${bound_type} \
#     --partial_scaling ${partial_scaling} \
#     --sigma_scaling ${sigma_scaling} \
#     --eps ${eps} \
#     --min_samples ${min_samples} \
#     --minmax ${minmax} \
#     --range_scale ${range_scale} \
#     --noise_method ${noise_method} \
#     --random_view_num ${random_view_num} \
#     --antialiasing \
#     --port $port 


for nn_start_iter in 1000 1500 2500 3000
do
    nn_derivatives_iter=$((nn_start_iter+500))
    nn_update_interval=1000
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
    bound_type=local
    partial_scaling=1e0
    sigma_scaling=1e-3
    eps=0.05
    min_samples=300
    port=12321
    minmax=wholeonly
    # noise_method=opacity-sigma-detach
    noise_method=sigma-detach
    random_view_num=3
    echo "[*****training*****] training with noise_method=${noise_method}"
    CUDA_VISIBLE_DEVICES=2 python train_experimental_branch3.py \
        --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
        --depths dav2_cached \
        --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
        --model_path outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_${init_type}_${nn_type}_${bound_type}_${eps}_${min_samples}_${depth_l1_weight_init}_${depth_l1_weight_final}_${minmax}_${partial_scaling}_${sigma_scaling}_${noise_method}_${random_view_num}_${scale_reg}_${opacity_reg}_${nn_start_iter}_${nn_derivatives_iter}_${nn_update_interval} \
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
        --noise_method ${noise_method} \
        --random_view_num ${random_view_num} \
        --nn_start_iter ${nn_start_iter} \
        --nn_derivatives_iter ${nn_derivatives_iter} \
        --nn_update_interval ${nn_update_interval} \
        --antialiasing \
        --port $port 
done
