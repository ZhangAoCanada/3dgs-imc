# eps
# ranges_scale
# depth
# random view
# interval

# for noise_method in sigma-detach opacity-sigma-detach 1-opacity-detach
for eps in 0.1
do
    for partial_scaling in 1e0
    do
        nn_start_iter=2000
        nn_derivatives_iter=$((nn_start_iter+500))
        nn_update_interval=500
        images=images
        depths=dav2
        init_type=sfm
        noise_lr=5e5
        # cap_max=3000000
        cap_max=2000000
        densify_from_iter=500
        densification_interval=200
        depth_l1_weight_init=0.3
        depth_l1_weight_final=$(echo "scale=10; $depth_l1_weight_init * 0.01" | bc)
        nn_type=allviews
        bound_type=local
        sigma_scaling=1e-6
        min_samples=500
        minmax=wholeonly
        # eps=0.05
        # partial_scaling=1e-2
        range_scale=0.5
        # range_scale=1
        noise_method=sigma-detach
        random_view_num=1
        dvoxel_size=0.01
        port=12345
        echo "[*****training*****] training with noise_method=${noise_method}"
        python train_experimental_branch3_mc.py \
            --source_path /mnt/c/Users/Aooooo/Documents/DATA/large_angle/Zeche1/train \
            --test_path /mnt/c/Users/Aooooo/Documents/DATA/large_angle/Zeche1/test \
            --model_path outputs/more/Zeche1/experimental_branch3_${init_type}_${nn_type}_${bound_type}_${eps}_${min_samples}_${depth_l1_weight_init}_${depth_l1_weight_final}_${minmax}_${range_scale}_${partial_scaling}_${sigma_scaling}_${noise_method}_${random_view_num}_${nn_start_iter}_${nn_derivatives_iter}_randmc${nn_update_interval} \
            --images ${images} \
            --depths ${depths} \
            --resolution -1 \
            --init_type ${init_type} \
            --cap_max ${cap_max} \
            --data_device cpu \
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
            --range_scale ${range_scale} \
            --minmax ${minmax} \
            --noise_method ${noise_method} \
            --random_view_num ${random_view_num} \
            --nn_start_iter ${nn_start_iter} \
            --nn_derivatives_iter ${nn_derivatives_iter} \
            --nn_update_interval ${nn_update_interval} \
            --antialiasing \
            --dvoxel_size ${dvoxel_size} \
            --port $port 
        python train_experimental_branch3.py \
            --source_path /mnt/c/Users/Aooooo/Documents/DATA/large_angle/Zeche1/train \
            --test_path /mnt/c/Users/Aooooo/Documents/DATA/large_angle/Zeche1/test \
            --model_path outputs/more/Zeche1/experimental_branch3_${init_type}_${nn_type}_${bound_type}_${eps}_${min_samples}_${depth_l1_weight_init}_${depth_l1_weight_final}_${minmax}_${range_scale}_${partial_scaling}_${sigma_scaling}_${noise_method}_${random_view_num}_${nn_start_iter}_${nn_derivatives_iter}_rand${nn_update_interval} \
            --images ${images} \
            --depths ${depths} \
            --resolution -1 \
            --init_type ${init_type} \
            --cap_max ${cap_max} \
            --data_device cpu \
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
            --range_scale ${range_scale} \
            --minmax ${minmax} \
            --noise_method ${noise_method} \
            --random_view_num ${random_view_num} \
            --nn_start_iter ${nn_start_iter} \
            --nn_derivatives_iter ${nn_derivatives_iter} \
            --nn_update_interval ${nn_update_interval} \
            --antialiasing \
            --dvoxel_size ${dvoxel_size} \
            --port $port 
    done
done
