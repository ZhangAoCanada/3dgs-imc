# eps
# ranges_scale
# depth
# random view
# interval

model_path_base=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3
model_path_ext=develop3

source_path1=/mnt/c/Users/Aooooo/Documents/DATA/tandt_db/tandt/truck_resize
cap_max1=2000000
source_path2=/mnt/c/Users/Aooooo/Documents/DATA/tandt_db/tandt/train_resize
cap_max2=1000000
for source_path in $source_path1 $source_path2
do
    if [ $source_path == $source_path1 ]; then
        cap_max=$cap_max1
    else
        cap_max=$cap_max2
    fi
    scene_name=$(basename $source_path)
    echo "Processing scene: $scene_name"
    test_selection_path=${source_path}/test_selection.txt
    images=images
    depths=dav2
    depth_l1_weight_init=0.3
    nn_type=allviews
    bound_type=local
    sigma_scaling=1e-6
    min_samples=500
    minmax=wholeonly
    partial_scaling=1e0
    range_scale=1
    noise_method=sigma-detach
    random_view_num=3
    dvoxel_size=0.001
    port=12345
    eps=0.25
    for partial_scaling in 1e-2 1e0
    do
        model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_randmc
        echo "[*****training*****] training with noise_method=${noise_method}"
        python train_experimental_branch3_mc.py \
            --source_path ${source_path} \
            --test_selection_path ${test_selection_path} \
            --model_path ${model_path} \
            --images ${images} \
            --depths ${depths} \
            --cap_max ${cap_max} \
            --depth_l1_weight_init ${depth_l1_weight_init} \
            --partial_scaling ${partial_scaling} \
            --nn_type ${nn_type} \
            --bound_type ${bound_type} \
            --sigma_scaling ${sigma_scaling} \
            --eps ${eps} \
            --min_samples ${min_samples} \
            --range_scale ${range_scale} \
            --minmax ${minmax} \
            --noise_method ${noise_method} \
            --random_view_num ${random_view_num} \
            --resolution 1 \
            --data_device cpu \
            --antialiasing \
            --dvoxel_size ${dvoxel_size} \
            --port $port 
    done
done

source_path1=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/building-small
cap_max1=7000000
source_path2=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/residence-small
cap_max2=6000000
source_path3=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/rubble-small
cap_max3=6000000
source_path4=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/sci-art-small
cap_max4=1000000
for source_path in $source_path1 $source_path2 $source_path3 $source_path4
do
    if [ $source_path == $source_path1 ]; then
        cap_max=$cap_max1
    elif [ $source_path == $source_path2 ]; then
        cap_max=$cap_max2
    elif [ $source_path == $source_path3 ]; then
        cap_max=$cap_max3
    else
        cap_max=$cap_max4
    fi
    scene_name=$(basename $source_path)
    echo "Processing scene: $scene_name"
    test_selection_path=${source_path}/test_selection.txt
    images=images_4
    depths=dav2_4
    depth_l1_weight_init=0.3
    nn_type=allviews
    bound_type=local
    sigma_scaling=1e-6
    min_samples=500
    minmax=wholeonly
    partial_scaling=1e0
    range_scale=1
    noise_method=sigma-detach
    random_view_num=3
    dvoxel_size=0.001
    port=12345
    eps=0.25
    if ${scene_name} == "residence-small"; then
        eps=0.5
    fi
    for range_scale in 1e-2 1e0
    do
        model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_randnmc
        echo "[*****training*****] training with noise_method=${noise_method}"
        python train_experimental_branch3_mc.py \
            --source_path ${source_path} \
            --test_selection_path ${test_selection_path} \
            --model_path ${model_path} \
            --images ${images} \
            --depths ${depths} \
            --cap_max ${cap_max} \
            --depth_l1_weight_init ${depth_l1_weight_init} \
            --partial_scaling ${partial_scaling} \
            --nn_type ${nn_type} \
            --bound_type ${bound_type} \
            --sigma_scaling ${sigma_scaling} \
            --eps ${eps} \
            --min_samples ${min_samples} \
            --range_scale ${range_scale} \
            --minmax ${minmax} \
            --noise_method ${noise_method} \
            --random_view_num ${random_view_num} \
            --resolution 1 \
            --data_device cpu \
            --antialiasing \
            --dvoxel_size ${dvoxel_size} \
            --port $port 
    done
done

# source_path1=/mnt/c/Users/Aooooo/Documents/DATA/large_angle/Zeche1
# cap_max1=2000000
# for source_path in $source_path1
# do
#     if [ $source_path == $source_path1 ]; then
#         cap_max=$cap_max1
#     fi
#     scene_name=$(basename $source_path)
#     echo "Processing scene: $scene_name"
#     train_path=${source_path}/train
#     test_path=${source_path}/test
#     images=images
#     depths=dav2
#     depth_l1_weight_init=0.3
#     nn_type=allviews
#     bound_type=local
#     sigma_scaling=1e-6
#     min_samples=500
#     minmax=wholeonly
#     partial_scaling=1e0
#     range_scale=1
#     noise_method=sigma-detach
#     random_view_num=1
#     dvoxel_size=0.001
#     port=12345
#     for eps in 0.05 0.01 0.1 0.5
#     do
#         model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_randnmc
#         echo "[*****training*****] training with noise_method=${noise_method}"
#         python train_experimental_branch3_mc.py \
#             --source_path ${train_path} \
#             --test_path ${test_path} \
#             --model_path ${model_path} \
#             --images ${images} \
#             --depths ${depths} \
#             --cap_max ${cap_max} \
#             --depth_l1_weight_init ${depth_l1_weight_init} \
#             --partial_scaling ${partial_scaling} \
#             --nn_type ${nn_type} \
#             --bound_type ${bound_type} \
#             --sigma_scaling ${sigma_scaling} \
#             --eps ${eps} \
#             --min_samples ${min_samples} \
#             --range_scale ${range_scale} \
#             --minmax ${minmax} \
#             --noise_method ${noise_method} \
#             --random_view_num ${random_view_num} \
#             --resolution 1 \
#             --data_device cpu \
#             --antialiasing \
#             --dvoxel_size ${dvoxel_size} \
#             --port $port
# done

# source_path1=/mnt/c/Users/Aooooo/Documents/DATA/large_angle/blockA_fusion_small_aerial+somestreet
# cap_max1=7000000
# for source_path in $source_path1
# do
#     if [ $source_path == $source_path1 ]; then
#         cap_max=$cap_max1
#     fi
#     scene_name=$(basename $source_path)
#     echo "Processing scene: $scene_name"
#     train_path=${source_path}/train
#     test_path=${source_path}/test
#     images=input_cached
#     depths=dav2_cached
#     depth_l1_weight_init=0.3
#     nn_type=allviews
#     bound_type=local
#     sigma_scaling=1e-6
#     min_samples=500
#     minmax=wholeonly
#     partial_scaling=1e0
#     range_scale=1
#     noise_method=sigma-detach
#     random_view_num=1
#     dvoxel_size=0.001
#     port=12345
#     for eps in 0.05 0.01 0.1 0.5
#     do
#         model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_randnmc
#         echo "[*****training*****] training with noise_method=${noise_method}"
#         python train_experimental_branch3_mc.py \
#             --source_path ${train_path} \
#             --test_path ${test_path} \
#             --model_path ${model_path} \
#             --images ${images} \
#             --depths ${depths} \
#             --cap_max ${cap_max} \
#             --depth_l1_weight_init ${depth_l1_weight_init} \
#             --partial_scaling ${partial_scaling} \
#             --nn_type ${nn_type} \
#             --bound_type ${bound_type} \
#             --sigma_scaling ${sigma_scaling} \
#             --eps ${eps} \
#             --min_samples ${min_samples} \
#             --range_scale ${range_scale} \ 
#             --minmax ${minmax} \
#             --noise_method ${noise_method} \
#             --random_view_num ${random_view_num} \
#             --resolution 1 \
#             --data_device cpu \
#             --antialiasing \
#             --dvoxel_size ${dvoxel_size} \
#             --port $port
# done

# source_path1=/mnt/c/Users/Aooooo/Documents/DATA/DroneSplat/Sculpture
# cap_max1=6000000
# source_path2=/mnt/c/Users/Aooooo/Documents/DATA/DroneSplat/Simingshan
# cap_max2=2000000
# for source_path in $source_path1 $source_path2
# do
#     if [ $source_path == $source_path1 ]; then
#         cap_max=$cap_max1
#     else
#         cap_max=$cap_max2
#     fi
#     scene_name=$(basename $source_path)
#     echo "Processing scene: $scene_name"
#     test_selection_path=${source_path}/test_selection.txt
#     images=images
#     depths=dav2
#     depth_l1_weight_init=0.3
#     nn_type=allviews
#     bound_type=local
#     sigma_scaling=1e-6
#     min_samples=500
#     minmax=wholeonly
#     partial_scaling=1e0
#     range_scale=1
#     noise_method=sigma-detach
#     random_view_num=1
#     dvoxel_size=0.001
#     port=12345
#     for eps in 0.1 0.25
#     do
#         model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_randnmc
#         echo "[*****training*****] training with noise_method=${noise_method}"
#         python train_experimental_branch3_mc.py \
#             --source_path ${source_path} \
#             --test_selection_path ${test_selection_path} \
#             --model_path ${model_path} \
#             --images ${images} \
#             --depths ${depths} \
#             --cap_max ${cap_max} \
#             --depth_l1_weight_init ${depth_l1_weight_init} \
#             --partial_scaling ${partial_scaling} \
#             --nn_type ${nn_type} \
#             --bound_type ${bound_type} \
#             --sigma_scaling ${sigma_scaling} \
#             --eps ${eps} \
#             --min_samples ${min_samples} \
#             --range_scale ${range_scale} \
#             --minmax ${minmax} \
#             --noise_method ${noise_method} \
#             --random_view_num ${random_view_num} \
#             --resolution 1 \
#             --data_device cpu \
#             --antialiasing \
#             --dvoxel_size ${dvoxel_size} \
#             --port $port
#     done
# done

# source_path1=/mnt/c/Users/Aooooo/Documents/DATA/HorizonGS/real/park
# cap_max1=7000000
# source_path2=/mnt/c/Users/Aooooo/Documents/DATA/HorizonGS/real/road
# cap_max2=6000000
# for source_path in $source_path1 $source_path2
# do
#     if [ $source_path == $source_path1 ]; then
#         cap_max=$cap_max1
#     else
#         cap_max=$cap_max2
#     fi
#     scene_name=$(basename $source_path)
#     echo "Processing scene: $scene_name"
#     train_path=${source_path}/train
#     test_path=${source_path}/test
#     images=images
#     depths=dav2
#     depth_l1_weight_init=0.3
#     nn_type=allviews
#     bound_type=local
#     sigma_scaling=1e-6
#     min_samples=500
#     minmax=wholeonly
#     partial_scaling=1e0
#     range_scale=1
#     noise_method=sigma-detach
#     random_view_num=1
#     dvoxel_size=0.001
#     port=12345
#     for eps in 5.0 10.0
#     do
#         model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_randnmc
#         echo "[*****training*****] training with noise_method=${noise_method}"
#         python train_experimental_branch3_mc.py \
#             --source_path ${train_path} \
#             --test_path ${test_path} \
#             --model_path ${model_path} \
#             --images ${images} \
#             --depths ${depths} \
#             --cap_max ${cap_max} \
#             --depth_l1_weight_init ${depth_l1_weight_init} \
#             --partial_scaling ${partial_scaling} \
#             --nn_type ${nn_type} \
#             --bound_type ${bound_type} \
#             --sigma_scaling ${sigma_scaling} \
#             --eps ${eps} \
#             --min_samples ${min_samples} \
#             --range_scale ${range_scale} \
#             --minmax ${minmax} \
#             --noise_method ${noise_method} \
#             --random_view_num ${random_view_num} \
#             --resolution 1 \
#             --data_device cpu \
#             --antialiasing \
#             --dvoxel_size ${dvoxel_size} \
#             --port $port
#     done
# done

