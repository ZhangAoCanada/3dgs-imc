# eps
# ranges_scale
# depth
# random view
# interval

model_path_base=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3_multi
model_path_ext=develop3

source_path1=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/bicycle
cap_max1=5000000
source_path2=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/bonsai
cap_max2=1000000
source_path3=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/counter
cap_max3=1000000
source_path4=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/flowers
cap_max4=3000000
source_path5=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/garden
cap_max5=4000000
source_path6=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/kitchen
cap_max6=1500000
source_path7=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/room
cap_max7=1200000
source_path8=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/stump
cap_max8=4200000
source_path9=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/treehill
cap_max9=3300000
for test_selection in test_selection_angle_5.0 test_selection_angle_10.0 test_selection_angle_15.0 test_selection_angle_20.0 test_selection_angle_30.0 test_selection_angle_40.0 test_selection_angle_50.0 test_selection_angle_60.0 test_selection_angle_70.0 test_selection_angle_80.0 test_selection_angle_90.0 test_selection_angle_100.0 test_selection_angle_110.0 test_selection_angle_120.0 test_selection_angle_130.0 test_selection_angle_140.0 test_selection_angle_150.0 test_selection_angle_160.0 test_selection_angle_170.0 test_selection_angle_180.0
do
    for source_path in $source_path1 $source_path2 $source_path3 $source_path4 $source_path5 $source_path6 $source_path7 $source_path8 $source_path9
    do
        if [ $source_path == $source_path1 ]; then
            cap_max=$cap_max1
        elif [ $source_path == $source_path2 ]; then
            cap_max=$cap_max2
        elif [ $source_path == $source_path3 ]; then
            cap_max=$cap_max3
        elif [ $source_path == $source_path4 ]; then
            cap_max=$cap_max4
        elif [ $source_path == $source_path5 ]; then
            cap_max=$cap_max5
        elif [ $source_path == $source_path6 ]; then
            cap_max=$cap_max6
        elif [ $source_path == $source_path7 ]; then
            cap_max=$cap_max7
        elif [ $source_path == $source_path8 ]; then
            cap_max=$cap_max8
        elif [ $source_path == $source_path9 ]; then
            cap_max=$cap_max9
        fi
        scene_name=$(basename $source_path)
        echo "Processing scene: $scene_name"
        test_selection_path=${source_path}/${test_selection}.txt
        images=images_4
        depths=dav2_4
        depth_l1_weight_init=0.3
        nn_type=allviews
        bound_type=local
        sigma_scaling=1e-6
        min_samples=500
        minmax=wholeonly
        partial_scaling=1e-2
        range_scale=1
        noise_method=sigma-detach
        random_view_num=3
        dvoxel_size=0.001
        port=12345
        eps=0.25
        if [ $scene_name == "bonsai" ]; then
            eps=0.5
        elif [ $scene_name == "room" ]; then
            eps=0.5
        fi
        for depth_l1_weight_init in 0.0 0.3
        do
            model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_${test_selection}
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
done

source_path1=/mnt/c/Users/Aooooo/Documents/DATA/tandt_db/tandt/truck_resize
cap_max1=2000000
source_path2=/mnt/c/Users/Aooooo/Documents/DATA/tandt_db/tandt/train_resize
cap_max2=1000000
for test_selection in test_selection_angle_5.0 test_selection_angle_10.0 test_selection_angle_15.0 test_selection_angle_20.0 test_selection_angle_30.0 test_selection_angle_40.0 test_selection_angle_50.0 test_selection_angle_60.0 test_selection_angle_70.0 test_selection_angle_80.0 test_selection_angle_90.0 test_selection_angle_100.0 test_selection_angle_110.0 test_selection_angle_120.0 test_selection_angle_130.0 test_selection_angle_140.0 test_selection_angle_150.0 test_selection_angle_160.0 test_selection_angle_170.0 test_selection_angle_180.0
do
    for source_path in $source_path1 $source_path2
    do
        if [ $source_path == $source_path1 ]; then
            cap_max=$cap_max1
        else
            cap_max=$cap_max2
        fi
        scene_name=$(basename $source_path)
        echo "Processing scene: $scene_name"
        test_selection_path=${source_path}/${test_selection}.txt
        images=images
        depths=dav2
        depth_l1_weight_init=0.3
        nn_type=allviews
        bound_type=local
        sigma_scaling=1e-6
        min_samples=500
        minmax=wholeonly
        partial_scaling=1e-2
        range_scale=1
        noise_method=sigma-detach
        random_view_num=3
        dvoxel_size=0.001
        port=12345
        eps=0.25
        for depth_l1_weight_init in 0.0 0.3
        do
            model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_${test_selection}
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
done

source_path1=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/building-small
cap_max1=7000000
source_path2=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/residence-small
cap_max2=6000000
source_path3=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/rubble-small
cap_max3=6000000
source_path4=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/sci-art-small
cap_max4=1000000
for test_selection in test_selection_5 test_selection_10 test_selection_15 test_selection_20 test_selection_25 test_selection_30 test_selection_35 test_selection_40 test_selection_45 test_selection_50 test_selection_60 test_selection_70 test_selection_80 test_selection_90 test_selection_100
do
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
        test_selection_path=${source_path}/${test_selection}.txt
        images=images_4
        depths=dav2_4
        depth_l1_weight_init=0.3
        nn_type=allviews
        bound_type=local
        sigma_scaling=1e-6
        min_samples=500
        minmax=wholeonly
        partial_scaling=1e-2
        range_scale=1
        noise_method=sigma-detach
        random_view_num=3
        dvoxel_size=0.001
        port=12345
        eps=0.25
        if [ ${scene_name} == "residence-small" ]; then
            eps=0.5
        fi
        for depth_l1_weight_init in 0.0 0.3
        do
            model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_${test_selection}
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
