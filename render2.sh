
export CUDA_VISIBLE_DEVICES=0

model_path_base=/data1/zhangao/outputs/develop3_discontinuous_woantialiasing
model_path_ext=develop3

source_path1=/data1/zhangao/DATA/mipnerf360/bicycle
cap_max1=5000000
source_path2=/data1/zhangao/DATA/mipnerf360/bonsai
cap_max2=1000000
source_path3=/data1/zhangao/DATA/mipnerf360/counter
cap_max3=1000000
source_path4=/data1/zhangao/DATA/mipnerf360/flowers
cap_max4=3000000
source_path5=/data1/zhangao/DATA/mipnerf360/garden
cap_max5=4000000
source_path6=/data1/zhangao/DATA/mipnerf360/kitchen
cap_max6=1500000
source_path7=/data1/zhangao/DATA/mipnerf360/room
cap_max7=1200000
source_path8=/data1/zhangao/DATA/mipnerf360/stump
cap_max8=4200000
source_path9=/data1/zhangao/DATA/mipnerf360/treehill
cap_max9=3300000
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
    test_selection_path=${source_path}/discontinuous.txt
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
    port=32345
    eps=0.25
    if [ ${scene_name} == "bicycle" ]; then 
        depth_l1_weight_init=0.3
    elif [ ${scene_name} == "bonsai" ]; then
        depth_l1_weight_init=0.1
        eps=0.5
    elif [ ${scene_name} == "counter" ]; then
        depth_l1_weight_init=0.0
    elif [ ${scene_name} == "flowers" ]; then
        depth_l1_weight_init=0.3
        eps=0.5
    elif [ ${scene_name} == "garden" ]; then
        depth_l1_weight_init=0.3
    elif [ ${scene_name} == "kitchen" ]; then
        depth_l1_weight_init=0.1
    elif [ ${scene_name} == "room" ]; then
        depth_l1_weight_init=0.3
        eps=0.5
    elif [ ${scene_name} == "stump" ]; then 
        depth_l1_weight_init=0.1
        partial_scaling=1e-2
    elif [ ${scene_name} == "treehill" ]; then 
        depth_l1_weight_init=0.3
    fi
    for partial_scaling in 1e0
    do
        model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_discontinuous_${dvoxel_size}
        echo "[*****training*****] training with noise_method=${noise_method}"
        python render.py \
            --source_path ${source_path} \
            --test_selection_path ${test_selection_path} \
            --model_path ${model_path} \
            --images ${images} \
            --resolution 1 \
            --data_device cpu \
            --skip_train
        python metrics.py \
            --model_path ${model_path} \
            --metrics_save_path ${model_path_base}/metrics/${model_path_ext}/${scene_name}_metrics.csv
    done
done

source_path1=/data1/zhangao/DATA/tandt/truck_resize
cap_max1=2000000
source_path2=/data1/zhangao/DATA/tandt/train_resize
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
    test_selection_path=${source_path}/discontinuous.txt
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
    dvoxel_size=0.01
    port=32345
    eps=0.25
    for partial_scaling in 1e0
    do
        model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_discontinuous_${dvoxel_size}
        echo "[*****training*****] training with noise_method=${noise_method}"
        python render.py \
            --source_path ${source_path} \
            --test_selection_path ${test_selection_path} \
            --model_path ${model_path} \
            --images ${images} \
            --resolution 1 \
            --data_device cpu \
            --skip_train
        python metrics.py \
            --model_path ${model_path} \
            --metrics_save_path ${model_path_base}/metrics/${model_path_ext}/${scene_name}_metrics.csv
    done
done

source_path1=/data1/zhangao/DATA/DATA-small/building-small
cap_max1=7000000
source_path2=/data1/zhangao/DATA/DATA-small/residence-small
cap_max2=6000000
source_path3=/data1/zhangao/DATA/DATA-small/rubble-small
cap_max3=6000000
source_path4=/data1/zhangao/DATA/DATA-small/sci-art-small
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
    test_selection_path=${source_path}/discontinuous.txt
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
    dvoxel_size=0.01
    port=32345
    eps=0.25
    if [ ${scene_name} == "residence-small" ]; then
        eps=0.5
    fi
    for partial_scaling in 1e0
    do
        model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_discontinuous_${dvoxel_size}
        echo "[*****training*****] training with noise_method=${noise_method}"
        python render.py \
            --source_path ${source_path} \
            --test_selection_path ${test_selection_path} \
            --model_path ${model_path} \
            --images ${images} \
            --resolution 1 \
            --data_device cpu \
            --skip_train
        python metrics.py \
            --model_path ${model_path} \
            --metrics_save_path ${model_path_base}/metrics/${model_path_ext}/${scene_name}_metrics.csv
    done
done