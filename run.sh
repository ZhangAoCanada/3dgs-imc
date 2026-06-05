#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

if [[ "${RUN_LEGACY_SWEEP:-0}" != "1" ]]; then
    # ===== Depth-free GS-LSS (CPF) best config — MipNeRF360 angle-80 =====
    # Counter angle-80, one GPU, ~12 min: beats MH-3DGS on PSNR+SSIM+LPIPS, no DAV2 depth.
    # The CPF is the method; it is developed standalone here (rendered-depth prior).
    export SCENE=${SCENE:-counter}
    export ITERATIONS=${ITERATIONS:-18000}

    # --- CPF (core method), depth-free ---
    export CPF_PRIOR_SOURCE=${CPF_PRIOR_SOURCE:-render}     # rendered-depth prior; NOT DAV2
    export CPF_MAX_POINTS=${CPF_MAX_POINTS:-100000}         # per-train CPF point budget; the 25k cap cost ~0.13dB PSNR
    export CPF_TRAIN_INTERVAL=${CPF_TRAIN_INTERVAL:-8}      # interval throttle is fine (8 >= 4 in quality)
    export CPF_DERIVATIVE_INTERVAL=${CPF_DERIVATIVE_INTERVAL:-8}
    export CPF_MAX_REFRESH_VIEWS=${CPF_MAX_REFRESH_VIEWS:-16}

    # --- 3DGS-MCMC base regularizers (compactness; required for sharp Gaussians) ---
    export SCALE_REG=${SCALE_REG:-0.01}
    export OPACITY_REG=${OPACITY_REG:-0.01}

    # --- DAV2 depth prior: OPTIONAL, OFF by default. To enable, set:
    #   DEPTH_L1_WEIGHT_INIT=1.0 DEPTH_L1_WEIGHT_FINAL=0.01 DEPTH_ALIGN_SFM_ON_DEMAND=1 CPF_PRIOR_SOURCE=sfm_depth
    export DEPTH_L1_WEIGHT_INIT=${DEPTH_L1_WEIGHT_INIT:-0.0}
    export DEPTH_L1_WEIGHT_FINAL=${DEPTH_L1_WEIGHT_FINAL:-0.0}
    export DEPTH_ALIGN_SFM_ON_DEMAND=${DEPTH_ALIGN_SFM_ON_DEMAND:-0}

    # --- base 3DGS-MCMC schedule ---
    export RANDOM_VIEW_NUM=${RANDOM_VIEW_NUM:-1}
    export DENSIFY_FROM_ITER=${DENSIFY_FROM_ITER:-500}
    export DENSIFY_UNTIL_ITER=${DENSIFY_UNTIL_ITER:-12000}
    export DENSIFICATION_INTERVAL=${DENSIFICATION_INTERVAL:-400}
    export TIMING_BREAKDOWN=${TIMING_BREAKDOWN:-1}
    export RUN_EVAL=${RUN_EVAL:-1}
    export LABEL=${LABEL:-${SCENE}_angle80_cpf_nodepth_best_i${ITERATIONS}}
    export MODEL_PATH=${MODEL_PATH:-outputs/fast_lss_dev/${SCENE}_angle80_cpf_nodepth_reg01_pts100k_i${ITERATIONS}_images4}

    STAMP=$(date +%Y%m%d_%H%M%S)
    LOG_PATH=${LOG_PATH:-outputs/monitor/${LABEL}_${STAMP}.log}
    exec scripts/launch_monitored_experiment.sh "${LOG_PATH}" -- bash run_fast_lss_dev.sh
fi

model_path_base=/data1/zhangao/outputs/develop3_multi_angle2
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
for test_selection in test_selection_angle_10.0 test_selection_angle_20.0 test_selection_angle_40.0 test_selection_angle_60.0 test_selection_angle_80.0 test_selection_angle_100.0 
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
        partial_scaling=1e0
        range_scale=1
        noise_method=sigma-detach
        random_view_num=3
        dvoxel_size=0.001
        port=22345
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
            model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_${test_selection}_${dvoxel_size}
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

source_path1=/data1/zhangao/DATA/tandt/truck_resize
cap_max1=2000000
source_path2=/data1/zhangao/DATA/tandt/train_resize
cap_max2=1000000
for test_selection in test_selection_angle_10.0 test_selection_angle_20.0 test_selection_angle_40.0 test_selection_angle_60.0 test_selection_angle_80.0 test_selection_angle_100.0 test_selection_angle_120.0
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
        dvoxel_size=0.01
        port=22345
        eps=0.25
        for partial_scaling in 1e0
        do
            model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_${test_selection}_${dvoxel_size}
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


#######################################################################
#######################################################################
#######################################################################




model_path_base=/data1/zhangao/outputs/develop3_discontinuous
model_path_ext=develop3

source_path1=/data1/zhangao/DATA/mipnerf360/bicycle
# cap_max1=5000000
cap_max1=1000000
source_path2=/data1/zhangao/DATA/mipnerf360/bonsai
# cap_max2=1000000
cap_max2=800000
source_path3=/data1/zhangao/DATA/mipnerf360/counter
# cap_max3=1000000
cap_max3=700000
source_path4=/data1/zhangao/DATA/mipnerf360/flowers
# cap_max4=3000000
cap_max4=800000
source_path5=/data1/zhangao/DATA/mipnerf360/garden
# cap_max5=4000000
cap_max5=800000
source_path6=/data1/zhangao/DATA/mipnerf360/kitchen
# cap_max6=1500000
cap_max6=700000
source_path7=/data1/zhangao/DATA/mipnerf360/room
# cap_max7=1200000
cap_max7=800000
source_path8=/data1/zhangao/DATA/mipnerf360/stump
# cap_max8=4200000
cap_max8=700000
source_path9=/data1/zhangao/DATA/mipnerf360/treehill
# cap_max9=3300000
cap_max9=700000
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
    port=22345
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
            --data_device cpu \
            --antialiasing \
            --skip_train
        python metrics.py \
            --model_path ${model_path} \
            --metrics_save_path ${model_path_base}/metrics/${model_path_ext}_1e-2/${scene_name}_discontinuous_metrics.csv
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
    partial_scaling=1e-2
    range_scale=1
    noise_method=sigma-detach
    random_view_num=3
    dvoxel_size=0.01
    port=22345
    eps=0.25
    for partial_scaling in 1e0
    do
        model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_discontinuous_${dvoxel_size}
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
    elif [ $source_path == $source_path4 ]; then
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
    partial_scaling=1e-2
    range_scale=1
    noise_method=sigma-detach
    random_view_num=3
    dvoxel_size=0.01
    port=22345
    eps=0.25
    if [ ${scene_name} == "residence-small" ]; then
        eps=0.5
    fi
    for partial_scaling in 1e0
    do
        model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_discontinuous_${dvoxel_size}
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



#######################################################################
#######################################################################
#######################################################################



model_path_base=/data1/zhangao/outputs/baselines_aerial_ground
model_path_ext=develop3
source_path=/data1/zhangao/DATA/large_angle/Zeche1

for cap_max in 1000000
do
    scene_name=$(basename $source_path)
    echo "Processing scene: $scene_name"
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
    eps=0.1
    for partial_scaling in 1e0
    do
        model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_woantialiasing
        echo "[*****training*****] training with noise_method=${noise_method}"
        python train_experimental_branch3_mc.py \
            --source_path ${source_path}/train \
            --test_path ${source_path}/test \
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
            --dvoxel_size ${dvoxel_size} \
            --port $port 
        python render.py \
            --source_path ${source_path}/train \
            --test_path ${source_path}/test \
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



for cap_max in 2000000 1000000
do
    scene_name=$(basename $source_path)
    echo "Processing scene: $scene_name"
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
    eps=0.1
    for partial_scaling in 1e0
    do
        model_path=${model_path_base}/${scene_name}/${model_path_ext}_${nn_type}_${bound_type}_eps${eps}_minsamples${min_samples}_depth${depth_l1_weight_init}_${minmax}_rangescale${range_scale}_partial${partial_scaling}_sigma${sigma_scaling}_${noise_method}_nview${random_view_num}_${cap_max}_antialiasing
        echo "[*****training*****] training with noise_method=${noise_method}"
        python train_experimental_branch3_mc.py \
            --source_path ${source_path}/train \
            --test_path ${source_path}/test \
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
            --dvoxel_size ${dvoxel_size} \
            --antialiasing \
            --port $port 
        python render.py \
            --source_path ${source_path}/train \
            --test_path ${source_path}/test \
            --model_path ${model_path} \
            --images ${images} \
            --resolution 1 \
            --data_device cpu \
            --antialiasing \
            --skip_train
        python metrics.py \
            --model_path ${model_path} \
            --metrics_save_path ${model_path_base}/metrics/${model_path_ext}/${scene_name}_antialiasing_metrics.csv
    done
done
