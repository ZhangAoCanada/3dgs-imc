export CUDA_VISIBLE_DEVICES=0

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
    echo "rendering scene: $scene_name"
    test_selection_path=${source_path}/test_selection.txt
    if [ $source_path == $source_path1 ]; then
        model_path=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/truck_resize/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1_partial1e-2_sigma1e-6_sigma-detach_nview3_2000000_randmc
    else
        model_path=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/train_resize/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1_partial1e-2_sigma1e-6_sigma-detach_nview3_1000000_randmc
    fi
    python render_camcustomized.py \
        --model_path ${model_path} \
        --data_device cpu \
        --resolution 1 \
        --antialiasing \
        --skip_train
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
    if [ $source_path == $source_path1 ]; then
        model_path=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/building-small/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1e0_partial1e0_sigma1e-6_sigma-detach_nview3_7000000_randnmc
    elif [ $source_path == $source_path2 ]; then
        model_path=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/residence-small/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1e-2_partial1e0_sigma1e-6_sigma-detach_nview3_6000000_randnmc
    elif [ $source_path == $source_path3 ]; then
        model_path=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/rubble-small/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1e0_partial1e0_sigma1e-6_sigma-detach_nview3_6000000_randnmc
    else
        model_path=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/sci-art-small/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1e0_partial1e0_sigma1e-6_sigma-detach_nview3_1000000_randnmc
    fi
    python render_camcustomized.py \
        --model_path ${model_path} \
        --data_device cpu \
        --resolution 1 \
        --antialiasing \
        --skip_train
done