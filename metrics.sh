########################################
# NOTE: run render.sh before metrics.sh
########################################
export CUDA_VISIBLE_DEVICES=0

model_path1=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/truck_resize/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1_partial1e-2_sigma1e-6_sigma-detach_nview3_2000000_randmc
model_path2=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/train_resize/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1_partial1e-2_sigma1e-6_sigma-detach_nview3_1000000_randmc

model_path3=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/building-small/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1e0_partial1e0_sigma1e-6_sigma-detach_nview3_7000000_randnmc
model_path4=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/residence-small/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1e-2_partial1e0_sigma1e-6_sigma-detach_nview3_6000000_randnmc
model_path5=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/rubble-small/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1e0_partial1e0_sigma1e-6_sigma-detach_nview3_6000000_randnmc
model_path6=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/develop3/sci-art-small/develop3_allviews_local_eps0.25_minsamples500_depth0.3_wholeonly_rangescale1e0_partial1e0_sigma1e-6_sigma-detach_nview3_1000000_randnmc
for model_path in $model_path1 $model_path2 $model_path3 $model_path4 $model_path5 $model_path6
do
    python metrics.py -m ${model_path} 
done
