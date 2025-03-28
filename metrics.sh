########################################
# NOTE: run render.sh before metrics.sh
########################################
model_path1=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/mcmc-original
model_path2=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_300_0.1_0.001_wholeonly_1e0_1e-3_sigma-detach_3_0.01_0.01_2000_2500_500

# model_path1=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/mcmc-original-copy
# model_path2=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_300_0.1_0.001_wholeonly_1e0_1e-3_sigma-detach_3_0.01_0.01_2000_2500_500-copy

for model_path in ${model_path1} ${model_path2}
do
    CUDA_VISIBLE_DEVICES=2 python metrics.py -m ${model_path} 
done