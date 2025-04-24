########################################
# NOTE: run render.sh before metrics.sh
########################################
# model_path1=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.3_.003_wholeonly_1e0_1e-3_1-opacity-detach_3_0.01_0.01_2000_2500_500___
# model_path2=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.1_0.001_wholeonly_1e0_1e-3_sigma-detach_3_0.01_0.01_2000_2500_500____
# model_path3=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/3dgs-original-until5000_dav2
# model_path4=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/3dgs-original-until5000
# model_path5=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/original_sfm_5000000-500-400
# model_path6=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/original_sfm_3000000-500-400
# model_path7=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/pgsr_original_3000000




# model_path1=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.1_.001_wholeonly_1e-2_1e-3_sigma-detach_1_0.01_0.01_2000_2500_500_ablation_retry
# model_path2=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.1_.001_wholeonly_1e-2_1e-3_sigma-detach_2_0.01_0.01_2000_2500_500_ablation_retry
# model_path3=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.1_.001_wholeonly_1e-2_1e-3_sigma-detach_3_0.01_0.01_2000_2500_500_ablation_retry
# model_path4=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.1_.001_wholeonly_1e-2_1e-3_sigma-detach_1_0.01_0.01_2000_2500_500_ablation_mean+mcmc_1
# model_path5=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.1_.001_wholeonly_1e-2_1e-3_sigma-detach_2_0.01_0.01_2000_2500_500_ablation_mean+mcmc_1
# model_path6=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.1_.001_wholeonly_1e-2_1e-3_sigma-detach_3_0.01_0.01_2000_2500_500_ablation_mean+mcmc_002
# model_path7=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.1_.001_wholeonly_1e-2_1e-3_sigma-detach_2_0.01_0.01_2000_2500_500_ablation_variance_1
# model_path8=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.1_.001_wholeonly_1e-2_1e-3_sigma-detach_3_0.01_0.01_2000_2500_500_ablation_variance_001


# model_path1=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/original_sfm_3000000-500-400



model_path1=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/3dgs-original-until5000_dav2
model_path2=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/3dgs-original-until5000
model_path3=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/pgsr_original_3000000

for model_path in ${model_path1} ${model_path2} ${model_path3}
do
    CUDA_VISIBLE_DEVICES=1 python metrics.py -m ${model_path} 
done