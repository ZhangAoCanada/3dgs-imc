images=input_cached
init_type=sfm
source_path=data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train
# test_path=data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_skyroad/test
# test_path=data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_skyroad_2nd/test
test_path=data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test

model_path1=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.3_.003_wholeonly_1e0_1e-3_1-opacity-detach_3_0.01_0.01_2000_2500_500___
model_path2=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.1_0.001_wholeonly_1e0_1e-3_sigma-detach_3_0.01_0.01_2000_2500_500____
model_path3=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/3dgs-original-until5000_dav2
model_path4=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/3dgs-original-until5000
model_path5=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/original_sfm_5000000-500-400
model_path6=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/original_sfm_3000000-500-400
model_path7=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/pgsr_original_3000000

for model_path in ${model_path1} ${model_path2} ${model_path3} ${model_path4} ${model_path5} ${model_path6} ${model_path7}
do
    echo "[rendering] rendering with ${model_path}"
    CUDA_VISIBLE_DEVICES=0 python render_camcustomized.py \
        --source_path ${source_path} \
        --test_path ${test_path} \
        --depths "" \
        --model_path ${model_path} \
        --images ${images} \
        --resolution -1 \
        --init_type ${init_type} \
        --data_device cpu \
        --antialiasing \
        # --skip_train \
        # --train_test_exp \
        # --if_render
done