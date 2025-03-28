images=input_cached
init_type=sfm
source_path=data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train
# test_path=data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_skyroad/test
test_path=data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_skyroad_2nd/test
# test_path=data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test

model_path1=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/mcmc-original
model_path2=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_300_0.1_0.001_wholeonly_1e0_1e-3_sigma-detach_3_0.01_0.01_2000_2500_500

# model_path1=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/mcmc-original-copy
# model_path2=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_300_0.1_0.001_wholeonly_1e0_1e-3_sigma-detach_3_0.01_0.01_2000_2500_500-copy

for model_path in ${model_path1} ${model_path2}
do
    echo "[rendering] rendering with ${model_path}"
    CUDA_VISIBLE_DEVICES=2 python render_camcustomized.py \
        --source_path ${source_path} \
        --test_path ${test_path} \
        --depths "" \
        --model_path ${model_path} \
        --images ${images} \
        --resolution -1 \
        --init_type ${init_type} \
        --data_device cpu \
        --train_test_exp \
        --skip_train \
        --antialiasing \
        # --if_render
done