# images=input_cached
# init_type=sfm
# # model_path=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_300_0.1_0.001_wholeonly_1e0_1e-3_sigma-detach_3_0.01_0.01
# model_path=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/mcmc-original
# echo "[rendering] rendering with ${model_path}"
# CUDA_VISIBLE_DEVICES=2 python render.py \
#     --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
#     --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_skyroad/test \
#     --depths "" \
#     --model_path ${model_path} \
#     --images ${images} \
#     --resolution -1 \
#     --init_type ${init_type} \
#     --antialiasing \
#     --data_device cpu \
#     --train_test_exp \
#     --skip_train \

images=input_cached
init_type=sfm
# model_path=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_300_0.1_0.001_wholeonly_1e0_1e-3_sigma-detach_3_0.01_0.01
# model_path=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/mcmc-original
model_path=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_300_0.1_0.001_wholeonly_1e0_1e-3_opacity-sigma-detach_3_0.01_0.01
echo "[rendering] rendering with ${model_path}"
CUDA_VISIBLE_DEVICES=2 python render_camcustomized.py \
    --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
    --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_skyroad/test \
    --depths "" \
    --model_path ${model_path} \
    --images ${images} \
    --resolution -1 \
    --init_type ${init_type} \
    --antialiasing \
    --data_device cpu \
    --train_test_exp \
    --skip_train \
    --if_render 