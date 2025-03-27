# NOTE: run render.sh before metrics.sh
model_path=outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/mcmc-original
CUDA_VISIBLE_DEVICES=2 python metrics.py -m ${model_path} 