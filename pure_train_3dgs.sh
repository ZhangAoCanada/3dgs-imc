CUDA_VISIBLE_DEVICES=1 python train.py \
    -s data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
    -d dav2_cached \
    --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
    -m outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/3dgs_original_depth-0.1~0.001_antialiasing \
    --images input_cached \
    --antialiasing \
    --depth_l1_weight_init 0.1 \
    --depth_l1_weight_final 0.001 \
    --port 2888 \
    --data_device cpu


CUDA_VISIBLE_DEVICES=1 python train.py \
    -s data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train \
    -d dav2_cached \
    --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test \
    -m outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/3dgs_original_depth-0.01~0.0001_antialiasing \
    --images input_cached \
    --antialiasing \
    --depth_l1_weight_init 0.01 \
    --depth_l1_weight_final 0.0001 \
    --port 2888 \
    --data_device cpu

