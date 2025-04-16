import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from collections import defaultdict

from tqdm import tqdm

# model_path1 = "outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.3_.003_wholeonly_1e0_1e-3_1-opacity-detach_3_0.01_0.01_2000_2500_500___"
model_path1 = "outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/experimental_branch3_sfm_allviews_local_0.05_500_0.1_0.001_wholeonly_1e0_1e-3_sigma-detach_3_0.01_0.01_2000_2500_500____"
model_path2 = "outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/3dgs-original-until5000_dav2"
# model_path2 = "outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/3dgs-original-until5000"
model_path3 = "outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/original_sfm_5000000-500-400"
# model_path3 = "outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/original_sfm_3000000-500-400"
model_path4 = "outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/pgsr_original_3000000"

sub_dirs = ["train/ours_30000/renders", "test/ours_30000/renders"]

# Create output directory
os.makedirs("tmp", exist_ok=True)

# Process each subdirectory
for sub_dir in sub_dirs:
    
    # Create subdirectory-specific output directory
    output_dir = os.path.join("tmp", sub_dir.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    image_files1 = sorted(glob.glob(os.path.join(model_path1, sub_dir, "*.png")))
    for image_file in tqdm(image_files1):
        image1 = cv2.imread(image_file)
        image2 = cv2.imread(image_file.replace(model_path1, model_path2))
        image3 = cv2.imread(image_file.replace(model_path1, model_path3))
        image4 = cv2.imread(image_file.replace(model_path1, model_path4))
        # resize
        image1 = cv2.resize(image1, None, fx=0.5, fy=0.5)
        image2 = cv2.resize(image2, None, fx=0.5, fy=0.5)
        image3 = cv2.resize(image3, None, fx=0.5, fy=0.5)
        image4 = cv2.resize(image4, None, fx=0.5, fy=0.5)

        if image1 is None or image2 is None or image3 is None or image4 is None:
            print(f"Error reading images for {image_file}")
            continue
        # add text to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text1 = "Ours"
        text2 = "3DGS"
        text3 = "MCMC"
        text4 = "PGSR"
        text_size1 = cv2.getTextSize(text1, font, font_scale, font_thickness)[0]
        text_size2 = cv2.getTextSize(text2, font, font_scale, font_thickness)[0]
        text_size3 = cv2.getTextSize(text3, font, font_scale, font_thickness)[0]
        text_size4 = cv2.getTextSize(text4, font, font_scale, font_thickness)[0]
        text_x1 = 10
        text_y1 = 30
        text_x2 = image1.shape[1] - text_size2[0] - 10
        text_y2 = 30
        text_x3 = 10
        text_y3 = image3.shape[0] - 10
        text_x4 = image4.shape[1] - text_size4[0] - 10
        text_y4 = image4.shape[0] - 10
        cv2.putText(image1, text1, (text_x1, text_y1), font, font_scale, (0, 255, 0), font_thickness)
        cv2.putText(image2, text2, (text_x2, text_y2), font, font_scale, (0, 255, 0), font_thickness)
        cv2.putText(image3, text3, (text_x3, text_y3), font, font_scale, (0, 255, 0), font_thickness)
        cv2.putText(image4, text4, (text_x4, text_y4), font, font_scale, (0, 255, 0), font_thickness)

        row1 = cv2.hconcat([image1, image2])
        row2 = cv2.hconcat([image3, image4])
        combined_image = cv2.vconcat([row1, row2])
        # Save the combined image
        output_file = os.path.join(output_dir, os.path.basename(image_file))
        cv2.imwrite(output_file, combined_image)
