source_path1=/mnt/c/Users/Aooooo/Documents/DATA/tandt_db/tandt/truck_resize
source_path2=/mnt/c/Users/Aooooo/Documents/DATA/tandt_db/tandt/train_resize
# source_path3=/mnt/c/Users/Aooooo/Documents/DATA/large_angle/Zeche1/train
# source_path4=/mnt/c/Users/Aooooo/Documents/DATA/large_angle/Zeche1/test
source_path5=/mnt/c/Users/Aooooo/Documents/DATA/DroneSplat/Sculpture
source_path6=/mnt/c/Users/Aooooo/Documents/DATA/DroneSplat/Simingshan
images=images
dav2=dav2
for source_path in ${source_path1} ${source_path2} ${source_path5} ${source_path6}
do
    cd ../Depth-Anything-V2
    python run.py --encoder vitl --pred-only --grayscale --img-path ${source_path}/${images} --outdir ${source_path}/${dav2}
    cd ../3dgs-imc
    python utils/make_depth_scale.py --base_dir ${source_path} --depths_dir ${source_path}/${dav2}
done

source_path1=/mnt/c/Users/Aooooo/Documents/DATA/HorizonGS/real/park/train
source_path2=/mnt/c/Users/Aooooo/Documents/DATA/HorizonGS/real/road/train
images=images
dav2=dav2
image_subdirs="aerial_h aerial_q aerial_x aerial_y aerial_z"
for source_path in ${source_path1} ${source_path2}
do
    for subdir in ${image_subdirs}
    do
        image_dir=${source_path}/${images}/${subdir}
        dav2_dir=${source_path}/${dav2}/${subdir}
        cd ../Depth-Anything-V2
        python run.py --encoder vitl --pred-only --grayscale --img-path ${image_dir} --outdir ${dav2_dir}
    done
    cd ../3dgs-imc
    python utils/make_depth_scale.py --base_dir ${source_path} --depths_dir ${source_path}/${dav2}
done
source_path1=/mnt/c/Users/Aooooo/Documents/DATA/HorizonGS/real/park/test
image_subdirs="street_cam1 street_cam5 street_cam6"
for source_path in ${source_path1}
do
    for subdir in ${image_subdirs}
    do
        image_dir=${source_path}/${images}/${subdir}
        dav2_dir=${source_path}/${dav2}/${subdir}
        cd ../Depth-Anything-V2
        python run.py --encoder vitl --pred-only --grayscale --img-path ${image_dir} --outdir ${dav2_dir}
    done
    cd ../3dgs-imc
    python utils/make_depth_scale.py --base_dir ${source_path} --depths_dir ${source_path}/${dav2}
done
source_path1=/mnt/c/Users/Aooooo/Documents/DATA/HorizonGS/real/road/test
image_subdirs="street_cam3 street_cam4"
for source_path in ${source_path1}
do
    for subdir in ${image_subdirs}
    do
        image_dir=${source_path}/${images}/${subdir}
        dav2_dir=${source_path}/${dav2}/${subdir}
        cd ../Depth-Anything-V2
        python run.py --encoder vitl --pred-only --grayscale --img-path ${image_dir} --outdir ${dav2_dir}
    done
    cd ../3dgs-imc
    python utils/make_depth_scale.py --base_dir ${source_path} --depths_dir ${source_path}/${dav2}
done

source_path1=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/building-small
source_path2=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/residence-small
source_path3=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/rubble-small
source_path4=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/sci-art-small
images=images_4
dav2=dav2_4
for source_path in ${source_path1} ${source_path2} ${source_path3} ${source_path4}
do
    cd ../Depth-Anything-V2
    python run.py --encoder vitl --pred-only --grayscale --img-path ${source_path}/${images} --outdir ${source_path}/${dav2}
    cd ../3dgs-imc
    python utils/make_depth_scale.py --base_dir ${source_path} --depths_dir ${source_path}/${dav2}
done

# source_path1=/mnt/c/Users/Aooooo/Documents/DATA/large_angle/blockA_fusion_small_aerial+somestreet/train
# source_path2=/mnt/c/Users/Aooooo/Documents/DATA/large_angle/blockA_fusion_small_aerial+somestreet/test
# images=input_cached
# dav2=dav2_cached
# for source_path in ${source_path1} ${source_path2}
# do
#     cd ../Depth-Anything-V2
#     python run.py --encoder vitl --pred-only --grayscale --img-path ${source_path}/${images} --outdir ${source_path}/${dav2}
#     cd ../3dgs-imc
#     python utils/make_depth_scale.py --base_dir ${source_path} --depths_dir ${source_path}/${dav2}
# done