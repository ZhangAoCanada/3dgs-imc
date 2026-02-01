source_path1=/mnt/c/Users/Aooooo/Documents/DATA/tandt_db/tandt/truck_resize
source_path2=/mnt/c/Users/Aooooo/Documents/DATA/tandt_db/tandt/train_resize
for source_path in $source_path1 $source_path2
do
    scene_name=$(basename ${source_path})
    if [ ${scene_name} == "truck_resize" ]; then
        image_name=000013.jpg
    elif [ ${scene_name} == "train_resize" ]; then
        image_name=00161.jpg
    fi
    python select_testdata.py \
        -s ${source_path} \
        -i ${image_name} \
        -n 5 10 15 20
done
source_path1=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/bicycle
source_path2=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/bonsai
source_path3=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/counter
source_path4=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/flowers
source_path5=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/garden
source_path6=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/kitchen
source_path7=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/room
source_path8=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/stump
source_path9=/mnt/c/Users/Aooooo/Documents/DATA/mipnerf360/treehill
for source_path in $source_path1 $source_path2 $source_path3 $source_path4 $source_path5 $source_path6 $source_path7 $source_path8 $source_path9
do
    scene_name=$(basename ${source_path})
    if [ ${scene_name} == "bicycle" ]; then #
        image_name=_DSC8780.JPG
    elif [ ${scene_name} == "bonsai" ]; then
        image_name=DSCF5798.JPG
    elif [ ${scene_name} == "counter" ]; then
        image_name=DSCF5859.JPG
    elif [ ${scene_name} == "flowers" ]; then #
        image_name=_DSC9128.JPG
    elif [ ${scene_name} == "garden" ]; then # 
        image_name=DSC08075.JPG
    elif [ ${scene_name} == "kitchen" ]; then
        image_name=DSCF0885.JPG
    elif [ ${scene_name} == "room" ]; then
        image_name=DSCF4816.JPG
    elif [ ${scene_name} == "stump" ]; then #
        image_name=_DSC9281.JPG
    elif [ ${scene_name} == "treehill" ]; then #
        image_name=_DSC8932.JPG
    fi
    python select_testdata.py \
        -s ${source_path} \
        -i ${image_name} \
        -n 5 10 15 20
done
source_path1=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/building-small
source_path2=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/residence-small
source_path3=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/rubble-small
source_path4=/mnt/c/Users/Aooooo/Documents/DATA/DATA-small/sci-art-small
for source_path in $source_path1 $source_path2 $source_path3 $source_path4
do
    scene_name=$(basename ${source_path})
    if [ ${scene_name} == "building-small" ]; then
        image_name=000159.jpg
    elif [ ${scene_name} == "residence-small" ]; then
        image_name=000217.JPG
    elif [ ${scene_name} == "rubble-small" ]; then
        image_name=000074.jpg
    elif [ ${scene_name} == "sci-art-small" ]; then
        image_name=000089.JPG
    fi
    python select_testdata.py \
        -s ${source_path} \
        -i ${image_name} \
        -n 5 10 15 20
done
