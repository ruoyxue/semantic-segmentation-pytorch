#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6 # visible GPU device

# change mode to choose either training 0 or testing 1
mode=0;

if [ $mode -eq 0 ];
then
echo "------------Training Process------------" 
python train.py \
--model fcn8s \
--epochs 300 \
<<<<<<< Updated upstream
--random_seed 312321 \
=======
--random_seed 3428 \
>>>>>>> Stashed changes
--n_class 2 \
--chip_size 512 \
--stride 256 \
--batch_size 16 \
--lr 0.001 \
--device cuda:6 \
--train_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/segmented/train \
--valid_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/segmented/valid \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/FCN8s_add_warmup \
--check_point_mode load \

elif [ $mode -eq 1 ];
then
echo "------------Testing Process-------------"
python test.py \
--model fcn8s \
--n_class 2 \
--chip_size 512 \
--stride 256 \
<<<<<<< Updated upstream
--batch_size 16 \
--device cuda:6 \
--test_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/segmented/test \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/FCN8s_add_warmup \
=======
--batch_size 14 \
--device cuda:1 \
--test_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/new/test \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/UNet \
>>>>>>> Stashed changes

fi
exit 0


# -m torch.utils.bottleneck 



