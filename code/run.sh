#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6 # visible GPU device

# change mode to choose either training 0 or testing 1
mode=0;

if [ $mode -eq 0 ];
then
echo "------------Training Process------------" 
python train.py \
--model unet \
--epochs 300 \
--random_seed 422432 \
--n_class 2 \
--chip_size 512 \
--stride 256 \
--batch_size 8 \
--lr 0.01 \
--device cuda:7 \
--train_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/segmented/train \
--valid_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/segmented/valid \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/UNet \
--check_point_mode save \

elif [ $mode -eq 1 ];
then
echo "------------Testing Process-------------"
python test.py \
--model fcn8s \
--n_class 2 \
--chip_size 512 \
--stride 256 \
--batch_size 16 \
--device cuda:6 \
--test_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/segmented/test \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/FCN8s \

fi
exit 0


# -m torch.utils.bottleneck 



