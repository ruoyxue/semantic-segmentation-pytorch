#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6 # visible GPU device

# change mode to choose either training 0 or testing 1
mode=1;

if [ $mode -eq 0 ];
then
echo "------------Training Process------------" 
python train.py \
--model unet \
--epochs 300 \
--random_seed 2536323 \
--n_class 2 \
--chip_size 512 \
--stride 256 \
--batch_size 8 \
--lr 0.01 \
--device cuda:7 \
--train_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/new/train \
--valid_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/new/valid \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/UNet \
--check_point_mode save \

elif [ $mode -eq 1 ];
then
echo "------------Testing Process-------------"
python test.py \
--model unet \
--n_class 2 \
--chip_size 512 \
--stride 256 \
--batch_size 16 \
--device cuda:1 \
--test_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/origin \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/UNet \

fi
exit 0


# -m torch.utils.bottleneck 



