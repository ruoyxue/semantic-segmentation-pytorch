#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6 # visible GPU device

# change mode to choose either training 0 or testing 1
mode=1;

if [ $mode -eq 0 ];
then
echo "------------Training Process------------" 
python train.py \
--model fcn8s \
--epochs 300 \
--random_seed 1 \
--n_class 2 \
--batch_size 16 \
--lr 0.001 \
--device cuda:1 \
--train_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/new/train \
--valid_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/new/valid \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/FCN8s \
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
--device cuda:0 \
--test_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/new/test \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/FCN8s \

fi
exit 0


# -m torch.utils.bottleneck 



