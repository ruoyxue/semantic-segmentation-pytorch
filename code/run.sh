#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6 # visible GPU device

# change mode to choose either training 0 or testing 1
mode=0;

if [ $mode -eq 0 ];
then
python trainer.py \
--model unet \
--epochs 600 \
--random_seed 23423 \
--n_class 2 \
--chip_size 512 \
--stride 256 \
--batch_size 8 \
--lr 0.001 \
--device cuda:7 \
--train_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/1024_segmented/train \
--valid_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/1024_segmented/valid \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/UNet \
--check_point_mode save \

fi

if [ $mode -eq 1 ];
then
python tester.py \
--model fcn8s \
--n_class 2 \
--chip_size 512 \
--stride 256 \
--batch_size 16 \
--device cuda:6 \
--test_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/1024_segmented/test \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/FCN8s \

fi
exit 0
