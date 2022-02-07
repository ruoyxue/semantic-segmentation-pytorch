#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6 # visible GPU device

# change mode to choose either training 0 or testing 1
mode=1;

if [ $mode -eq 0 ];
then
python trainer.py \
--model dlinknet34 \
--epochs 600 \
--random_seed 23423 \
--n_class 2 \
--chip_size 1024 \
--stride 512 \
--batch_size 4 \
--lr 0.0001 \
--device cuda:7 \
--train_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/1024_segmented/train \
--valid_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/1024_segmented/valid \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/UNet \
--check_point_mode load \

fi

if [ $mode -eq 1 ];
then
python tester.py \
--model unet \
--n_class 2 \
--chip_size 512 \
--stride 256 \
--batch_size 8 \
--device cuda:6 \
--test_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/1024_segmented/test \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/UNet \

fi
exit 0
