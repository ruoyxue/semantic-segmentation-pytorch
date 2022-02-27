#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6 # visible GPU device

# change mode to choose either training 0 or testing 1
mode=0;


if [ $mode -eq 0 ];
then
python trainer.py \
--model seghrnet \
--epochs 300 \
--random_seed 24355464765 \
--n_class 2 \
--chip_size 1024 \
--stride 512 \
--batch_size 4 \
--lr 0.0005 \
--device cuda:7 \
--train_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/1024_segmented/train \
--valid_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/1024_segmented/valid \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/seghrnet \
--check_point_mode save \

fi

if [ $mode -eq 1 ];
then
python tester.py \
--model seghrnet \
--n_class 2 \
--chip_size 1024 \
--stride 512 \
--batch_size 2 \
--device cuda:7 \
--test_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/1024_segmented/test \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/seghrnet \

fi
exit 0
