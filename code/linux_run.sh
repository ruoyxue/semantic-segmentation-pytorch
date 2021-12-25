#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6 # visible GPU device

# change mode to choose either training 0 or testing 1
mode=1;

if [ $mode -eq 0 ];
then
python trainer.py \
--model unet \
--epochs 300 \
--random_seed 423 \
--n_class 2 \
--chip_size 256 \
--stride 128 \
--batch_size 2 \
--lr 0.001 \
--device cuda:0 \
--train_data_path /home/xueruoyao/Documents/PythonProgram/dataset/segmented/train \
--valid_data_path /home/xueruoyao/Documents/PythonProgram/dataset/segmented/valid \
--exp_path /home/xueruoyao/Documents/PythonProgram/exp/UNet \
--check_point_mode load \

elif [ $mode -eq 1 ];
then
python tester.py \
--model unet \
--n_class 2 \
--chip_size 256 \
--stride 128 \
--batch_size 2 \
--device cuda:0 \
--test_data_path /home/xueruoyao/Documents/PythonProgram/dataset/segmented/test \
--exp_path /home/xueruoyao/Documents/PythonProgram/exp/UNet \

fi
exit 0


# -m torch.utils.bottleneck 



