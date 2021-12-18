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
--random_seed 432432 \
--n_class 2 \
--chip_size 256 \
--stride 128 \
--batch_size 4 \
--lr 0.01 \
--device cuda:0 \
--train_data_path /home/xueruoyao/Documents/PythonProgram/dataset/new/train/ \
--valid_data_path /home/xueruoyao/Documents/PythonProgram/dataset/new/valid/ \
--exp_path /home/xueruoyao/Documents/PythonProgram/exp/UNet \
--check_point_mode load \

elif [ $mode -eq 1 ];
then
echo "------------Testing Process-------------"
python test.py \
--model unet \
--n_class 2 \
--chip_size 512 \
--stride 256 \
--batch_size 14 \
--device cuda:1 \
--test_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/new/test \
--exp_path /data/xueruoyao/experiment/road_extraction/deepglobe/UNet \

fi
exit 0


# -m torch.utils.bottleneck 



