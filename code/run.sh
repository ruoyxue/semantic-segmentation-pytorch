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
--random_seed 1 \
--n_class 2 \
--batch_size 16 \
--lr 0.001 \
--device cuda:1 \
--train_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/new/train \
--valid_data_path /data/xueruoyao/dataset/road_extraction/deepglobe/new/valid \
--save_model_path /data/xueruoyao/experiment/road_extraction/deepglobe/FCN8s/train/model_saved \
--exp_train_path /data/xueruoyao/experiment/road_extraction/deepglobe/FCN8s/train \
--check_point_path /data/xueruoyao/experiment/road_extraction/deepglobe/FCN8s/train/checkpoint_saved/checkpoint.pt \
--check_point_mode save \

elif [ $mode -eq 1 ];
then
echo "------------Testing Process-------------"
python test.py \
--model fcn8s \
--n_class 2 \
--chip_size 256 \
--stride 128 \
--batch_size 4 \
--device cuda:0 \
--test_data_path ../dataset/semantic_segmentation/original \
--load_model_path ./save/model_saved/FCN8s/model_epoch_10.pth \
--save_output_path ./save/prediction_saved/FCN8s

fi
exit 0


# -m torch.utils.bottleneck 



