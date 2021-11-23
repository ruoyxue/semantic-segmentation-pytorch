#!/bin/bash

. ../venv/bin/activate
# export CUDA_VISIBLE_DEVICES=0 # visible GPU device

# change mode to choose either training 0 or testing 1
mode=0;

if [ $mode -eq 0 ];
then
echo "------------Training Process------------" 
python train.py \
--model fcn8s \
--epochs 10 \
--n_class 13 \
--batch_size 8 \
--lr 0.1 \
--device cuda:0 \
--train_data_path ../dataset/semantic_segmentation \
--save_model_path ./save/model_saved/FCN8s \
--check_point_path ./save/checkpoint_saved/FCN8s/checkpoint.pt \
--check_point_mode save \

elif [ $mode -eq 1 ];
then
echo "------------Testing Process-------------"
python test.py \
--model fcn8s \
--n_class 13 \
--batch_size 96 \
--device cuda:0 \
--test_data_path ../dataset/semantic_segmentation \
--load_model_path ./save/model_saved/MLP/model_epoch_30.pth \

fi
exit 0





