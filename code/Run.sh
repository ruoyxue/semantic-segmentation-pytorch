. ../venv/bin/activate
# export CUDA_VISIBLE_DEVICES=0 # visible GPU device

# change mode to choose either training 0 or testing 1
mode=0;

if [ $mode -eq 0 ];
then
echo "------------Training Process------------" 
python Train.py \
--model mlp \
--epochs 100 \
--batchsize 192 \
--lr 0.1 \
--device cuda:0 \
--train_data_path ../dataset/train \
--save_model_path ./save/model_saved/MLP \
--check_point_path ./save/checkpoint_saved/MLP \
--check_point_mode load \

elif [ $mode -eq 1 ];
then
echo "------------Testing Process-------------"
python Test.py \
--model mlp \
--batchsize 96 \
--device cuda:0 \
--test_data_path ../dataset/train \
--load_model_path ./save/model_saved/MLP/model_epoch_30.pth \

fi







