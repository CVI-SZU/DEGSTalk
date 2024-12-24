dataset=$1
gpu_id=$2

export CUDA_VISIBLE_DEVICES=$gpu_id
# 文件夹的名字
dataset_name=$(basename "$dataset")

python data_utils/process.py $dataset/$dataset_name.mp4