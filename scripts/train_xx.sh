dataset=$1
workspace=$2
gpu_id=$3
other_args=$4

export CUDA_VISIBLE_DEVICES=$gpu_id

if [ ! -f "$workspace/chkpnt_mouth_latest.pth" ]; then
    python train_mouth.py -s $dataset -m $workspace $other_args
fi
echo "$workspace/chkpnt_mouth_latest.pth exists"

if [ ! -f "$workspace/chkpnt_face_latest.pth" ]; then
    python train_face_w_hair.py -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005 $other_args
fi
echo "$workspace/chkpnt_face_latest.pth exists"

if [ ! -f "$workspace/chkpnt_fuse_latest.pth" ]; then
    python train_fuse_w_hair.py -s $dataset -m $workspace --opacity_lr 0.001 $other_args
fi
echo "$workspace/chkpnt_fuse_latest.pth exists"

# # Parallel. Ensure that you have aleast 2 GPUs and over 64GB memory.
# CUDA_VISIBLE_DEVICES=$gpu_id python train_mouth.py -s $dataset -m $workspace &
# CUDA_VISIBLE_DEVICES=$((gpu_id+1)) python train_face.py -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005
# CUDA_VISIBLE_DEVICES=$gpu_id python train_fuse.py -s $dataset -m $workspace --opacity_lr 0.001

python synthesize_fuse.py -s $dataset -m $workspace  --eval $other_args
python metrics.py $workspace/test/ours_None/renders/out.mp4 $workspace/test/ours_None/gt/out.mp4