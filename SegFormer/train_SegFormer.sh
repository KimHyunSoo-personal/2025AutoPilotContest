CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29505 train_SegFormer.py \
 --result_dir "./pths/SegFormer_003" \
 --epochs 40 \
 --lr 1.e-4 \
 --loadpath "./mit_b0.pth" \
 --scale_range [0.75,1.25] \
 --crop_size [512,512] \
 --batch_size 16 \
 --dataset_dir "./SemanticDataset_final" \
 


