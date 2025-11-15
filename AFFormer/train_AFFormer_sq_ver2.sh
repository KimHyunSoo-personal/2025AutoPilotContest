CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29505 train_AFFormer_ver2.py \
 --result_dir "./pths/AFFormer_base_001" \
 --epochs 40 \
 --lr 2.e-2 \
 --loadpath "./AFFormer_base_cityscapes.pth" \
 --scale_range [0.75,1.25] \
 --crop_size [1024,1024] \
 --batch_size 16 \
 --dataset_dir "./SemanticDataset_final" \
 


