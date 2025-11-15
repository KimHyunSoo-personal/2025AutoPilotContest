CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29600 prediction_SegFormer.py \
 --dataset_dir "./SemanticDatasetTest" \
 --weight_path "./pths/SegFormer_003/model_best.pth" \
 --result_dir "./result_SegFormer" \
 --num_classes 19 \
 --input_size 1200 1920