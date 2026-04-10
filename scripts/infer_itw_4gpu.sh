#!/bin/bash
GPU_IDS=4,5,6,7
SEED=846514

export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Enable expandable segments to reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Using GPUs: $GPU_IDS (cuda:0->GPU4, cuda:1->GPU5, cuda:2->GPU6, cuda:3->GPU7)"
echo "Single process + module-level device assignment"
echo "Transformer split: blocks[0:16] -> cuda:0, blocks[16:40] -> cuda:1"

python3 infer_multi_gpu.py \
    --meta_data_file ./example/in_the_wild/meta.json \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --lora_path ./checkpoints/EgoX/pytorch_lora_weights.safetensors \
    --lora_rank 256 \
    --out ./results_4gpu_lorarank256 \
    --seed $SEED \
    --use_GGA \
    --cos_sim_scaling_factor 3.0 \
    --in_the_wild \
    --idx 0