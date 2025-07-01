#!/bin/bash

# Set environment variables
export WANDB_API_KEY="4fb601e9329ccdcb5fdd99d556d5c8365d92301b"
export WANDB_BASE_URL="https://wandb.nicehiro.org"
export HF_DATASETS_CACHE=/data/fywang/Calvin-hf-ds-cache/hf_datasets_cache

# Run the training script with all parameters from launch.json
/root/lerobot/.venv/bin/python lerobot/scripts/train.py \
    --batch_size 256 \
    --steps 100000 \
    --log_freq 10 \
    --eval_freq -1 \
    --num_workers 8 \
    --output_dir "output/train/smolvla_calvin_task_ABCD_D" \
    --policy.type smolvla \
    --policy.push_to_hub false \
    --policy.vlm_model_name "/model/fywang/SmolVLM2/SmolVLM2-500M-Video-Instruct" \
    --policy.chunk_size 8 \
    --policy.n_action_steps 8 \
    --dataset.repo_id "fywang/calvin-task-ABCD-D-lerobot" \
    --dataset.root "/data/fywang/Calvin/task_ABCD_D/lerobot_v2_dataset" \
    --wandb.enable true \
    --wandb.project "lerobot" \
    --wandb.entity "fywang96" \
