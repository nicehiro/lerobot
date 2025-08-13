#!/bin/bash

# Default values
GPUS=4
BATCH_SIZE=256
STEPS=100000
LR=1e-4
SEED=42
OUTPUT_DIR="output/train/smolvla_calvin_task_ABCD_D"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Set environment variables
export WANDB_API_KEY="4fb601e9329ccdcb5fdd99d556d5c8365d92301b"
export WANDB_BASE_URL="https://wandb.nicehiro.org"
export HF_DATASETS_CACHE=/data/fywang/Calvin-hf-ds-cache/hf_datasets_cache

# Run the training script with torch distributed
# /root/lerobot/.venv/bin/python -m accelerate.commands.launch --num_processes=${GPUS} --mixed_precision=fp16 lerobot/scripts/train.py \
/root/lerobot/.venv/bin/python lerobot/scripts/train.py \
    --batch_size ${BATCH_SIZE} \
    --steps ${STEPS} \
    --log_freq 10 \
    --eval_freq -1 \
    --save_freq 5000 \
    --num_workers 16 \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --policy.type smolvla \
    --policy.push_to_hub false \
    --policy.use_amp true \
    --policy.vlm_model_name "/model/fywang/SmolVLM2/SmolVLM2-500M-Video-Instruct" \
    --policy.chunk_size 8 \
    --policy.n_action_steps 8 \
    --dataset.repo_id "fywang/calvin-task-ABCD-D-lerobot" \
    --dataset.root "/data/fywang/Calvin/task_ABCD_D/lerobot_v2_dataset" \
    --use_policy_training_preset false \
    --optimizer.type adamw \
    --optimizer.lr ${LR} \
    --scheduler.type cosine_decay_with_warmup \
    --scheduler.num_warmup_steps 100 \
    --scheduler.num_decay_steps 30000 \
    --scheduler.peak_lr ${LR} \
    --scheduler.decay_lr 2.5e-6 \
    --eval.n_episodes 50 \
    --wandb.enable true \
    --wandb.project "lerobot" \
    --wandb.entity "fywang96" \
