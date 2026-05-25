#!/bin/bash

# SFT 8B 학습 스크립트
# 사용법: bash run_sft_training_8b.sh

set -e

cd "$(dirname "$0")"

echo "Starting SFT training (accelerate, 2 GPUs)..."

accelerate launch \
    --config_file config/deepspeed/zero2_2GPU.yaml \
    train_sft.py \
    --config-name=sft_config

echo "SFT training completed!"
echo "Model saved to: outputs/Qwen2.5-7B-SFT/model"
