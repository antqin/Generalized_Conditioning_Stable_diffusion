#!/bin/bash

# Set environment variables
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="./finetune-lora-dreambooth-output"
export DATASET_DIR="./lora-train/image"
export INSTANCE_PROMPT="generate a realistic interior room design"

# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

# Run the training script
accelerate launch train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=${DATASET_DIR} \
  --instance_prompt="${INSTANCE_PROMPT}" \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --max_train_steps=1000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --checkpointing_steps=100 \
  --validation_epochs=15 \
  --validation_prompt="${INSTANCE_PROMPT}" \
  --seed=1337
