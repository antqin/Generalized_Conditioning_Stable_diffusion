#!/bin/bash

# Set environment variables
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="./finetune-lora-output"
export DATASET_DIR="./lora-train/image"

# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

# Run the training script
accelerate launch finetune_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=${DATASET_DIR} \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --max_train_steps=3000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --checkpointing_steps=100 \
  --validation_epochs=25 \
  --validation_prompt="generate a realistic interior room design" \
  --seed=1337
