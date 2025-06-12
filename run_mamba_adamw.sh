#!/bin/bash
# Mamba GPT-2 training with fused AdamW optimizer
# Fast training to reach validation loss < 3.3821

torchrun --standalone --nproc_per_node=1 train_gpt2_mamba.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M_mamba_adamw \
  --model d12 \
  --sequence_length 1024 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --optimizer lion \
  --beta1 0.9 \
  --beta2 0.99 \
  --lr_schedule cosine \
  --learning_rate 3e-4 \
  --min_lr_ratio 0.04 \
  --warmup_iters 128 \
  --warmdown_iters 800 \
  --weight_decay 0.1 \
  --dropout 0.15 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --num_iterations 1400 \
  --target_val_loss 3.3821 \
  --mixer mamba \
  --mamba_d_state 16 \
  --mamba_d_conv 4 \
  --mamba_expand 2 \
  --log_wandb