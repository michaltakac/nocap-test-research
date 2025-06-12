#!/bin/bash
# Mamba GPT-2 training with OPTIMIZED learning rate schedule
# Targets early convergence to validation loss < 3.3821

torchrun --standalone --nproc_per_node=1 train_gpt2_mamba.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M_mamba_adamw_optimized \
  --model d12 \
  --batch_size 8 \
  --grad_accumulation_steps 64 \
  --sequence_length 2048 \
  --val_loss_every 64 \
  --val_batch_size 16 \
  --num_iterations 1200 \
  --weight_decay 0.2 \
  --learning_rate 1.5e-4 \
  --warmup_iters 256 \
  --warmdown_iters 400 \
  --target_val_loss 3.3821 \
  --mixer "mamba" \
  --mamba_d_state 16 \
  --mamba_d_conv 4 \
  --mamba_expand 2 \
  --optimizer "adamw" \
  #--log_wandb