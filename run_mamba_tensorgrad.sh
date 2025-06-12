#!/bin/bash
# Mamba GPT-2 training with TensorGRaD optimizer
# Experimental: Testing TensorGRaD's sparse+low-rank gradient decomposition with Mamba

torchrun --standalone --nproc_per_node=1 train_gpt2_mamba.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M_mamba_tensorgrad \
  --model d12 \
  --batch_size 8 \
  --grad_accumulation_steps 64 \
  --sequence_length 2048 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --num_iterations 3000 \
  --weight_decay 0.2 \
  --learning_rate 1.5e-4 \
  --warmup_iters 256 \
  --warmdown_iters 1024 \
  --target_val_loss 3.3821 \
  --mixer "mamba" \
  --mamba_d_state 16 \
  --mamba_d_conv 4 \
  --mamba_expand 2 \
  --optimizer "tensorgrad" \
  --tg_rank 4 \
  --tg_sparsity 0.02 \
  --tg_lambda 1.0 \
  --tg_update_freq 4 \
  #--log_wandb