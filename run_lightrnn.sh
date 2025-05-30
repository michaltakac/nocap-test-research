torchrun --standalone --nproc_per_node=1 train_lightrnn.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M_lightrnn \
  --model d24 \
  --batch_size 32 \
  --grad_accumulation_steps 1 \
  --sequence_length 512 \
  --val_loss_every 128 \
  --val_batch_size 32 \
  --num_iterations 5000 \
  --weight_decay 0.1 \
  --learning_rate 0.0018 \
  --warmup_iters 64 \
  --warmdown_iters 256 \
  --target_val_loss 3.3821 \
  --tie_embedding light \
  #--log_wandb
