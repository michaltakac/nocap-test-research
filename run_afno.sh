torchrun --standalone --nproc_per_node=1 train_gpt2_rtx4090_optim_afno.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M_afno \
  --model d12 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --sequence_length 1024 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --num_iterations 4768 \
  --weight_decay 0.1 \
  --learning_rate 0.0018 \
  --warmup_iters 256 \
  --warmdown_iters 1024 \
  --mixer "afno" \
  --afno_num_blocks 8 \
  --afno_sparsity_threshold 0.01 \
  --afno_hard_thresholding_fraction 1.0 \
  --afno_hidden_size_factor 1 \
  #--log_wandb
  # --bias # Optional: uncomment to enable bias in linear layers
