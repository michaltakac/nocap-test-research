# Changelog of optimizations focused on RTX4090

### 1. Multi-Token Prediction (MTP) auxiliary loss

This feature aims to improve training efficiency by encouraging the model to predict multiple tokens into the future, not just the immediate next token.

**Plan:**

1.  **Update Command Line Arguments:**
    *   Modify the argument parser to include:
        *   `--mtp_enabled`: A boolean flag (defaults to `False`). If `True`, MTP loss is calculated and added.
        *   `--mtp_max_steps`: An integer (e.g., default `2`, configurable from 2 to 4). This determines the maximum number of future tokens to predict (e.g., if 3, the model predicts 1-step, 2-steps, and 3-steps ahead). The auxiliary loss will apply to predictions from 2-steps up to `mtp_max_steps`.
        *   `--mtp_weight`: A float (default `0.1`). This is the weight for the MTP auxiliary loss when combining it with the primary next-token prediction loss.
        *   `--mtp_rampup_steps`: An integer (default `64`). Number of steps to linearly ramp up MTP weight from 0 to full weight, helping stabilize early training.

2.  **Modify `GPTConfig` Dataclass:**
    *   Add fields:
        *   `mtp_enabled: bool`
        *   `mtp_max_steps: int`
        *   `mtp_weight: float`
        *   `mtp_rampup_steps: int`
    *   In the main script, populate these config fields from the parsed command-line arguments.

3.  **Enhance `DistributedDataLoader`:**
    *   **Constructor (`__init__`)**:
        *   Accept `mtp_enabled` and `mtp_max_steps` arguments. Store an effective `self.actual_max_pred_steps` (which is `mtp_max_steps` if enabled, else 1).
        *   Adjust the assertion for shard token count: `assert shard_ntok >= num_processes * B * T + self.actual_max_pred_steps`. This ensures enough tokens are available in a shard segment for all processes to form batches including the furthest lookahead targets.
    *   **`next_batch()` Method**:
        *   Calculate `buf_len = B * T + self.actual_max_pred_steps`. This is the number of tokens to retrieve from `self.tokens` to form one batch of inputs and all corresponding targets.
        *   Fetch `buf = self.tokens[self.current_position : self.current_position + buf_len]`.
        *   Create the input `x = (buf[:B*T]).view(B, T)`.
        *   Create a dictionary `targets_dict`. For each `k` from 1 to `self.actual_max_pred_steps`:
            *   `targets_dict[f'target_{k}'] = (buf[k : B*T + k]).view(B, T)`.
        *   Advance `current_position` by `B * T * self.num_processes` for the next batch.
        *   Check if current shard has enough data for the next batch:
            ```python
            # Check if the current shard has enough data for the *next* batch for this process
            if self.current_position + B*T + self.actual_max_pred_steps > len(self.tokens):
                self.advance() # Loads new shard and resets current_position
            ```
        *   Return `x.cuda()` and `targets_dict_cuda` (where all tensors in the dictionary are moved to the GPU).

4.  **Modify `GPT.forward()` Method:**
    *   Change the signature to `forward(self, idx, targets_dict=None, return_logits=True, return_loss_components=False)`.
    *   After computing hidden states `x` and `logits = self.lm_head(x)`:
    *   If `targets_dict` is provided:
        *   Calculate `main_loss` using `targets_dict['target_1']` and `logits` (this is the standard next-token prediction loss).
        *   Initialize `total_loss = main_loss`.
        *   If `self.config.mtp_enabled` and `self.config.mtp_max_steps > 1` and `self.config.mtp_weight > 0`:
            *   Initialize `aux_loss_sum = torch.tensor(0.0, device=idx.device)` and `num_aux_losses = 0`.
            *   For `k` from 2 to `self.config.mtp_max_steps`:
                *   Retrieve `aux_targets_k = targets_dict[f'target_{k}']`.
                *   Calculate `aux_loss_k = F.cross_entropy(logits.view(-1, logits.size(-1)), aux_targets_k.view(-1), ignore_index=-1)`. (This uses the same logits from the single forward pass for all k-step ahead predictions, prioritizing speed).
                *   `aux_loss_sum += aux_loss_k`
                *   `num_aux_losses += 1`
            *   If `num_aux_losses > 0`:
                *   `avg_aux_loss = aux_loss_sum / num_aux_losses`
                *   `total_loss = main_loss + self.config.mtp_weight * avg_aux_loss`
        *   If `return_loss_components=True`:
            *   Return `logits, (main_loss, aux_loss)` to enable separate tracking of loss components.
        *   Otherwise return `logits, total_loss`.
    *   If `targets_dict` is `None` (e.g., during inference), the logic for `logits` calculation remains, and `loss` is `None`.

5.  **Update Training Loop in `if __name__ == "__main__":`**:
    *   When instantiating `GPTConfig`, pass the new MTP-related arguments (`args.mtp_enabled`, `args.mtp_max_steps`, `args.mtp_weight`, `args.mtp_rampup_steps`).
    *   Update instantiation of `train_loader` and `val_loader` to pass `args.mtp_enabled` and `args.mtp_max_steps`.
    *   Implement MTP weight ramp-up:
        ```python
        if args.mtp_enabled and step < args.mtp_rampup_steps:
            current_mtp_weight = (step / args.mtp_rampup_steps) * args.mtp_weight
            raw_model.config.mtp_weight = current_mtp_weight
        ```
    *   For the validation loop:
        *   Initialize loss accumulators as tensors:
            ```python
            val_loss = torch.zeros(1, device=device)      # Main metric (next-token only)
            val_ntp_loss = torch.zeros(1, device=device)  # Same as val_loss
            val_mtp_loss = torch.zeros(1, device=device)  # Combined MTP loss
            ```
        *   Get separate loss components:
            ```python
            _, (main_loss, aux_loss) = model(x_val, targets_dict_val, 
                return_logits=False, return_loss_components=True)
            ```
        *   Accumulate detached tensors:
            ```python
            val_ntp_loss += main_loss.detach()
            if aux_loss is not None:
                val_mtp_loss += (main_loss + raw_model.config.mtp_weight * aux_loss).detach()
            val_loss += main_loss.detach()  # Main metric uses only next-token loss
            ```
        *   Perform distributed reduction before converting to Python floats:
            ```python
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_ntp_loss, op=dist.ReduceOp.AVG)
            if args.mtp_enabled:
                dist.all_reduce(val_mtp_loss, op=dist.ReduceOp.AVG)
            
            val_loss = val_loss.item() / val_steps
            val_ntp_loss = val_ntp_loss.item() / val_steps
            val_mtp_loss = val_mtp_loss.item() / val_steps if args.mtp_enabled else val_loss
            ```
        *   Log all metrics to wandb and logfile:
            ```python
            wandb.log({
                "val_loss": val_loss,        # Main metric - next-token only
                "val_ntp_loss": val_ntp_loss,  # Same as val_loss, for clarity
                "val_mtp_loss": val_mtp_loss,  # Combined loss if MTP enabled
                "mtp_weight": raw_model.config.mtp_weight if args.mtp_enabled else 0.0,
                "time": training_time_ms
            }, step=step * tokens_per_iter)
            ```

This plan implements MTP by reusing the final hidden states to predict multiple future tokens, which is computationally efficient. The dataloader is adjusted to provide the necessary future targets, with careful handling of data shard advancement and validation token counting to ensure correct operation in both single-process and distributed settings. The implementation includes weight ramp-up for stability and separate loss component tracking to monitor training progress.

### 2. Memory-Optimized Attention (GQA/MLA)

This feature implements Grouped-Query Attention (similar to Llama2's GQA and DeepSeek's Multi-head Latent Attention) to reduce memory usage and increase model capacity within the fixed 24GB VRAM of RTX 4090.

**Goals:**
- Reduce the activation memory footprint of attention by using fewer key/value heads than query heads
- Use the saved memory to scale up model capacity (wider embeddings) to reach target loss faster
- Maintain throughput while potentially improving training speed due to compute-memory balance

**Implementation Plan:**

1. **Command-line Interface Updates**
   - Added flags to enable and configure GQA:
     - `--gqa_enabled`: Boolean flag to enable GQA 
     - `--n_kv_head`: Explicitly set number of KV heads (if not given, calculated from ratio)
     - `--kv_head_ratio`: Ratio of KV heads to query heads (default 0.25, so 1/4 as many KV heads)
     - `--embd_scale`: Factor to scale up embedding dimension with freed memory (default 1.12, or +12%)

2. **GPTConfig Dataclass Extension**
   - Added two key fields:
     - `gqa_enabled: bool` - Whether to use GQA or standard multi-head attention
     - `n_kv_head: int` - Number of key/value heads (None = same as n_head, i.e., standard attention)

3. **CausalSelfAttention Refactoring**
   - Complete rewrite of the attention module with separate Q and KV projections:
     ```python
     # Separate projections for query and key/value
     self.q_proj = nn.Linear(self.n_embd, self.n_head_q * self.head_dim, bias=False)
     self.kv_proj = nn.Linear(self.n_embd, 2 * self.n_head_kv * self.head_dim, bias=False)
     ```
   - Smart sharing of KV heads across multiple Q heads via `repeat_interleave`:
     ```python
     if self.n_head_q > self.n_head_kv:
         repeats = self.n_head_q // self.n_head_kv
         k = k.repeat_interleave(repeats, dim=2)  # (B, T, n_head_q, head_dim)
         v = v.repeat_interleave(repeats, dim=2)  # (B, T, n_head_q, head_dim)
     ```
   - Validation of divisibility constraints:
     ```python
     assert self.n_embd % self.n_head_q == 0, "n_embd must be divisible by n_head_q"
     assert self.n_head_q % self.n_head_kv == 0, "n_head_q must be divisible by n_head_kv"
     ```

4. **Dynamic KV Head Calculation**
   - Supports ratio-based calculation when specific head count isn't given:
     ```python
     n_kv_head = max(1, int(n_head * args.kv_head_ratio))
     # Ensure n_head is divisible by n_kv_head
     while n_head % n_kv_head != 0:
         n_kv_head -= 1
     ```
   - The divisibility enforcement ensures clean head-sharing patterns

5. **Model Size Scaling**
   - When GQA is enabled, we reinvest the saved memory by scaling up dimensions:
     ```python
     if args.gqa_enabled:
         original_embd = model_config_params["n_embd"]
         model_config_params["n_embd"] = int(original_embd * args.embd_scale)
     ```
   - Default scale of 1.12 = 12% wider embeddings with the same VRAM footprint

6. **Memory Usage Monitoring**
   - Added memory tracking to WandB logs:
     ```python
     "peak_mem_MB": torch.cuda.max_memory_allocated() // (1024 * 1024)
     ```
   - Memory validation report at the end of training:
     ```python
     mem_details = {
         "peak_memory_mb": peak_mem,
         "n_head": model_config.n_head,
         "n_kv_head": model_config.n_kv_head,
         "n_embd": model_config.n_embd,
         "n_layer": model_config.n_layer,
         "kv_sharing_ratio": model_config.n_head / model_config.n_kv_head,
         "embd_scale_factor": args.embd_scale
     }
     ```

7. **Run Script Update**
   - `run.sh` updated with GQA parameters:
     ```
     --gqa_enabled \
     --kv_head_ratio 0.25 \
     --embd_scale 1.12 \
     ```

**Expected Benefits**

1. **Memory Efficiency**: With a KV head ratio of 0.25 (typical), the KV cache size is reduced by ~75%, which accounts for a significant portion of the activation memory during training.

2. **Reinvested Capacity**: The default configuration increases the embedding dimension by 12%, increasing model capacity without requiring more GPU memory.

3. **Throughput Preservation**: While reducing activations, we avoid reduction in computational throughput by maintaining the same number of query heads, which are responsible for most of the interaction with token representations.

4. **Loss Convergence**: Wider embeddings typically result in better representation capacity, which we expect to help reach the target validation loss (3.3821) faster than the baseline.

The GQA implementation is based on research from Llama 2 and DeepSeek, which showed that many attention heads can share key/value projections without significant loss in model quality. Our implementation allows for flexible head-sharing ratios and automatic scaling of model capacity to efficiently utilize the fixed memory budget of a single RTX 4090.

### 3. Mixed Precision Training

This feature implements mixed precision training to leverage the tensor cores on modern NVIDIA GPUs (RTX 3090/4090), potentially doubling computational throughput with minimal impact on training quality.

**Goals:**
- Accelerate training by using tensor cores for matrix operations in lower precision
- Support both BF16 (bfloat16) and FP16 (float16) precision modes
- Keep memory usage efficient while ensuring training stability
- Fix compilation issues with dynamic parameters (e.g., MTP weight ramp-up)

**Implementation Plan:**

1. **Command-line Interface Updates**
   - Added a new precision selection flag:
     ```
     --precision [fp32|fp16|bf16]
     ```
   - Default is `bf16` (bfloat16), which provides the best balance of performance and stability on RTX 4090

2. **GPTConfig Dataclass Extension**
   - Added a field to track selected precision mode:
     ```python
     precision: str = "bf16"  # Options: "fp32", "fp16", "bf16"
     ```

3. **Automatic Precision and Scaler Configuration**
   ```python
   # Decide mixed-precision mode
   if args.precision == "fp16":
       autocast_dtype = torch.float16
       scaler = torch.cuda.amp.GradScaler()  # necessary for FP16 stability
   elif args.precision == "bf16":
       autocast_dtype = torch.bfloat16
       scaler = None  # BF16 doesn't need a scaler
   else:  # fp32
       autocast_dtype = torch.float32
       scaler = None
   ```

4. **Training/Validation Context Management**
   - Updated the autocast context to use the selected precision:
     ```python
     ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, 
                             enabled=autocast_dtype!=torch.float32)
     ```
   - Applied this context to both training and validation sections

5. **Gradient Scaling for FP16**
   - For FP16 precision, implemented gradient scaling to prevent underflow:
     ```python
     # Backward pass with scaling if needed
     if scaler:
         scaler.scale(loss).backward()
     else:
         loss.backward()
         
     # Optimizer step with unscaling if needed
     if scaler:
         scaler.step(optimizer)
         scaler.update()
     else:
         optimizer.step()
     ```

6. **Torch.compile Compatibility Fix**
   - Added a tensor buffer for the MTP weight to prevent recompilation issues:
     ```python
     # In GPT class __init__
     self.register_buffer("mtp_weight_buffer", torch.tensor(config.mtp_weight, dtype=torch.float32))
     
     # In training loop
     raw_model.mtp_weight_buffer.fill_(current_mtp_weight)
     ```
   - This avoids the recompilation warning that occurred when directly modifying `config.mtp_weight`
   - Updated the forward pass to use this buffer instead of the config value, preventing graph recompilations during weight ramp-up

7. **Performance Monitoring**
   - Added precision tracking to logs and W&B:
     ```python
     print0(f"Peak memory consumption: {peak_mem} MiB with precision={args.precision}")
     wandb.log({"peak_memory_mb": peak_mem, "precision": args.precision})
     ```

**Expected Benefits**

1. **Training Speed**: Up to 2× faster matrix multiplications when using tensor cores with FP16/BF16
2. **Memory Efficiency**: Lower precision formats require less memory for activations
3. **Stable Training**: BF16 provides better numeric stability than FP16 while still being faster than FP32
4. **Compatibility with torch.compile**: Fixed recompilation issues with dynamically changing parameters

The mixed precision implementation is based on PyTorch's native AMP (Automatic Mixed Precision) system, providing a seamless way to leverage tensor cores on RTX GPUs. BF16 is set as the default because it offers the best balance between performance and stability for language models on RTX 4090, avoiding underflow issues while still benefiting from tensor core acceleration.

