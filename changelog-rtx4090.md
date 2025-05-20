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

