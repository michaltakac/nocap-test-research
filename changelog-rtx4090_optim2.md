# Changelog of optimizations focused on RTX4090, version 2

### 1. Adaptive Softmax

**Goal:**
- Replace the standard dense softmax layer with an adaptive softmax layer (`torch.nn.AdaptiveLogSoftmaxWithLoss`).
- This aims to speed up the forward and backward passes for the output layer, especially for large vocabularies, by using smaller matrix multiplications for less frequent words (tail clusters).
- Expected speedup: 1.3x - 1.8x overall step time, potentially reducing total training time to reach target validation loss (≤ 3.3821) from 5.4 hours to ~3.5 hours on an RTX 4090.

**Plan & Implementation Details:**

1.  **Command Line Interface (CLI) Additions:**
    *   `--adaptive_softmax` (boolean flag): Enables the use of adaptive softmax.
    *   `--asoft_cutoffs` (string, e.g., "2000,10000"): Comma-separated list of integers defining the vocabulary cutoffs for different clusters. The vocabulary size is implicitly the final cutoff.
    *   `--asoft_div_value` (float, default `4.0`): Divisor for determining the projection size of tail clusters.

2.  **Model (`GPT`) Modifications:**
    *   **`__init__` Method:**
        *   Accepts `use_asoft`, `cutoffs`, and `div_value` arguments.
        *   If `use_asoft` is `True`, `self.lm_head` is replaced with `self.asoft = nn.AdaptiveLogSoftmaxWithLoss(...)`.
        *   Parameters for `AdaptiveLogSoftmaxWithLoss`:
            *   `in_features`: `config.n_embd`
            *   `n_classes`: `config.vocab_size` (dynamically determined from data)
            *   `cutoffs`: Parsed list from `args.asoft_cutoffs`.
            *   `div_value`: `args.asoft_div_value`.
            *   `head_bias=False`: To maintain consistency with the original `lm_head` (which had `bias=False`).
        *   Weight tying between token embeddings (`wte`) and `lm_head` is disabled when adaptive softmax is active, as `AdaptiveLogSoftmaxWithLoss` manages its own internal projection matrices.
    *   **`forward` Method:**
        *   If `self.use_asoft` is `True`:
            *   The input `x` (output of transformer blocks and RMSNorm) is reshaped to `(B*T, C)` for the `AdaptiveLogSoftmaxWithLoss` layer during training/validation.
            *   The `AdaptiveLogSoftmaxWithLoss` layer directly computes the loss.
            *   If `return_logits` is `True`, `self.asoft.log_prob(x)` is called to get log probabilities.
            *   For inference (`targets is None`), `self.asoft.log_prob(x[:, [-1], :])` is used for the last token prediction.

3.  **Main Script Logic (`if __name__ == "__main__":`)**
    *   **Argument Parsing:** New CLI arguments are parsed.
    *   **Vocabulary Size:** `effective_vocab_size` is dynamically determined by `train_loader.tokens.max().item() + 1` to ensure the `AdaptiveLogSoftmaxWithLoss` layer is initialized with the correct number of classes from the actual dataset.
    *   **Cutoff Parsing & Validation:** The `asoft_cutoffs` string is parsed into a list of integers. These cutoffs are validated to be sorted, unique, and strictly less than `effective_vocab_size`. If no valid cutoffs remain (e.g., if all provided cutoffs are too large), adaptive softmax is disabled, and the model falls back to the standard softmax.
    *   **Model Instantiation:** The `GPT` model is initialized with the adaptive softmax configuration.
    *   **Mixed Precision Handling (BF16/FP16 Compatibility Fix):**
        *   The `RuntimeError: index_copy_(): self and source expected to have the same dtype, but got (self) BFloat16 and (source) Float` was encountered when using `bf16` precision with `AdaptiveLogSoftmaxWithLoss`.
        *   **Fix Implemented:**
            1.  After the main model is cast to the target `dtype` (e.g., `bfloat16`), if adaptive softmax is enabled and the target `dtype` is not `float32`, the `model.asoft` submodule is explicitly cast to `torch.float32` (`model.asoft.to(torch.float32)`).
            2.  In the `GPT.forward` method, when `self.asoft` is used:
                *   The input tensor `x` is explicitly cast to `torch.float32` before being passed to `self.asoft`.
                *   The output `loss` (and `logits`, if computed) from `self.asoft` are cast back to the original `dtype` of `x` (e.g., `bfloat16`).
        *   This isolates `AdaptiveLogSoftmaxWithLoss` to operate in `float32`, resolving internal dtype conflicts while allowing the rest of the model to leverage mixed precision.

4.  **Loss Calculation:**
    *   The loss is directly obtained from `AdaptiveLogSoftmaxWithLoss` output object (`output.loss`). No changes were needed in the main training loop's loss accumulation logic, as the `.loss` attribute already provides the per-token NLL.

5.  **`torch.compile` Compatibility:**
    *   The `AdaptiveLogSoftmaxWithLoss` module is a standard PyTorch module and is generally compatible with `torch.compile`. The dtype casting strategy ensures that `torch.compile` can handle the mixed-precision setup effectively.

**Patch Outline (Conceptual - Key Snippets):**

```python
# In train_gpt2_rtx4090_optim2.py

from torch.nn import AdaptiveLogSoftmaxWithLoss

# --- In GPT class __init__ ---
if use_asoft:
    self.asoft = AdaptiveLogSoftmaxWithLoss(
        in_features=config.n_embd,
        n_classes=config.vocab_size,
        cutoffs=cutoffs,
        div_value=div_value,
        head_bias=False
    )
else:
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    # ... weight tying ...

# --- In GPT class forward ---
if self.use_asoft:
    original_x_dtype = x.dtype
    x_for_asoft = x.to(torch.float32)
    if targets is not None:
        output_asoft = self.asoft(x_for_asoft.view(-1, x_for_asoft.size(-1)), targets.view(-1))
        loss = output_asoft.loss.to(original_x_dtype)
        # ... handle logits if needed, casting back ...
    else: # Inference
        logits = self.asoft.log_prob(x_for_asoft[:, [-1], :]).to(original_x_dtype)
        loss = None
# ... else: standard softmax path ...

# --- In __main__ ---
# Parse args: --adaptive_softmax, --asoft_cutoffs, --asoft_div_value
# Determine effective_vocab_size
# Parse and validate asoft_cutoffs_parsed

model = GPT(model_config, use_asoft=use_asoft, cutoffs=asoft_cutoffs_parsed, div_value=args.asoft_div_value)
model.to(device)
if dtype != torch.float32:
    model.to(dtype=dtype)
    if use_asoft and hasattr(model, 'asoft'):
        model.asoft.to(torch.float32) # Cast asoft to float32
# ... rest of the setup ...
```

**Run Script (`run_optim_2.sh`) Update:**
- Added flags:
  ```bash
  --adaptive_softmax \
  --asoft_cutoffs "2000,10000" \
  --asoft_div_value 4.0 \
  ```

**Expected Benefits:**
-   **Faster Training:** Significant reduction in computation for the softmax layer, leading to faster overall training iterations.
-   **Reduced Memory Footprint:** Smaller weight matrices for tail vocabulary clusters can reduce the model's memory usage for parameters, though activations related to the head cluster still dominate.
-   **Maintained Accuracy:** Adaptive softmax is designed to approximate the full softmax with minimal loss in predictive accuracy.
-   The target metric (cross-entropy on the full validation set) remains comparable as `AdaptiveLogSoftmaxWithLoss` computes an exact NLL over the entire vocabulary.

**Potential Risks & Mitigations:**
-   **Slight Accuracy Deviation:** Removing weight tying (a side effect of using `AdaptiveLogSoftmaxWithLoss` which has its own parameters) might slightly alter model behavior. This is generally minor for models of this scale.
-   **Hyperparameter Sensitivity:** The choice of `cutoffs` can impact performance and accuracy. The defaults (`2000,10000`) are common starting points but might need tuning for optimal results on FineWeb.
-   **Numerical Stability with Mixed Precision:** Addressed by casting `AdaptiveLogSoftmaxWithLoss` and its inputs/outputs to/from `float32` when the main model uses `bf16` or `fp16`.

This implementation introduces adaptive softmax to potentially accelerate training towards the benchmark goal by optimizing the computationally intensive output layer.

