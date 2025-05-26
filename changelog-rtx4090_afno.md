# Changelog of optimizations focused on RTX4090 using operator learning (AFNO)

## Initial AFNO Integration (Phase 1)

**Goal:** Integrate Adaptive Fourier Neural Operator (AFNO) as an alternative token mixer to the baseline attention mechanism, aiming for improved speed and memory efficiency, especially at longer sequence lengths.

**Plan & Implementation Details:**

1.  **`afno.py` Module Creation:**
    *   Created `afno.py` containing the `AFNO1D` class.
    *   Adapted from the NVIDIA AFNO1D reference, focusing on 1D sequences.
    *   **Key Features:**
        *   Learnable block-diagonal weights in the Fourier domain.
        *   Complex MLP operations (`w1, b1, w2, b2`) for mixing modes.
        *   `torch.fft.rfft` and `torch.fft.irfft` for transformations.
        *   Hard thresholding (`hard_thresholding_fraction`) to keep a subset of Fourier modes.
        *   Soft-thresholding (`sparsity_threshold` via `F.softshrink`) for frequency sparsity.
        *   Residual connection.
    *   **Causality (Approach A):** Implemented causality by padding the input sequence to `2*N` before `rfft` and truncating the output of `irfft` back to `N`. This allows the global convolution via FFT to behave causally in an autoregressive setting.

2.  **GPT Model Architecture Modifications (`train_gpt2_rtx4090_optim_afno.py`):**
    *   **`AFNO1D` Import:** Added `from afno import AFNO1D`.
    *   **`GPTConfig` Update:**
        *   Added `mixer: str` (defaulting to "attn", choices: "attn", "afno").
        *   Added AFNO-specific parameters: `afno_num_blocks`, `afno_sparsity_threshold`, `afno_hard_thresholding_fraction`, `afno_hidden_size_factor`.
        *   Added `bias: bool` (default `False`) for `nn.Linear` layers, controlled by CLI.
        *   Added `sequence_length: int` to `GPTConfig` to be used by positional embeddings when AFNO is active.
    *   **`Block` Class Modification:**
        *   Conditionally instantiates `self.mixer` as either `AFNO1D(...)` or `CausalSelfAttention(config)` based on `config.mixer`.
        *   Renamed `self.attn_scale` to `self.mixer_scale` for generality.
    *   **Positional Encoding for AFNO:**
        *   Since RoPE is part of `CausalSelfAttention`, learned absolute positional embeddings (`wpe`) are introduced for AFNO.
        *   `self.transformer` `nn.ModuleDict` in `GPT.__init__` conditionally includes `wpe = nn.Embedding(config.sequence_length, config.n_embd)` if `config.mixer == "afno"`.
        *   In `GPT.forward`, `pos_emb` from `wpe` is added to `tok_emb` if `config.mixer == "afno"`.

3.  **Training Script CLI and Logic (`train_gpt2_rtx4090_optim_afno.py` - `if __name__ == "__main__"`):**
    *   **Added CLI Arguments:**
        *   `--mixer` (choices: "attn", "afno")
        *   `--afno_num_blocks`
        *   `--afno_sparsity_threshold`
        *   `--afno_hard_thresholding_fraction`
        *   `--afno_hidden_size_factor`
        *   `--bias` (action="store_true")
    *   **Model Instantiation:** `GPTConfig` is now populated using these new CLI arguments, including `args.sequence_length` for `config.sequence_length`.

4.  **Run Script (`run_afno.sh`):**
    *   Created `run_afno.sh` to launch training with AFNO.
    *   Sets `--mixer "afno"` and includes default values for other AFNO parameters.
    *   Initial configuration for `d12` model: `batch_size=16`, `grad_accumulation_steps=32`, `sequence_length=1024` to align with baseline for initial comparison.

**Next Steps (Phases 2 & 3):**
- Initial training runs and benchmarking against baseline attention.
- Rigorous causality verification.
- Sequence length and batch size scaling experiments.

## First Training Runs & Debugging (Ongoing)

**Goal:** Verify the integrated AFNO model trains, identify and fix initial bugs, and get baseline performance metrics.

**Implementation & Observations:**

1.  **Runtime Error with `torch.view_as_complex`:**
    *   **Issue:** Initial training attempts failed with `TorchRuntimeError: Dynamo failed to run FX node ... RuntimeError('Tensor must have a last dimension with stride 1')` when calling `torch.view_as_complex(x_complex_shrunk)` in `afno.py`.
    *   **Cause:** The tensor `x_complex_shrunk` (output of `torch.stack` followed by `F.softshrink`) was not memory-contiguous in the way expected by `view_as_complex`.
    *   **Fix:** Applied `.contiguous()` to `x_complex_shrunk` before the `view_as_complex` call: `torch.view_as_complex(x_complex_shrunk.contiguous())`.

2.  **Successful Initial Training:**
    *   After the fix, the model started training successfully with the AFNO mixer (`./run_afno.sh`).
    *   Initial validation loss and training losses are being logged, with training loss showing a decreasing trend.
    *   `output_dir` confirmed as `pylog124M_afno`.

3.  **Performance & Compiler Warnings:**
    *   **TF32 Precision:** `torch.compile` issued a warning: `UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting torch.set_float32_matmul_precision('high') for better performance.`
        *   **Action:** Added `torch.set_float32_matmul_precision('high')` to `train_gpt2_rtx4090_optim_afno.py`.
    *   **Complex Operators Warning:** `torch.compile` also warned: `UserWarning: Torchinductor does not support code generation for complex operators. Performance may be worse than eager.` This is noted and will be monitored.

4.  **Initial Performance Analysis (Up to ~500 steps):**
    *   **Training Stability:** Model trains stably with AFNO mixer.
    *   **Validation Loss:** Achieved a validation loss of **1.765293** at step 512. This is significantly better than the target of ≤ 3.3821 and very promising for learning capability.
    *   **Training Loss:** Shows a consistent decreasing trend, reaching ~0.05 by step 500+ (from `s:N trn:X` log lines).
    *   **Step Time:** Average step time after `torch.compile` warmup is ~8.3 seconds.
        *   Baseline attention model average step time: ~4.1 seconds.
        *   AFNO is currently ~2x slower per step.
    *   **Tokens/Second:**
        *   AFNO: ~63k tokens/sec.
        *   Baseline Attention: ~128k tokens/sec.
    *   **Compiler Warning Impact:** The `UserWarning: Torchinductor does not support code generation for complex operators. Performance may be worse than eager.` is likely contributing to the slower step time for AFNO.

**Next Steps:**
- Continue monitoring the current training run to observe long-term stability and loss progression.
- Investigate the ~2x slower step time. Potential actions:
    - Temporarily disable `torch.compile` to compare with eager mode speed.
    - Profile the AFNO model to identify specific bottlenecks within `AFNO1D` when compiled.
- If wall-clock time to target loss becomes an issue despite fewer steps, explore advanced positional encodings or further AFNO optimizations.
- Update changelog with further findings.

