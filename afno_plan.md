# AFNO Implementation Plan: Operator-Learning Token Mixing for GPT-2

Training infrastructure, where code will be running: we're using single NVIDIA GeForce RTX 4090, 8 vCPUs, 62 GB RAM, PyTorch nightly 2.8.0 and CUDA Toolkit 12.8.1, on cloud server used for testing the optimizations (Runpod.io)

## Objective

Train a language model on the **FineWeb** dataset to reach validation loss ≤ **3.3821** as fast as possible using **1 RTX 4090 GPU** by replacing quadratic self-attention with **Adaptive Fourier Neural Operator (AFNO)** token mixing.

### Key Innovation
Replace O(N²) self-attention with O(N log N) Fourier-domain global convolution, enabling:
- Linear memory scaling with sequence length
- Quasi-linear computational complexity
- Ability to handle much longer sequences (4096+ tokens vs baseline 1024)
- Potential for better generalization per parameter

---

## Background & Motivation

Based on the ICLR 2022 paper ["Efficient Token Mixing for Transformers via Adaptive Fourier Neural Operators"](https://openreview.net/forum?id=EXHG-A3jlM) and the [official implementation](https://github.com/NVlabs/AFNO-transformer), AFNO achieves:

- **Memory efficiency**: Linear in sequence length (vs quadratic for attention)
- **Speed**: O(N log N) complexity via FFT operations
- **Scalability**: Handles sequences up to 65k tokens where self-attention fails
- **Global context**: Maintains full sequence receptive field like attention

### Key Technical Insights
1. **Operator Learning**: Frame token mixing as continuous global convolution in function space
2. **Block-diagonal structure**: Efficient channel mixing weights
3. **Adaptive weight sharing**: Across token positions  
4. **Frequency sparsification**: Via soft-thresholding and shrinkage
5. **Resolution agnostic**: Same weights work across different sequence lengths

---

## Implementation Strategy

### Phase 1: Core AFNO Integration

#### 1.1 Create AFNO Module (`afno.py`)
- Port `AFNO1D` from [NVlabs reference implementation](https://github.com/NVlabs/AFNO-transformer)
- Strip 2D/image-specific components, focus on 1D sequence processing
- Key components:
  ```python
  class AFNO1D(nn.Module):
      def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01):
          # Block-diagonal mixing weights
          # Soft thresholding parameters
          # FFT operations for global convolution
  ```

#### 1.2 Modify GPT Architecture
- Add mixer type selection to `GPTConfig`:
  ```python
  @dataclass
  class GPTConfig:
      mixer: str = "attn"  # "attn" | "afno"
      afno_num_blocks: int = 8
      afno_sparsity_threshold: float = 0.01
      afno_hard_thresholding_fraction: float = 1.0
  ```

- Update `Block` class to use configurable mixer:
  ```python
  class Block(nn.Module):
      def __init__(self, config):
          if config.mixer == "afno":
              self.mixer = AFNO1D(config)
          else:
              self.mixer = CausalSelfAttention(config)
  ```

#### 1.3 Handle Causality
**Critical**: Ensure autoregressive property is preserved
- **Approach A** (Simple): Zero-pad future positions, truncate after inverse FFT
- **Approach B** (Elegant): Apply causal mask in frequency domain via complex phase

### Phase 2: Training Configuration

#### 2.1 Positional Encoding Strategy
Since RoPE is attention-specific, use alternatives:
- **Start with**: Learned absolute positional embeddings (simple, fast to test)
- **Upgrade to**: Sinusoidal + Fourier gating (from AFNO paper Appendix B.3)

#### 2.2 Sequence Length Scaling
- **Baseline**: 1024 tokens (limited by quadratic attention)
- **Target**: 2048-4096 tokens (leverage AFNO's linear memory)
- **Ultimate**: 8192+ tokens if memory permits

#### 2.3 Batch Size Optimization
With attention overhead removed:
- Increase `--batch_size` to saturate GPU utilization (~18-20GB usage)
- Target 10x+ tokens/step vs baseline
- Monitor SM utilization and memory bandwidth

### Phase 3: Validation & Testing

#### 3.1 Causality Verification
```python
def test_causality(model, seq_len=100):
    # Create sequence with only first k tokens non-zero
    x = torch.zeros(1, seq_len, dtype=torch.long)
    x[0, :k] = torch.randint(1, 1000, (k,))
    
    # Verify outputs depend only on past tokens
    for pos in range(k, seq_len):
        assert output[0, pos] == output_with_future_masked[0, pos]
```

#### 3.2 Tiny Shakespeare Sanity Check
- 1-layer AFNO-GPT on 100k tokens for 100 steps
- Verify loss convergence similar to attention baseline
- Confirm gradient flow through FFT operations

#### 3.3 Speed/Memory Benchmarking
Track key metrics:
- `tokens/sec` throughput
- `peak_memory_usage` 
- `wall_clock_time_to_target_loss`
- `memory_scaling` with sequence length

### Phase 4: FineWeb Optimization

#### 4.1 Hyperparameter Tuning
Starting from baseline, adjust:
- **Learning rate**: May need adjustment due to different optimization landscape
- **Warmup steps**: Reduce to 64-128 (expect smoother optimization)
- **Weight decay**: Keep baseline 0.1 initially
- **AFNO-specific**:
  - `num_blocks`: 4-16 range
  - `sparsity_threshold`: 0.01-0.1 range
  - Enable soft-thresholding regularization

#### 4.2 Context Length Experiments
- **1024 tokens**: Baseline comparison
- **2048 tokens**: First scaling test  
- **4096 tokens**: Target for production run
- Monitor perplexity vs. wall-clock trade-offs

#### 4.3 Production Training Run
```bash
# Updated run_afno.sh
torchrun --standalone --nproc_per_node=1 train_gpt2_afno.py \
  --mixer afno \
  --sequence_length 4096 \
  --batch_size 8 \
  --grad_accumulation_steps 64 \
  --afno_num_blocks 8 \
  --afno_sparsity_threshold 0.01 \
  # ... other params
```

### Phase 5: Advanced Optimizations

#### 5.1 Hybrid Architecture
- **Early layers**: Local attention or convolution for fine-grained patterns
- **Middle/Late layers**: AFNO for long-range dependencies
- Inspired by hierarchical processing in vision transformers

#### 5.2 Curriculum Learning
- Start training with shorter sequences (512 tokens) for speed
- Gradually increase to 4096 tokens
- Leverage AFNO's resolution-agnostic property

#### 5.3 Frequency Domain MoE
- Separate block-diagonal weights per expert
- Route based on frequency characteristics or positional buckets
- Potential for massive parameter scaling

---

## Technical Implementation Details

### AFNO Core Algorithm
1. **Input**: Token embeddings `x ∈ R^(B×T×C)`
2. **FFT**: Convert to frequency domain `X = FFT(x)`
3. **Mixing**: Apply learnable block-diagonal weights `Y = W ⊙ X`
4. **Sparsification**: Soft threshold low-magnitude frequencies
5. **IFFT**: Convert back to spatial domain `y = IFFT(Y)`
6. **Residual**: Add to input with scaling

### Key Components to Port from NVlabs
- `ComplexReLU` activation for frequency domain
- Block-diagonal weight structure for efficiency
- Soft thresholding for sparsity
- Proper initialization schemes

### Causality Implementation Options

#### Option A: Padding + Truncation
```python
def forward(self, x):
    B, T, C = x.shape
    # Pad with zeros for future positions
    x_padded = F.pad(x, (0, 0, 0, T, 0, 0))
    # Apply AFNO
    y_padded = self.afno_core(x_padded)
    # Truncate to original length
    return y_padded[:, :T, :]
```

#### Option B: Frequency Domain Masking
```python
def apply_causal_mask_freq(X):
    # Apply phase shift equivalent to causal convolution
    # More complex but potentially more efficient
    pass
```

---

## Expected Performance Gains

### Memory Scaling
- **Attention**: O(N²) memory for N-length sequences
- **AFNO**: O(N) memory → 16x reduction at 4096 tokens

### Computational Complexity  
- **Attention**: O(N²·C) FLOPs
- **AFNO**: O(N·log(N)·C) FLOPs → ~10x speedup at 4096 tokens

### Sequence Length Capability
- **Baseline**: 1024 tokens (memory limited)
- **AFNO Target**: 4096-8192 tokens on same hardware

### Wall-Clock Speedup Targets
- **Conservative**: 2-3x faster training to same validation loss
- **Optimistic**: 5-10x faster with longer context benefits
- **Stretch Goal**: Beat baseline time even with 4x longer sequences

---

## Risk Mitigation

### Technical Risks
1. **Causality bugs**: Extensive unit testing, gradual rollout
2. **FFT numerical stability**: Use stable implementations, monitor gradients
3. **Memory regression**: Careful profiling, fallback options
4. **Convergence issues**: Start with proven hyperparameters

### Fallback Plans
1. **Hybrid attention-AFNO**: Use attention for early layers if issues arise
2. **Reduced sequence length**: Fall back to 1024-2048 if 4096 too aggressive  
3. **Simpler baselines**: Test on Tiny Shakespeare before FineWeb

### Validation Strategy
1. **Unit tests**: Causality, gradient flow, memory scaling
2. **Ablation studies**: AFNO vs attention at same sequence length
3. **Scaling curves**: Performance vs sequence length, batch size
4. **Checkpointing**: Save intermediate results for analysis

---

## Success Metrics

### Primary Goal
- **Validation loss ≤ 3.3821** in **< 5.4 hours** (beat baseline)

### Secondary Metrics
- **Tokens/second throughput**: Target 2-10x improvement
- **Memory efficiency**: Handle 4x longer sequences in same memory
- **Scaling**: Sub-quadratic scaling with sequence length

### Research Insights
- **Generalization per parameter**: Compare model quality at same parameter count
- **Long-range dependencies**: Evaluate on tasks requiring distant context
- **Transfer learning**: Test AFNO weights across different domains

---

## Timeline & Milestones

| Day | Milestone | Deliverable |
|-----|-----------|-------------|
| 1-2 | Core AFNO implementation | `afno.py`, updated `Block` class |
| 3-4 | Causality validation | Unit tests passing, Shakespeare sanity check |
| 5-6 | Speed benchmarking | Comparison tables, memory profiles |
| 7-9 | FineWeb scaling | 1024→4096 token experiments |
| 10+ | Production run | Sub-5.4hr training to target loss |

---

## Code Structure

```
/
├── afno.py                    # AFNO1D implementation
├── train_gpt2_afno.py        # Modified training script
├── run_afno.sh              # Training configuration
├── test_causality.py        # Validation tests
└── experiments/
    ├── ablation_studies.py   # Attention vs AFNO comparisons
    ├── scaling_analysis.py   # Sequence length scaling
    └── benchmark_results.md  # Performance documentation
```

---

## References

1. **Paper**: ["Efficient Token Mixing for Transformers via Adaptive Fourier Neural Operators"](https://arxiv.org/abs/2111.13587) (ICLR 2022)
2. **Code**: [NVlabs/AFNO-transformer](https://github.com/NVlabs/AFNO-transformer) 
3. **OpenReview**: [Conference discussion](https://openreview.net/forum?id=EXHG-A3jlM)
4. **Baseline**: Current GPT-2 implementation in this repository

---

## Notes

- Focus on **algorithmic innovation** over hyperparameter tuning
- Prioritize **generalizability** - solutions should work across different setups
- Maintain **scientific rigor** - proper ablations and controls
- Document **lessons learned** for future scaling efforts

The key insight is that language modeling may benefit from the same operator-learning principles that have revolutionized scientific computing. By treating text as a continuous function and token mixing as global convolution, AFNO could unlock new scaling regimes for transformer architectures. 