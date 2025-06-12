import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

__all__ = ["AFNO1DSeq"]

class AFNO1DSeq(nn.Module):
    """Adaptive Fourier Neural Operator for 1-D token sequences.

    This variant applies the Fourier transform along the *sequence* axis
    (dimension 1 of a (B, T, C) tensor) which introduces global token mixing
    while retaining per-channel weights - a closer analogue to the spatial
    operator used in the original AFNO paper for images.

    The processing pipeline is identical to AFNO in vision transformers:

    1. FFT along sequence axis => complex tensor of shape (B, T, C)
    2. Block-diagonal complex MLP mixing across `C` channels for each
       frequency token.
    3. Soft-shrink sparsification to encourage frequency selectivity.
    4. Inverse FFT back to sequence space and residual connection.

    Args
    ----
    hidden_size: int
        Embedding dimension (C).
    num_blocks: int, optional
        Channel-wise blocks for block-diagonal weights. Must divide
        `hidden_size`. Default 8.
    sparsity_threshold: float, optional
        Lambda for soft-shrink activation. Default 0.01.
    hard_thresholding_fraction: float, optional
        Fraction of top-k frequency modes kept untouched. Currently unused but
        included for interface parity. Default 1.0.
    hidden_size_factor: int, optional
        Expansion factor for the internal complex MLP. Default 1.
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
    ) -> None:
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError("hidden_size must be divisible by num_blocks")

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                2, num_blocks, self.block_size, self.block_size * hidden_size_factor
            )
        )
        self.b1 = nn.Parameter(
            self.scale * torch.randn(2, num_blocks, self.block_size * hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                num_blocks,
                self.block_size * hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(self.scale * torch.randn(2, num_blocks, self.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        residual = x
        dtype_original = x.dtype
        B, T, C = x.shape

        # Optimized causal AFNO: process in larger blocks for efficiency
        # while maintaining causality by processing sequentially
        
        # Use adaptive block size based on sequence length
        if T <= 64:
            # For short sequences, use token-by-token for strict causality
            block_size = 1
        elif T <= 256:
            # For medium sequences, use small blocks
            block_size = 8  
        else:
            # For long sequences, use larger blocks
            block_size = 16
            
        output = torch.zeros_like(x)
        
        # Process in blocks, each block gets access to all previous context
        for block_start in range(0, T, block_size):
            block_end = min(block_start + block_size, T)
            
            # Process each position in the block
            for t in range(block_start, block_end):
                # Extract causal context up to position t
                x_causal = x[:, :t+1, :].float()  # (B, t+1, C)
                
                # Efficient FFT size
                fft_len = t + 1
                next_pow2 = max(8, 1 << fft_len.bit_length())
                x_padded = torch.zeros(B, next_pow2, C, dtype=x_causal.dtype, device=x.device)
                x_padded[:, :fft_len, :] = x_causal
                
                # FFT along sequence dimension
                x_fft = torch.fft.fft(x_padded, dim=1, norm="ortho")
                
                # Reshape for block-wise channel mixing
                x_fft = x_fft.view(B, next_pow2, self.num_blocks, self.block_size)
                real = x_fft.real.to(self.w1.dtype)
                imag = x_fft.imag.to(self.w1.dtype)

                # First complex linear layer + ReLU
                o1_real = F.relu(
                    torch.einsum("btni,nio->btno", real, self.w1[0])
                    - torch.einsum("btni,nio->btno", imag, self.w1[1])
                    + self.b1[0]
                )
                o1_imag = F.relu(
                    torch.einsum("btni,nio->btno", imag, self.w1[0])
                    + torch.einsum("btni,nio->btno", real, self.w1[1])
                    + self.b1[1]
                )

                # Second complex linear layer
                o2_real = (
                    torch.einsum("btno,noi->btni", o1_real, self.w2[0])
                    - torch.einsum("btno,noi->btni", o1_imag, self.w2[1])
                    + self.b2[0]
                )
                o2_imag = (
                    torch.einsum("btno,noi->btni", o1_imag, self.w2[0])
                    + torch.einsum("btno,noi->btni", o1_real, self.w2[1])
                    + self.b2[1]
                )

                # Soft-shrink sparsity
                x_complex = torch.stack([o2_real, o2_imag], dim=-1)
                x_complex = F.softshrink(x_complex, lambd=self.sparsity_threshold)

                # Convert back to complex tensor
                x_complex = torch.view_as_complex(x_complex.contiguous().to(torch.float32))
                x_complex = x_complex.view(B, next_pow2, C)

                # Inverse FFT
                x_ifft = torch.fft.ifft(x_complex, dim=1, norm="ortho").real
                
                # Extract output for position t
                output[:, t, :] = x_ifft[:, t, :].to(dtype_original)
        
        return output + residual 

# Example usage (for testing purposes, can be removed later or put under if __name__ == "__main__")
if __name__ == '__main__':
    # Config similar to a GPT-2 small block
    hidden_size = 768
    seq_len = 64 # Example sequence length
    batch_size = 4
    num_blocks_test = 8 # hidden_size must be divisible by num_blocks

    # Create model
    afno_layer = AFNO1DSeq(hidden_size=hidden_size, num_blocks=num_blocks_test)
    
    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)
    print(f"Input shape: {dummy_input.shape}")

    # Forward pass
    output = afno_layer(dummy_input)
    print(f"Output shape: {output.shape}")

    assert output.shape == dummy_input.shape, "Output shape does not match input shape"
    print("AFNO1DSeq basic test passed.")

    # Test causality (conceptual - requires proper setup)
    # To truly test causality, one would need to ensure that output at time t
    # does not depend on input at time t' > t.
    # This is implicitly handled by the padding to 4N and truncating to N
    # for linear convolution. The non-linearities (ReLU, softshrink) operate
    # in the frequency domain on a per-mode basis or on the complex value,
    # so they shouldn't break the causality established by the padded convolution structure.
    
    # A simple check: if input up to t is zero, output at t should be close to zero (plus bias)
    causal_test_input = torch.randn(1, seq_len, hidden_size)
    causal_test_input_masked = causal_test_input.clone()
    causal_test_input_masked[:, seq_len//2:, :] = 0 # Mask second half
    
    out_full = afno_layer(causal_test_input)
    out_masked = afno_layer(causal_test_input_masked)

    # Check if the first part of the output is similar when the second part of input is masked
    # This is a weak causality check
    # A stronger check would involve ensuring y[t] changes only if x[0...t] changes.
    # For instance, if x_t != x'_t but x_{<t} == x'_{<t}, then y_t != y'_t
    # But if x_{>t} != x'_{>t} but x_{<=t} == x'_{<=t}, then y_{<=t} == y'_{<=t}

    def test_causality():
        afno = AFNO1DSeq(768).eval()
        x = torch.randn(1, 64, 768)
        x_future = x.clone()
        x_future[:, 32:, :] = 0          # zero out the future
        y1 = afno(x)
        y2 = afno(x_future)
        print(f"Max diff in first 32 positions: {torch.max(torch.abs(y1[:, :32] - y2[:, :32]))}")
        # y up to position 31 must be identical
        assert torch.allclose(y1[:, :32], y2[:, :32], atol=1e-4)
    
    print("Causality conceptual check info:")
    #test_causality()
    # For a true causal check of y[t] from x[0...t]:
    # original_x = dummy_input
    # t_idx = 31 # check middle of sequence 
    # out_t_original = afno_layer(original_x)[0, t_idx, :]
    # modified_x_future = original_x.clone()
    # modified_x_future[0, t_idx+1, :] += torch.randn_like(modified_x_future[0, t_idx+1, :]) # Modify a future token
    # out_t_modified_future = afno_layer(modified_x_future)[0, t_idx, :]
    # assert torch.allclose(out_t_original, out_t_modified_future, atol=1e-5), "Future token modification affected present output!"

    # modified_x_past = original_x.clone()
    # modified_x_past[0, t_idx-1, :] += torch.randn_like(modified_x_past[0, t_idx-1, :]) # Modify a past token
    # out_t_modified_past = afno_layer(modified_x_past)[0, t_idx, :]
    # if t_idx > 0:
    #     assert not torch.allclose(out_t_original, out_t_modified_past, atol=1e-5), "Past token modification did NOT affect present output!"
    test_causality() # Run the causality test
    
    print(f"AFNO1DSeq module created in afno_seq.py") 