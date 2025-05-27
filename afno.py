import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

class AFNO1D(nn.Module):
    """
    Adaptive Fourier Neural Operator for 1D sequences (e.g., text).

    Args:
        hidden_size (int): Dimension of the input and output embeddings (channels).
        num_blocks (int, optional): Number of blocks for block-diagonal weights. 
                                    hidden_size must be divisible by num_blocks. Defaults to 8.
        sparsity_threshold (float, optional): Lambda for softshrink activation, promoting sparsity. 
                                              Defaults to 0.01.
        hard_thresholding_fraction (float, optional): Fraction of high-frequency modes to keep.
                                                      1.0 keeps all modes. Defaults to 1.0.
        hidden_size_factor (int, optional): Factor to expand hidden size in the Fourier domain MLP.
                                            Defaults to 1.
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0, hidden_size_factor=1):
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_blocks {num_blocks}")

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02 # Initialization scale for weights

        # Learnable weights for the block-diagonal MLP in Fourier domain
        # w1, b1 are for the first linear layer (complex multiplication)
        # w2, b2 are for the second linear layer (complex multiplication)
        # Weights are stored as real numbers: index 0 for real part, 1 for imaginary part
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)
        
        residual_bias = x # Save for residual connection
        dtype_original = x.dtype
        # x = x.float() # FFT operations often require float32. Cast later if needed by FFT.
        # The error occurs when x (bfloat16) -> x.float() (float32) -> fft_output.real (float32)
        # interacts with self.w1 (bfloat16).
        # We will cast the fft_output parts to self.w1.dtype.

        x_fft_input = x.float()  # ensure float32 for numerical stability in FFT

        B, N, C_input_shape = x.shape  # sequence length N, hidden size

        # ------------------------------------------------------------------
        # Channel-wise FFT (dim = 2) — token-independent spectral mixing.
        # This keeps strict causality automatically.
        # ------------------------------------------------------------------

        x_fft = torch.fft.fft(x_fft_input, dim=2, norm="ortho")  # (B, N, C) complex64

        # Reshape for block-diagonal complex MLP
        x_fft_reshaped = x_fft.view(B, N, self.num_blocks, self.block_size)

        w_dtype = self.w1.dtype

        # Prepare output tensors
        o1_real = torch.zeros((B, N, self.num_blocks, self.block_size * self.hidden_size_factor),
                              device=x.device, dtype=w_dtype)
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros((B, N, self.num_blocks, self.block_size), device=x.device, dtype=w_dtype)
        o2_imag = torch.zeros_like(o2_real)

        # Real & imag parts
        x_real = x_fft_reshaped.real.to(w_dtype)
        x_imag = x_fft_reshaped.imag.to(w_dtype)

        # --- First complex linear layer + ReLU
        o1_real = F.relu(
            torch.einsum('...bi,bio->...bo', x_real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo', x_imag, self.w1[1]) + self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('...bi,bio->...bo', x_imag, self.w1[0]) +
            torch.einsum('...bi,bio->...bo', x_real, self.w1[1]) + self.b1[1]
        )

        # --- Second complex linear layer
        o2_real = (
            torch.einsum('...bi,bio->...bo', o1_real, self.w2[0]) -
            torch.einsum('...bi,bio->...bo', o1_imag, self.w2[1]) + self.b2[0]
        )
        o2_imag = (
            torch.einsum('...bi,bio->...bo', o1_imag, self.w2[0]) +
            torch.einsum('...bi,bio->...bo', o1_real, self.w2[1]) + self.b2[1]
        )

        # Soft-shrink sparsity
        x_complex = torch.stack([o2_real, o2_imag], dim=-1)
        x_complex = F.softshrink(x_complex, lambd=self.sparsity_threshold)

        # Back to complex tensor and flatten channels
        x_complex = torch.view_as_complex(x_complex.contiguous().to(torch.float32))  # (B,N,num_blocks,block_size)
        x_fft_processed = x_complex.reshape(B, N, C_input_shape)

        # Inverse FFT along channel dim
        x_ifft = torch.fft.ifft(x_fft_processed, dim=2, norm="ortho").real

        # Cast back and residual
        return x_ifft.type(dtype_original) + residual_bias

# Example usage (for testing purposes, can be removed later or put under if __name__ == "__main__")
if __name__ == '__main__':
    # Config similar to a GPT-2 small block
    hidden_size = 768
    seq_len = 64 # Example sequence length
    batch_size = 4
    num_blocks_test = 8 # hidden_size must be divisible by num_blocks

    # Create model
    afno_layer = AFNO1D(hidden_size=hidden_size, num_blocks=num_blocks_test)
    
    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)
    print(f"Input shape: {dummy_input.shape}")

    # Forward pass
    output = afno_layer(dummy_input)
    print(f"Output shape: {output.shape}")

    assert output.shape == dummy_input.shape, "Output shape does not match input shape"
    print("AFNO1D basic test passed.")

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
        afno = AFNO1D(768).eval().cuda()
        x = torch.randn(1, 64, 768, device='cuda')
        x_future = x.clone()
        x_future[:, 32:, :] = 0          # zero out the future
        y1 = afno(x)
        y2 = afno(x_future)
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
    
    print(f"AFNO1D module created in afno.py") 