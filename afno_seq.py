import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

__all__ = ["AFNO1DSeq"]

class AFNO1DSeq(nn.Module):
    """Adaptive Fourier Neural Operator for 1-D token sequences.

    This variant applies the Fourier transform along the *sequence* axis
    (dimension 1 of a (B, T, C) tensor) which introduces global token mixing
    while retaining per-channel weights – a closer analogue to the spatial
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

        # FFT along the *sequence* dimension
        x_fft = torch.fft.fft(x.float(), dim=1, norm="ortho")  # complex64

        # Reshape for block-wise channel mixing
        x_fft = x_fft.view(B, T, self.num_blocks, self.block_size)
        real = x_fft.real.to(self.w1.dtype)
        imag = x_fft.imag.to(self.w1.dtype)

        # First complex linear layer + ReLU in frequency domain
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

        # Second complex linear layer: btno x noi -> btni
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

        # Convert back to complex tensor and reshape to (B, T, C)
        x_complex = torch.view_as_complex(x_complex.contiguous().to(torch.float32))
        x_complex = x_complex.view(B, T, C)

        # Inverse FFT back to sequence space
        x_ifft = torch.fft.ifft(x_complex, dim=1, norm="ortho").real
        return x_ifft.to(dtype_original) + residual 