import math
import numpy as np
import torch
import os

__all__ = [
    "BigramTopK",
    "topk_to_row_distribution",
]


class BigramTopK:
    """Light-weight bigram language model that stores *top-K* continuations for each token.

    The model is expected to be pre-computed offline and saved as a ``.npz`` file containing two
    arrays with identical first dimensions (``vocab_size``) and second dimension K::

        tokens : np.ndarray[int32]  shape = (V, K)
        probs  : np.ndarray[float32] shape = (V, K)  (rows sum to 1)

    For rare tokens where fewer than *K* next tokens were observed the corresponding rows may be
    padded with the special token id ``-1`` and probability 0.  Those entries are ignored at run-time.

    When queried with a batch of previous tokens ``prev`` the model returns two *torch* tensors of
    shape ``prev.shape + (K,)`` – ``next_tokens`` and ``probs``.
    """

    def __init__(self, path: str, k: int = 16, device: torch.device | str | None = None):
        """Load a saved top-K bigram model.

        Supports two on-disk formats:

        • ``*.npz`` – NumPy zipped arrays with keys ``tokens`` (int32/uint16) and ``probs`` (float32/float16)
        • ``*.pt``  – Torch file produced by ``torch.save({'tokens':T_int,'probs':T_float})``
        """

        suffix = os.path.splitext(path)[1]
        if suffix == ".npz":
            data = np.load(path)
            tk = torch.from_numpy(data["tokens"]).long()
            pr = torch.from_numpy(data["probs"]).float()
        elif suffix in {".pt", ".pth"}:
            obj = torch.load(path, map_location="cpu")
            tk = obj["tokens"].long()
            pr = obj["probs"].float()
        else:
            raise ValueError(f"Unsupported bigram file format: {suffix}")

        self.tokens = tk
        self.probs = pr
        assert self.tokens.shape == self.probs.shape, "tokens and probs must have same shape"
        V, K_loaded = self.tokens.shape
        if k > K_loaded:
            raise ValueError(f"Requested top-k={k} but file only has K={K_loaded}")
        self.k = k
        if k < K_loaded:
            # Truncate to requested k for memory savings
            self.tokens = self.tokens[:, :k]
            self.probs = self.probs[:, :k]
        # Normalise (safety)
        row_sums = self.probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.probs = self.probs / row_sums

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # PyTorch helper methods
    # ------------------------------------------------------------------
    def to(self, device):
        self.tokens = self.tokens.to(device, non_blocking=True)
        self.probs = self.probs.to(device, non_blocking=True)
        return self

    def cuda(self):
        return self.to("cuda")

    # ------------------------------------------------------------------
    # Batch lookup
    # ------------------------------------------------------------------
    @torch.no_grad()
    def batch_lookup(self, prev_tokens: torch.Tensor):
        """Return top-K continuations for *prev_tokens*.

        Args:
            prev_tokens: torch.LongTensor of arbitrary shape containing *previous* tokens.
        Returns:
            next_tokens:  prev_tokens.shape + (K,)
            probs      :  same shape (float32)
        """
        # prev_tokens are indices into first dim of tokens / probs.
        next_tokens = self.tokens.index_select(0, prev_tokens.view(-1)).view(*prev_tokens.shape, self.k)
        probs = self.probs.index_select(0, prev_tokens.view(-1)).view(*prev_tokens.shape, self.k)
        return next_tokens, probs


# ----------------------------------------------------------------------
# Utility: convert token-level distribution to *row* distribution for LightRNN
# ----------------------------------------------------------------------

def topk_to_row_distribution(topk_tokens: torch.Tensor, topk_probs: torch.Tensor, table_size: int) -> torch.Tensor:
    """Convert *top-K* token distribution into LightRNN row distribution.

    Args:
        topk_tokens : LongTensor  shape = (*batch_dims, K)
        topk_probs  : FloatTensor shape = (*batch_dims, K)  (rows sum to 1)
        table_size  : size of LightRNN codebook (number of rows == cols).
    Returns:
        row_dist : FloatTensor shape = (*batch_dims, table_size)  (rows sum to 1)
    """
    assert topk_tokens.shape == topk_probs.shape, "tokens and probs must have same shape"
    *batch_dims, K = topk_tokens.shape
    # Flatten batch dims → (N, K)
    flat_tokens = topk_tokens.view(-1, K)
    flat_probs = topk_probs.view(-1, K)

    N = flat_tokens.shape[0]
    device = flat_tokens.device

    # Row ids of each token
    row_ids = (flat_tokens // table_size).long()  # (N, K)

    # Prepare output and scatter-add probabilities
    row_dist = torch.zeros(N, table_size, device=device, dtype=flat_probs.dtype)
    row_dist.scatter_add_(1, row_ids, flat_probs)

    # Normalise (some rows might have <1 mass because of truncated top-K list)
    row_dist = row_dist / row_dist.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # Reshape back
    row_dist = row_dist.view(*batch_dims, table_size)
    return row_dist 