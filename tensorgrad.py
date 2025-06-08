import math
from typing import Iterable, Optional, Tuple, Dict, Any

import torch
from torch.optim import Optimizer
from torch import Tensor

__all__ = ["TensorGRaD"]


def _topk_mask(x: Tensor, k: int) -> Tensor:
    """Return a boolean mask with the *k* largest absolute values set to True."""
    if k <= 0 or k >= x.numel():
        # keep everything (dense) or nothing
        return torch.ones_like(x, dtype=torch.bool)
    # torch.topk works on 1-D input; flatten for convenience
    flat = x.view(-1)
    # Retrieve indices of k largest magnitudes (no gradient tracking)
    _, idx = torch.topk(flat.abs(), k, sorted=False)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    return mask.view_as(x)


class TensorGRaD(Optimizer):
    r"""Implementation of the *TensorGRaD* optimiser (Robust tensor-gradient
    decomposition) inspired by the paper:

        "TensorGRaD: Tensor Gradient Robust Decomposition for Memory-Efficient
        Neural Operator Training" (Zheng et al., 2024).

    The method decomposes every dense gradient **g** into a *sparse* component
    consisting of the largest-magnitude entries and a *low-rank* approximation
    of the residual.  The update is then formed from the sum of the two parts
    and fed through an AdamW-style adaptive rule.

    This implementation focuses on practicality for large language models.  It
    therefore applies the expensive decomposition *only* to tensors whose total
    number of elements is below ``max_elements``;  larger tensors fall back to
    the standard AdamW step.  This still preserves the original high-dimensional
    structure for most projection matrices and convolution kernels without
    incurring a prohibitive cost for gigantic embedding tables.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        # TensorGRaD-specific hyper-parameters
        rank: int = 4,
        sparsity: float = 0.01,
        lambda_sparse: float = 1.0,
        update_freq: int = 1,
        max_elements: int = 2_000_000,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0,1)")
        if not rank >= 0:
            raise ValueError("rank must be non-negative")
        if not update_freq >= 1:
            raise ValueError("update_freq must be ≥ 1")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            sparsity=sparsity,
            lambda_sparse=lambda_sparse,
            update_freq=update_freq,
            max_elements=max_elements,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            rank: int = group["rank"]
            sparsity: float = group["sparsity"]
            lambda_sparse: float = group["lambda_sparse"]
            update_freq: int = group["update_freq"]
            max_elements: int = group["max_elements"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad: Tensor = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("TensorGRaD does not support sparse gradients yet")

                state: Dict[str, Any] = self.state[p]

                # State initialisation --------------------------------------
                if len(state) == 0:
                    state["step"] = 0
                    # First and second moment buffers (Adam-style)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Cache for sparse indices (Ω) reused across *update_freq* steps
                    state["sparse_mask"] = None

                state["step"] += 1
                step = state["step"]

                # ----------------------------------------------------------
                # 1.  Gradient decomposition (Sparse + Low-rank)
                # ----------------------------------------------------------
                g_processed: Tensor
                if grad.numel() <= max_elements and grad.dim() >= 2:
                    # --- Sparse component ---------------------------------
                    k = int(sparsity * grad.numel())
                    if k > 0:
                        # Re-compute sparse mask every *update_freq* steps
                        if state["sparse_mask"] is None or (step % update_freq == 0):
                            state["sparse_mask"] = _topk_mask(grad, k)
                        sparse_mask: Tensor = state["sparse_mask"]
                        g_sparse = torch.where(sparse_mask, grad, torch.zeros_like(grad))
                    else:
                        g_sparse = torch.zeros_like(grad)

                    # --- Low-rank component -------------------------------
                    g_residual = grad - g_sparse
                    # Flatten to matrix for truncated SVD
                    if rank > 0 and g_residual.numel() > 0:
                        try:
                            matrix = g_residual.view(g_residual.shape[0], -1)
                            if hasattr(torch, "svd_lowrank"):
                                # Fast path where available (PyTorch >=2.1 w/ CUDA)
                                u, s, v = torch.svd_lowrank(matrix, q=rank, niter=1)
                            else:
                                # Fallback to full SVD then truncate
                                u_full, s_full, v_full = torch.linalg.svd(matrix, full_matrices=False)
                                u, s, v = u_full[:, :rank], s_full[:rank], v_full[:rank, :]
                            s_diag = torch.diag_embed(s) if s.dim() == 1 else s
                            lowrank_mat = (u[:, :rank] @ torch.diag(s[:rank])) @ v[:rank, :]
                            g_lowrank = lowrank_mat.view_as(grad)
                        except RuntimeError:
                            # Fallback: skip low-rank if SVD fails (e.g., on CPU)
                            g_lowrank = g_residual
                    else:
                        g_lowrank = g_residual

                    # Combine, applying scaling λ to sparse part
                    g_processed = lambda_sparse * g_sparse + g_lowrank
                else:
                    # Fallback to original gradient (too large or 1-D param)
                    g_processed = grad

                # ----------------------------------------------------------
                # 2.  Weight decay (decoupled) -----------------------------
                if weight_decay != 0.0:
                    p.data.mul_(1 - lr * weight_decay)

                # ----------------------------------------------------------
                # 3.  AdamW update with decomposed gradient ---------------
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(g_processed, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_processed, g_processed, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
