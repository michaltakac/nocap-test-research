import torch
import torch.nn.functional as F
from typing import Dict, Tuple

__all__ = [
    "compute_mtp_loss",
]


def compute_mtp_loss(
    logits: torch.Tensor,
    targets_dict: Dict[str, torch.Tensor],
    mtp_weight: torch.Tensor,
    ignore_index: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined loss for Multi-Token Prediction.

    Args:
        logits: (B, T, V) tensor – predictions produced from **one** forward pass.
        targets_dict: mapping {"target_k": Tensor(B,T)} for k≥1.  Must contain
            at least "target_1" (standard next-token targets).
        mtp_weight: 0-dim tensor broadcastable – weight applied to *mean* of
            auxiliary losses (k≥2).
        ignore_index: label id that will be ignored in CE (typically -1).

    Returns:
        total_loss   : main + mtp_weight * aux_mean
        main_loss    : CE for k=1 predictions only
        aux_mean_loss: mean CE over k≥2 (or None if no aux targets)
    """
    # Flatten logits to (B*T, V) once for efficiency
    B, T, V = logits.shape
    logits_flat = logits.view(-1, V).to(torch.float32)

    main_target = targets_dict["target_1"].view(-1)
    main_loss = F.cross_entropy(logits_flat, main_target, ignore_index=ignore_index)

    # Gather auxiliary losses if any additional targets provided
    aux_losses = []
    for key in sorted(targets_dict.keys()):
        if key == "target_1":
            continue
        aux_tgt = targets_dict[key].view(-1)
        loss_k = F.cross_entropy(logits_flat, aux_tgt, ignore_index=ignore_index)
        aux_losses.append(loss_k)

    if aux_losses:
        aux_mean = torch.stack(aux_losses).mean()
        total_loss = main_loss + mtp_weight * aux_mean
    else:
        aux_mean = None
        total_loss = main_loss

    return total_loss, main_loss, aux_mean 