from __future__ import annotations

from typing import Optional, Tuple

import torch

from .config import TokenIdRemapConfig

__all__ = [
    "get_remapped_period_token_range",
    "zero_period_losses",
]


def get_remapped_period_token_range(
    token_id_remap: Optional[TokenIdRemapConfig],
) -> Optional[Tuple[int, int]]:
    if token_id_remap is None or token_id_remap.replacement_token_count <= 0:
        return None

    start = int(token_id_remap.replacement_token_start)
    stop = start + int(token_id_remap.replacement_token_count)
    return start, stop


def zero_period_losses(
    losses: torch.Tensor,
    labels: torch.Tensor,
    period_token_range: Optional[Tuple[int, int]],
) -> torch.Tensor:
    if period_token_range is None:
        return losses

    start, stop = period_token_range
    period_mask = (labels != -100) & (labels >= start) & (labels < stop)
    if not torch.any(period_mask):
        return losses

    return losses.masked_fill(period_mask, 0.0)
