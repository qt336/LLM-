from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F


class DataCollator:
    def __init__(self, pad_direction: str = "right", pad_token_id: int = 0):
        self.pad_direction = str(pad_direction)
        self.pad_token_id = pad_token_id

    def _pad_1d_tensor(self, key: str, tensor: torch.Tensor, max_len: int) -> torch.Tensor:
        pad_width = max_len - tensor.shape[0]
        if pad_width <= 0:
            return tensor

        pad_value = self.pad_token_id if key == "input_ids" else 0
        pad = (pad_width, 0) if self.pad_direction == "left" else (0, pad_width)
        return F.pad(tensor, pad, value=pad_value)

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        for key in items[0]:
            values = [item[key] for item in items]
            first = values[0]

            if isinstance(first, torch.Tensor):
                if first.ndim == 1:
                    max_len = max(value.shape[0] for value in values)
                    values = [self._pad_1d_tensor(key, value, max_len) for value in values]
                batch[key] = torch.stack(values)
            elif isinstance(first, bool):
                batch[key] = torch.tensor(values, dtype=torch.bool)
            elif isinstance(first, int):
                batch[key] = torch.tensor(values, dtype=torch.long)
            else:
                batch[key] = values
        return batch
