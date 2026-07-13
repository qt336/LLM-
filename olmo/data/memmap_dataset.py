from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import SampleRangeConfig, TokenIdRemapConfig


@dataclass
class _ShardSpec:
    label: str
    path: str
    sample_count: int
    token_offset: int
    sample_start: int


class MemMapDataset(Dataset):
    def __init__(
        self,
        paths: Optional[List[str]],
        datasets: Optional[Dict[str, List[str]]],
        chunk_size: int,
        memmap_dtype: str,
        sample_range: Optional[SampleRangeConfig],
        token_id_remap: Optional[TokenIdRemapConfig],
        include_instance_metadata: bool,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.chunk_size = chunk_size
        self.dtype = np.dtype(memmap_dtype)
        self.sample_range = sample_range
        self.token_id_remap = token_id_remap
        self.include_instance_metadata = include_instance_metadata
        self._memmaps: Dict[str, np.memmap] = {}
        self._shards: List[_ShardSpec] = []
        self._prefix_sample_counts: List[int] = []

        shard_entries: List[Tuple[str, str]] = []
        if paths is not None:
            shard_entries.extend(("", path) for path in paths)
        if datasets is not None:
            for label, label_paths in datasets.items():
                shard_entries.extend((label, path) for path in label_paths)
        if not shard_entries:
            raise ValueError("No data paths were provided")

        total_available_samples = 0
        for _, path in shard_entries:
            token_count = os.path.getsize(path) // self.dtype.itemsize
            total_available_samples += token_count // self.chunk_size

        range_start = 0 if self.sample_range is None else self.sample_range.start
        range_stop = total_available_samples if self.sample_range is None or self.sample_range.stop is None else self.sample_range.stop
        if range_start < 0 or range_stop < range_start or range_stop > total_available_samples:
            raise ValueError(
                f"Invalid sample_range [{range_start}, {range_stop}) for dataset with {total_available_samples} samples"
            )

        total_samples = 0
        total_tokens = 0
        samples_consumed = 0
        for label, path in shard_entries:
            token_count = os.path.getsize(path) // self.dtype.itemsize
            sample_count = token_count // self.chunk_size
            if sample_count == 0:
                continue

            shard_start = max(range_start - samples_consumed, 0)
            shard_stop = min(range_stop - samples_consumed, sample_count)
            kept_sample_count = max(shard_stop - shard_start, 0)
            if kept_sample_count > 0:
                self._shards.append(
                    _ShardSpec(
                        label=label,
                        path=path,
                        sample_count=kept_sample_count,
                        token_offset=(samples_consumed + shard_start) * self.chunk_size,
                        sample_start=shard_start,
                    )
                )
                total_samples += kept_sample_count
                self._prefix_sample_counts.append(total_samples)

            total_tokens += token_count
            samples_consumed += sample_count

        if not self._shards:
            raise ValueError("No training samples could be constructed from the provided memmap shards")

    def __len__(self) -> int:
        return self._prefix_sample_counts[-1]

    def _get_memmap(self, path: str) -> np.memmap:
        if path not in self._memmaps:
            self._memmaps[path] = np.memmap(path, mode="r", dtype=self.dtype)
        return self._memmaps[path]

    def _locate_shard(self, index: int) -> Tuple[_ShardSpec, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self._prefix_sample_counts, index)
        previous = 0 if shard_idx == 0 else self._prefix_sample_counts[shard_idx - 1]
        shard = self._shards[shard_idx]
        return shard, index - previous

    def _apply_token_id_remap(self, token_ids: np.ndarray, shard: _ShardSpec, sample_index: int) -> np.ndarray:
        if self.token_id_remap is None:
            return token_ids

        mask = token_ids == self.token_id_remap.source_token_id
        if not np.any(mask):
            return token_ids

        absolute_positions = shard.token_offset + sample_index * self.chunk_size + np.nonzero(mask)[0]
        absolute_positions = absolute_positions.astype(np.uint64, copy=False)
        seed = np.uint64(self.token_id_remap.seed)
        with np.errstate(over="ignore"):
            mixed = absolute_positions * np.uint64(6364136223846793005) + seed * np.uint64(1442695040888963407)
        replacement = self.token_id_remap.replacement_token_start + (
            mixed % np.uint64(self.token_id_remap.replacement_token_count)
        ).astype(token_ids.dtype, copy=False)

        remapped = token_ids.copy()
        remapped[mask] = replacement
        return remapped

    def __getitem__(self, index: int):
        shard, sample_index = self._locate_shard(index)
        local_sample_index = shard.sample_start + sample_index
        start = local_sample_index * self.chunk_size
        end = start + self.chunk_size
        token_ids = np.asarray(self._get_memmap(shard.path)[start:end], dtype=np.int64)
        token_ids = self._apply_token_id_remap(token_ids, shard, sample_index)

        item = {
            "input_ids": torch.from_numpy(token_ids.astype(np.int64, copy=False)),
            "index": index,
        }
        if self.include_instance_metadata:
            label = shard.label or Path(shard.path).stem
            item["metadata"] = {"label": label, "path": shard.path}
        return item
