from __future__ import annotations

import os
from typing import Optional

from torch.utils.data import DataLoader

from ..config import DataConfig, TrainConfig
from .collator import DataCollator
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset

__all__ = [
    "DataCollator",
    "IterableDataset",
    "MemMapDataset",
    "build_eval_dataloader",
    "build_memmap_dataset",
    "build_train_dataloader",
]


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def build_memmap_dataset(
    cfg: TrainConfig,
    data_cfg: DataConfig,
    include_instance_metadata: bool = True,
) -> MemMapDataset:
    if data_cfg.paths is None and data_cfg.datasets is None:
        raise ValueError("One of data.paths or data.datasets is required")

    chunk_size = data_cfg.chunk_size or cfg.model.max_sequence_length
    return MemMapDataset(
        paths=data_cfg.paths,
        datasets=data_cfg.datasets,
        chunk_size=chunk_size,
        memmap_dtype=data_cfg.memmap_dtype,
        sample_range=data_cfg.sample_range,
        token_id_remap=data_cfg.token_id_remap,
        include_instance_metadata=include_instance_metadata,
    )


def build_train_dataloader(cfg: TrainConfig, world_size: Optional[int] = None) -> DataLoader:
    resolved_world_size = world_size or _get_world_size()
    if cfg.global_train_batch_size % resolved_world_size != 0:
        raise ValueError("global_train_batch_size must be divisible by world size")

    if cfg.device_train_batch_size is None:
        cfg.device_train_batch_size = cfg.global_train_batch_size // resolved_world_size

    dataset = build_memmap_dataset(cfg, cfg.data, include_instance_metadata=False)
    seed = cfg.data.seed if cfg.data.seed is not None else cfg.seed
    iterable_dataset = IterableDataset(
        dataset,
        cfg.global_train_batch_size,
        seed=seed,
        shuffle=True,
        drop_last=cfg.data.drop_last,
        work_dir=None,
        rank=_get_rank(),
        world_size=resolved_world_size,
        device_batch_size=cfg.device_train_batch_size,
    )
    collator = DataCollator(pad_direction=cfg.data.pad_direction, pad_token_id=cfg.model.pad_token_id)
    return DataLoader(
        iterable_dataset,
        batch_size=cfg.device_train_batch_size,
        drop_last=cfg.data.drop_last,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=None if cfg.data.num_workers == 0 else cfg.data.prefetch_factor,
        persistent_workers=False if cfg.data.num_workers == 0 else cfg.data.persistent_workers,
        timeout=cfg.data.timeout,
    )


def build_eval_dataloader(cfg: TrainConfig, data_cfg: DataConfig, batch_size: int) -> DataLoader:
    dataset = build_memmap_dataset(cfg, data_cfg, include_instance_metadata=True)
    collator = DataCollator(pad_direction=data_cfg.pad_direction, pad_token_id=cfg.model.pad_token_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=data_cfg.drop_last,
        collate_fn=collator,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        prefetch_factor=None if data_cfg.num_workers == 0 else data_cfg.prefetch_factor,
        persistent_workers=False if data_cfg.num_workers == 0 else data_cfg.persistent_workers,
        timeout=data_cfg.timeout,
    )
