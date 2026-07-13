from __future__ import annotations

import os
from typing import Optional

import numpy as np
from torch.utils.data import IterableDataset as TorchIterableDataset


class IterableDataset(TorchIterableDataset):
    def __init__(
        self,
        dataset,
        global_batch_size: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = False,
        work_dir=None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        device_batch_size: Optional[int] = None,
    ):
        super().__init__()
        del work_dir

        self.dataset = dataset
        self.total_size = len(dataset)
        self.global_batch_size = global_batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        self.world_size = int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else world_size
        if global_batch_size % self.world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size")
        self.device_batch_size = device_batch_size or (global_batch_size // self.world_size)
        self.epoch = 0
        self.start_index = 0
        self._order = np.arange(self.total_size, dtype=np.int64)
        self._effective_size = self.total_size
        self.reshuffle(0)

    def reshuffle(self, epoch: int) -> None:
        self.epoch = epoch
        if self.shuffle:
            rng = np.random.default_rng(self.seed + epoch)
            self._order = rng.permutation(self.total_size)
        else:
            self._order = np.arange(self.total_size, dtype=np.int64)

        if self.drop_last:
            self._effective_size = self.total_size - (self.total_size % self.global_batch_size)
        else:
            self._effective_size = self.total_size

    def __iter__(self):
        if self.start_index >= self._effective_size:
            return

        for global_example_index in range(self.start_index, self._effective_size):
            global_micro_batch = global_example_index // self.device_batch_size
            if global_micro_batch % self.world_size != self.rank:
                continue
            yield self.dataset[int(self._order[global_example_index])]
