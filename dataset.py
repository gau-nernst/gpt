import math

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info


class TokenDataset(IterableDataset):
    def __init__(self, path: str, context_length: int, n_shards: int = 1000, device: str = "cpu") -> None:
        self.data = torch.from_numpy(np.memmap(path, np.int16, "r"))
        self.context_length = context_length
        self.n_shards = n_shards
        self.device = device

    def __iter__(self) -> tuple[Tensor, Tensor]:
        assert get_worker_info() is None, "Do not use num_workers > 0"
        data_size = self.data.shape[0]
        shard_size = math.ceil(data_size / self.n_shards)
        chunk_size = self.context_length + 1

        while True:
            for shard_idx in torch.randperm(self.n_shards):
                shard = self.data[shard_idx * shard_size : min((shard_idx + 1) * shard_size, data_size)]
                shard = shard.to(self.device, torch.long)

                n_samples = shard.shape[0] // chunk_size
                offset = torch.randint(shard.shape[0] - chunk_size * n_samples + 1, ())

                for i in torch.randperm(n_samples):
                    chunk = shard[offset + i * chunk_size : offset + (i + 1) * chunk_size]
                    yield chunk[:-1], chunk[1:]
