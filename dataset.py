import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info


class TokenDataset(IterableDataset):
    def __init__(self, path: str, context_length: int) -> None:
        self.data = torch.from_numpy(np.memmap(path, np.int16, "r"))
        self.context_length = context_length

    def __iter__(self) -> tuple[Tensor, Tensor]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        print(f"Worker {worker_id}: seed={torch.initial_seed()}")

        while True:
            total_size = self.data.shape[0]
            chunk_size = self.context_length + 1
            n_samples = total_size // chunk_size
            offset = torch.randint(total_size - chunk_size * n_samples + 1, ())

            for i in torch.randperm(n_samples):
                chunk = self.data[offset + i * chunk_size : offset + (i + 1) * chunk_size]
                yield chunk[:-1], chunk[1:]
