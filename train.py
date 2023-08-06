import contextlib
import math
import time

import torch
from torch.utils.data import DataLoader

from dataset import TokenDataset
from model import GPT


class CosineAnealingLR:
    def __init__(self, optim: torch.optim.Optimizer, lr: float, n_iters: int, n_warmup: int = 0) -> None:
        self.optim = optim
        self.lr = lr
        self.n_iters = n_iters
        self.n_warmup = n_warmup

    def set_lr(self, iter: int) -> float:
        if iter < self.n_warmup:
            lr = 0.1 * self.lr + 0.9 * self.lr * iter / self.n_warmup
        else:
            ratio = min((iter - self.n_warmup) / (self.n_iters - self.n_warmup), 1)
            lr = 0.01 * self.lr + 0.5 * 0.99 * self.lr * (1 + math.cos(ratio * math.pi))

        for grp in self.optim.param_groups:
            grp["lr"] = lr
        return lr


device = "cuda" if torch.cuda.is_available() else "cpu"
fp16 = device == "cuda"
grad_accumulation = 2
batch_size = 64
lr = 3e-3
wd = 1e-1

# Chinchilla: https://arxiv.org/abs/2203.15556
vocab_size = 8000
context_length = 1024
d_model = 512
n_heads = 8
n_layers = 8

n_iters = 100_000
n_warmup = 1000

model = GPT(vocab_size, context_length, d_model, n_heads, n_layers).to(device)
model: GPT = torch.compile(model)
train_ds = TokenDataset("tiny_stories_train.bin", context_length, device=device)
valid_ds = TokenDataset("tiny_stories_valid.bin", context_length, 1, device)

optim = model.configure_optimizer(lr, wd, (0.9, 0.95))
lr_scheduler = CosineAnealingLR(optim, lr, n_iters, n_warmup)
fp16_ctx = torch.autocast(device, torch.float16) if fp16 else contextlib.nullcontext()
scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=fp16)

n_tokens_per_iter = context_length * batch_size * grad_accumulation
print(f"No. of tokens per iteration: {n_tokens_per_iter:,}")

train_batches = iter(DataLoader(train_ds, batch_size))
inputs, targets = next(train_batches)

iter_idx = 0
time0 = time.time()
log_interval = 10
while True:
    _lr = lr_scheduler.set_lr(iter_idx)
    for i in range(grad_accumulation):
        with fp16_ctx:
            loss = model(inputs, targets) / grad_accumulation
        inputs, targets = next(train_batches)
        scaler.scale(loss).backward()

    scaler.step(optim)
    scaler.update()
    optim.zero_grad(True)

    if iter_idx % log_interval == 0:
        time1 = time.time()
        speed = log_interval / (time1 - time0)
        time0 = time1
        print(f"Iter {iter_idx} - lr {_lr:.3e} | {speed:.2f} it/s | loss {loss.item() * grad_accumulation:.3e}")

    iter_idx += 1
    if iter_idx > n_iters:
        break

torch.save(model.state_dict(), "model.pth")
