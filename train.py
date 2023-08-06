import contextlib
import math
import time

import torch
from torch.utils.data import DataLoader, IterableDataset

from dataset import TokenDataset
from model import GPT


device = "cuda" if torch.cuda.is_available() else "cpu"
fp16 = device == "cuda"
grad_accumulation = 4
batch_size = 16
lr = 1e-3
wd = 1e-5

# Chinchilla: https://arxiv.org/abs/2203.15556
vocab_size = 8000
context_length = 1024
d_model = 512
n_heads = 8
n_layers = 8

model = GPT(vocab_size, context_length, d_model, n_heads, n_layers)
model = torch.compile(model)
train_ds = TokenDataset("tiny_stories_train.bin", context_length)
valid_ds = TokenDataset("tiny_stories_valid.bin", context_length)

optim = model.configure_optimizer(lr, wd, (0.9, 0.95))
fp16_ctx = torch.autocast(device, torch.float16) if fp16 else contextlib.nullcontext()
scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=fp16)

n_tokens_per_iter = context_length * batch_size * grad_accumulation
print(f"No. of tokens per iteration: {n_tokens_per_iter}")

n_iters = 100_000
n_warmup = 1000


def get_lr(it: int) -> float:
    if it < n_warmup:
        return 0.1 * lr + 0.9 * lr * it / n_warmup
    return 0.01 * lr + 0.5 * 0.99 * lr * (1 + math.cos((it - n_warmup) / (n_iters - n_warmup) * math.pi))


def iter_batches(ds: IterableDataset):
    for inputs, targets in DataLoader(ds, batch_size, pin_memory=True):
        inputs = inputs.to(device=device, dtype=torch.int, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
        yield inputs, targets


train_batches = iter_batches(train_ds)
inputs, targets = next(train_batches)

step_idx = 0
time0 = time.time()
log_interval = 1
while True:
    _lr = get_lr(step_idx)
    for param_group in optim.param_groups:
        param_group["lr"] = _lr

    for i in range(grad_accumulation):
        with fp16_ctx:
            loss = model(inputs, targets) / grad_accumulation
        inputs, targets = next(train_batches)
        scaler.scale(loss).backward()

    scaler.step(optim)
    scaler.update()
    optim.zero_grad(True)

    if step_idx % log_interval == 0:
        time1 = time.time()
        speed = log_interval / (time1 - time0)
        time0 = time1
        print(f"Iter {step_idx}: lr {_lr:.3e} | {speed:.2f} it/s")

    step_idx += 1
    if step_idx > n_iters:
        break
