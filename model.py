import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GPTAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 3, bias)
        self.out_proj = nn.Linear(d_model, d_model, bias)
        self.n_heads = n_heads
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.in_proj(x)
        q, k, v = qkv.unflatten(-1, (3, self.n_heads, -1)).transpose(-2, -4).unbind(-3)
        out = F.scaled_dot_product_attention(q, k, v, None, self.dropout if self.training else 0, True)
        return self.out_proj(out.transpose(-2, -3).flatten(-2))


class GPTBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, bias: bool = False, dropout: float = 0.0
    ) -> None:
        super().__init__()
        mlp_dim = int(d_model * mlp_ratio)
        self.mha = nn.Sequential(
            nn.LayerNorm(d_model),
            GPTAttention(d_model, n_heads, bias, dropout),
            nn.Dropout(dropout),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_dim, bias),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model, bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mha(x)
        x = x + self.mlp(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, context_length: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, context_length, d_model))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.weight[:, : x.shape[1]]


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,  # should pad up to nearest multiple of 64 for efficiency
        context_length: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEmbedding(context_length, d_model)
        self.blocks = nn.Sequential(*[GPTBlock(d_model, n_heads, mlp_ratio, bias, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if name.endswith(".weight"):
                nn.init.normal_(param, 0, 0.02)
            elif name.endswith(".bias"):
                nn.init.zeros_(param)
            else:
                print(f"Unrecognized param {name} with shape {param.shape}")

    def forward(self, x: Tensor, targets: Tensor | None = None) -> Tensor:
        """
        Args:
            x: shape (B, L)
            targets: shape (B, L)
        """
        for module in (self.embedding, self.pe, self.blocks, self.norm):
            x = module(x)  # (B, L, D)
        logits = x @ self.embedding.weight.T  # (B, L, vocab_size)

        if targets is not None:
            return F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits

    def configure_optimizer(self, lr: float, wd: float, betas: tuple[float, float]) -> torch.optim.Optimizer:
        # From nanoGPT/llama2.c: Only 2D params will be weight decayed.
        # i.e. matmuls + embeddings are decayed, biases and layernorms are not.
        params = [p for p in self.parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        no_decay_params = [p for p in params if p.dim() < 2]

        def num_params(params: list[Tensor]) -> int:
            return sum(p.numel() for p in params)

        print(f"No. of trainable params: {num_params(params):,}")
        print(f"  - w/ weight decay: {num_params(decay_params):,}")
        print(f"  - w/o weight decay: {num_params(no_decay_params):,}")

        optim_groups = [
            dict(params=decay_params, weight_decay=wd),
            dict(params=no_decay_params, weight_decay=0),
        ]
        optim = torch.optim.AdamW(optim_groups, lr, betas)
        return optim
