"""Transformer model."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TransformerLM(nn.Module):
    """Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embed_size: int,
        head_size: int,
        num_blocks: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)  # VxE
        self.position_embedding = nn.Embedding(context_size, embed_size)  # TxE
        self.feedforward = LinearRELULayerNorm(embed_size, dropout)

        bargs_ = (embed_size, embed_size // head_size, context_size)
        self.blocks = nn.Sequential(*[Block(*bargs_) for _ in range(num_blocks)])

        self.ln = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Make a prediction of the next token given the current."""
        B, T = x.shape

        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(torch.arange(T, device=x.device))

        z = self.blocks(token_embedding + position_embedding)
        z = self.feedforward(z)
        logits = self.lm_head(z)

        if y is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        loss = F.cross_entropy(logits, y.view(B * T))
        return logits, loss

    def generate(self, x: Tensor, tokens: int = 1) -> Tensor:
        """Take an input (B, T) and sample a new token."""
        block_size = self.position_embedding.num_embeddings
        for _ in range(tokens):
            xc = x[:, -block_size:]
            logits, _ = self(xc)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            newx = torch.multinomial(probs, 1)
            x = torch.cat((x, newx), -1)
        return x


class Head(nn.Module):
    def __init__(
        self,
        head_size: int,
        embed_size: int,
        context_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("trl", torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        k: Tensor = self.key(x)
        q: Tensor = self.query(x)
        v: Tensor = self.value(x)

        _, T, C = x.shape
        w = q @ k.transpose(-2, -1) * C ** (-1 / 2)
        w = w.masked_fill(torch.as_tensor(self.trl)[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        embed_size: int,
        batch_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hargs_ = (head_size, embed_size, batch_size, dropout)
        self.heads = nn.ModuleList([Head(*hargs_) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class LinearRELULayerNorm(nn.Module):
    """Feedforward section of a Transformer w/residual connections."""

    def __init__(self, n: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 4 * n),
            nn.ReLU(),
            nn.Linear(4 * n, n),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        context_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        aargs_ = (num_heads, embed_size // num_heads, embed_size, context_size, dropout)
        self.sa = MultiHeadAttention(*aargs_)

        self.ff = LinearRELULayerNorm(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
