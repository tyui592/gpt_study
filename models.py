# -*- coding: utf-8 -*-
"""GPT Model Code."""

import torch
import torch.nn as nn
import logging
from typing import Optional


class MHA(nn.Module):
    """Multi-Head Attention Module."""

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 p_drop: float = 0.1):
        """Init."""
        super().__init__()
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """Forward.

        mask: boolean mask to ignore future words in decoder during training.
        """
        #  x: (batch) x (# of tokens) x (d_model)
        batch_size, _, d_model = q.shape
        d_head = d_model // self.n_heads
        scale = d_model / self.n_heads

        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # Current Tensor Dim: (batch) x (# of tokens) x (d_model)
        # (1) split (d_model) to (n_heads) x (head_dim)
        #      -> (batch) x (# of tokens) x (n_heads) x (head_dim)
        # (2) transpose dimensions for multiplication
        #      -> (batch) x (n_heads) x (# of tokens) x (head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, d_head).transpose(1, 2)

        # calculate attention weights,
        #   Dim: (batch) x (n_heads) x (# of tokens) x (# of tokens)
        attention_score = torch.matmul(Q, K.transpose(2, 3)) / scale ** 0.5

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e10)
        attention_weight = torch.softmax(attention_score, dim=-1)

        # y: (batch) x (n_heads) x (# of tokens) x (head_dim)
        y = torch.matmul(self.dropout(attention_weight), V)

        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, -1, d_model)
        y = self.w_o(y)
        return y, attention_weight


class FeedForward(nn.Module):
    """Feed forward module."""

    def __init__(self,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 p_drop: float = 0.1):
        """init."""
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        """Forward."""
        x = self.dropout(self.activation(self.w1(x)))
        x = self.w2(x)
        return x


class DecoderLayer(nn.Module):
    """Decoder Layer."""

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 attention_drop: float = 0.0,
                 residual_drop: float = 0.1):
        """Init.

        d_model: dimension of model.
        d_ff: dimension of feedforward layers.
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)  # for mha
        self.layer_norm2 = nn.LayerNorm(d_model)  # for feed forward

        self.mha = MHA(d_model, n_heads, attention_drop)
        self.ff = FeedForward(d_model, d_ff, residual_drop)
        self.dropout = nn.Dropout(residual_drop)  # residual dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Forward with q, k, v and mask."""
        _x = self.layer_norm1(x)
        _x, attention = self.mha(_x, _x, _x, mask)
        x = x + self.dropout(_x)

        _x = self.layer_norm2(x)
        _x = self.ff(_x)
        x = x + self.dropout(_x)

        return x, attention


class Decoder(nn.Module):
    """Decoder."""

    def __init__(self,
                 n_layers: int = 6,
                 vocab_size: int = 100,
                 d_model: int = 512,
                 n_heads: int = 6,
                 d_ff: int = 2048,
                 attention_drop: float = 0.0,
                 residual_drop: float = 0.1,
                 embedding_drop: float = 0.1,
                 max_len: int = 100):
        """Init.

        max_len: Maximum number of token.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                         n_heads=n_heads,
                         d_ff=d_ff,
                         attention_drop=attention_drop,
                         residual_drop=residual_drop) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(embedding_drop)  # embedding dropout

        self.layer_norm = nn.LayerNorm(d_model)
        self.w_out = nn.Linear(d_model, vocab_size)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """Forward."""
        batch_size, num_token = x.shape[:2]

        # make a input tensor for positional embedding
        pos = torch.arange(0, num_token).unsqueeze(0)
        pos = pos.to(x.device)
        pos_embedding = self.pos_embedding(pos)
        x = self.dropout(self.token_embedding(x) + pos_embedding)

        for layer in self.layers:
            x, attention = layer(x, mask)

        x = self.layer_norm(x)
        x = self.w_out(x)

        return x, attention


class GPT(nn.Module):
    """GPT."""

    def __init__(self,
                 vocab_size: int = 100,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 attention_drop: float = 0.0,
                 residual_drop: float = 0.1,
                 embedding_drop: float = 0.1,
                 max_len: int = 100):
        """Init."""
        super().__init__()
        self.decoder = Decoder(vocab_size=vocab_size,
                               d_model=d_model,
                               n_layers=n_layers,
                               n_heads=n_heads,
                               d_ff=d_ff,
                               max_len=max_len,
                               attention_drop=attention_drop,
                               residual_drop=residual_drop,
                               embedding_drop=embedding_drop)

    def make_mask(self, x: torch.Tensor):
        """Make a mask for a decoder to don't cheat next tokens."""
        num_token = x.shape[1]

        # masking for subsequent tokens
        mask = torch.tril(torch.ones((num_token, num_token))).bool().to(x.device)

        return mask

    def forward(self, x: torch.Tensor):
        """Forward."""
        mask = self.make_mask(x)

        y, attention = self.decoder(x, mask)
        return y, attention

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


def count_parameters(model):
    """Count number of model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(args):
    """Get a model."""
    model = GPT(vocab_size=args.vocab_size,
                d_model=args.d_model,
                d_ff=args.d_ff,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                attention_drop=args.attention_drop,
                residual_drop=args.residual_drop,
                embedding_drop=args.embedding_drop,
                max_len=args.max_len)

    model.apply(model._init_weights)
    num_param = count_parameters(model)
    logging.info(f"Number of parameters: {num_param:,}")

    return model
