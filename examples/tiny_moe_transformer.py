# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn


class Experts(nn.Module):
    def __init__(self, d_model, hidden_dim, num_experts):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, d_model))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        nn.init.normal_(self.w1, std=0.05)
        nn.init.normal_(self.w2, std=0.05)

    def forward(self, x):
        hidden = F.gelu(torch.einsum("btd,ehd->bteh", x, self.w1))
        return torch.einsum("bteh,edh->bted", hidden, self.w2)


class MoEBlock(nn.Module):
    def __init__(self, d_model, n_heads, num_experts):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = Experts(d_model, 2 * d_model, num_experts)

    def forward(self, x, src_mask=None, is_causal=False):
        normalized = self.norm1(x)
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            attn_mask=src_mask,
            is_causal=is_causal,
            need_weights=False,
        )
        x = x + attended
        normalized = self.norm2(x)
        scores = self.router(normalized).softmax(dim=-1)
        # Evaluate all experts locally after their parameters are gathered.
        mixed = torch.einsum("bte,bted->btd", scores, self.experts(normalized))
        return x + mixed


class TinyMoETransformer(nn.Module):
    def __init__(
        self,
        d_model=16,
        n_heads=4,
        n_layers=2,
        num_experts=4,
        vocab_size=32,
        seq_len=8,
    ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList(
            [MoEBlock(d_model, n_heads, num_experts) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.tok_embeddings(tokens) + self.pos_embeddings(positions)
        causal_mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=tokens.device
        ).triu(1)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal_mask, is_causal=True)
        return self.output(self.norm(hidden))
