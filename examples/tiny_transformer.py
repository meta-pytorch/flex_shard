# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=32, seq_len=8, d_model=16, n_heads=4, n_layers=2):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    dim_feedforward=2 * d_model,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
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
