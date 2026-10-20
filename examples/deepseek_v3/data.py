# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training batches: seeded random tokens, or C4 packed into fixed-length rows.

Each batch is a dict of ``[batch_size, seq_len]`` int64 tensors: ``input``,
``labels`` (the next tokens) and ``positions``, which count from 0 within each
document. Both sources save and restore their position for checkpoints.
"""

import copy
import json
import os

import torch


class SyntheticBatches:
    """Uniformly random tokens, one document per row. Each rank and step gets
    its own seeded generator, so a run is reproducible and resumable."""

    def __init__(self, *, vocab_size, batch_size, seq_len, rank, seed=0):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.seed = seed
        self.step = 0

    def __iter__(self):
        return self

    def __next__(self):
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_000 * self.step + self.rank
        )
        self.step += 1
        tokens = torch.randint(
            self.vocab_size, (self.batch_size, self.seq_len + 1), generator=generator
        )
        positions = torch.arange(self.seq_len).expand(self.batch_size, -1)
        return {
            "input": tokens[:, :-1],
            "labels": tokens[:, 1:],
            "positions": positions,
        }

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]


def _tokenizer_file(tokenizer, filename):
    """``filename`` from a local directory, or from a Hugging Face Hub repo."""
    if os.path.isdir(tokenizer):
        return os.path.join(tokenizer, filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(tokenizer, filename)


class C4Batches:
    """English C4, streamed from the Hugging Face Hub and split across ranks.

    Each document becomes ``[BOS] text [EOS]``, predicting each next token
    within the document. Documents are concatenated and cut into rows of
    ``seq_len`` tokens; a document that crosses a row boundary continues in the
    next row, and documents longer than ``seq_len`` restart their positions every
    ``seq_len`` tokens, so positions stay inside the rotary cache.
    """

    def __init__(self, *, tokenizer, batch_size, seq_len, rank, world_size):
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(
            _tokenizer_file(tokenizer, "tokenizer.json")
        )
        with open(_tokenizer_file(tokenizer, "tokenizer_config.json")) as f:
            config = json.load(f)
        self.bos_id = self._special_token_id(config["bos_token"])
        self.eos_id = self._special_token_id(config["eos_token"])
        dataset = load_dataset("allenai/c4", name="en", split="train", streaming=True)
        self.dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.documents = None
        # Tokens tokenized but not yet batched: (input, label, position) lists.
        self.buffer = ([], [], [])

    def _special_token_id(self, token):
        content = token["content"] if isinstance(token, dict) else token
        token_id = self.tokenizer.token_to_id(content)
        if token_id is None:
            raise ValueError(f"Tokenizer has no token {content!r}")
        return token_id

    def __iter__(self):
        return self

    def __next__(self):
        if self.documents is None:
            self.documents = iter(self.dataset)
        needed = self.batch_size * self.seq_len
        while len(self.buffer[0]) < needed:
            text = next(self.documents)["text"]
            ids = [self.bos_id, *self.tokenizer.encode(text).ids, self.eos_id]
            self.buffer[0].extend(ids[:-1])
            self.buffer[1].extend(ids[1:])
            self.buffer[2].extend(i % self.seq_len for i in range(len(ids) - 1))
        batch = {}
        for key, values in zip(("input", "labels", "positions"), self.buffer):
            batch[key] = torch.tensor(values[:needed]).view(
                self.batch_size, self.seq_len
            )
            del values[:needed]
        return batch

    def state_dict(self):
        return copy.deepcopy(
            {"dataset": self.dataset.state_dict(), "buffer": self.buffer}
        )

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict["dataset"])
        self.buffer = tuple(list(values) for values in state_dict["buffer"])
        self.documents = None
