# FlexShard

FlexShard is an experimental parameter-sharding library for PyTorch. It groups
model parameters into buckets and lets placement implementations control how
each bucket stores shards, gathers parameters, and reduces gradients.

## Requirements

| Component | Requirement |
| --- | --- |
| Python | 3.10 or later |
| PyTorch | `>=2.14,<2.15` |
| Platform | Linux with NVIDIA GPUs for training |
| CUDA and NCCL | A CUDA-enabled PyTorch build with NCCL and a compatible NVIDIA driver |
| Triton | `~=3.8.0` on Linux |

FlexShard does not pin a separate CUDA toolkit version. Use the CUDA runtime
supported by your PyTorch build and a compatible driver. The example below uses
two GPUs; some tests require four. CPU-only tests cover a subset of the library,
not the CUDA training runtime.

## Installation

Install a CUDA-enabled PyTorch build satisfying the version range above, then
install FlexShard from this repository:

```bash
git clone https://github.com/meta-pytorch/flex_shard.git
cd flex_shard
python -m pip install -e .
```

The distribution is named `flex-shard`; the Python import is `flex_shard`.

## Example

Save this as `train.py`. It shards each parameter along dimension 0 and runs
three training steps. Initialize the same model on every rank and create the
optimizer after applying `flex_shard`.

```python
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from flex_shard import BucketSpec, flex_shard
from flex_shard.custom_placements.shard import per_param_placements


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        torch.manual_seed(0)
        # Initialize on the GPU to avoid copying parameters from the CPU.
        model = nn.Sequential(
            nn.Linear(8, 16, device="cuda"),
            nn.ReLU(),
            nn.Linear(16, 4, device="cuda"),
        )
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                )
            ],
        )
        # Batch optimizer updates to reduce kernel-launch overhead.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        torch.manual_seed(1 + dist.get_rank())
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = model(torch.randn(4, 8, device="cuda")).square().mean()
            loss.backward()
            optimizer.step()
        if dist.get_rank() == 0:
            print(f"Final loss: {loss.item():.4f}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Run one process per GPU:

```bash
torchrun --standalone --nproc-per-node=2 train.py
```

The example keeps full parameters after forward with
`reshard_after_forward=False`. Enabling resharding uses activation recomputation
and requires a compatible model execution pattern.

## Limitations

- The API is experimental and may change. It uses private PyTorch APIs, so the
  declared PyTorch version range matters.
- Training currently requires a one-dimensional CUDA device mesh. CPU offload
  is not supported.
- Every parameter must match exactly one bucket. The example's dimension-0
  sharding placement does not support scalar parameters.
- A regular `state_dict()` contains rank-local shards. Do not treat it as a
  complete, gathered model checkpoint.

## Development and testing

From the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

GPU tests skip when the required hardware is unavailable; a run with skipped
GPU tests does not validate distributed training. Use a host with at least four
GPUs for the full suite.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development and pull-request
workflow and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.

## License

FlexShard is BSD-3-Clause licensed, as found in the [LICENSE](LICENSE) file.
