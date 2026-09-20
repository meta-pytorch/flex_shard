# DistMuon provenance

FlexShard maintains this implementation independently. Runtime imports use
FlexShard and PyTorch; TorchTitan is not a package dependency.

Upstream repository: https://github.com/pytorch/torchtitan

Pinned revision: `610bb6f6b99d16f2314f9ddf520ab3cd2423ebc0`.

The source and test copyright headers are retained. `LICENSE.torchtitan` is an
exact copy of the upstream BSD-3-Clause license at that revision.

## Imported files

Production destinations are relative to `src/flex_shard/dist_muon/`.
Test destinations below are relative to `src/flex_shard/`.

| Upstream path | Local destination | Upstream SHA-256 |
| --- | --- | --- |
| `torchtitan/distributed/flex_shard/dist_muon.py` | `optimizer.py` | `24e4003aa4606e2c31e65853d2ab9bea95fbdabc04101fe8c6ae59072f66e06a` |
| `torchtitan/distributed/flex_shard/optimizer_reshard.py` | `optimizer_reshard.py` | `f2ffcb294068ea7edf309d4a23b6f6b58f273bd7f21bd14e0041abe0d913ec3b` |
| `torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py` | `_optimizer_reshard_schedule.py` | `59b154b6f8159badf5046f2d76b13beb16a0b64019cdb62a141e186fd514d861` |
| `torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py` | `_optimizer_reshard_runtime.py` | `0e645d7b22c1e42c615fc495e0098ce74e2fea3ea65b4d4f36b5f0339e6b730d` |
| `LICENSE` | `LICENSE.torchtitan` | `6eea30995941126beeb99ef775f0968ed8320beb4834ad28d6ae6704a1a92930` |
| `tests/unit_tests/cpu/flex_shard/test_dist_muon_storage_validation.py` | `tests/test_dist_muon_storage_validation.py` | `1117167fa17fdca33997e376720bf548ed36310a163564da7266719e1ca0946a` |
| `tests/unit_tests/cpu/flex_shard/test_optimizer_reshard_schedule.py` | `tests/test_optimizer_reshard_schedule.py` | `c1d04f4c979b4037f24bb71110ee973dcfb12ee32f99e30431f59dd6d3f31f79` |
| `tests/unit_tests/cpu/flex_shard/test_optimizer_reshard_runtime.py` | `tests/test_optimizer_reshard_runtime.py` | `5654d9a6a80a9e39ae84d975a426b64490d71c8ed17b47cd231aac1a8a392ca6` |
| `tests/unit_tests/gpu/flex_shard/test_dist_muon.py` | `tests/test_dist_muon_distributed.py` | `4f3620a80485ac4598d4e67216de6745d7f0fe01347f0bdca14bd69d5ac9441d` |

## Local changes

- Rename `dist_muon.py` to `optimizer.py` and update imports to the FlexShard
  package. The three redistribution helper modules retain their upstream source.
- Extract the upstream numerical kernels into `_muon_math.py` for both execution
  paths; retain their behavior and re-export the previous private helper names
  from `optimizer.py` for compatibility with the imported tests.
- Extend `optimizer.py` with native FlexShard parameter dispatch and optional
  compute configuration. Implement native layout validation, local execution,
  and checkpoint handling in `_native.py`, retaining the DTensor redistribution
  path. Native execution uses real local parameters and does not construct
  adapter DTensors or optimizer process groups.
- Export `DistMuon` and `build_dist_muon` from `flex_shard.dist_muon`; preserve
  the existing explicit-factory adapter and its exports.
- Port the upstream CPU tests and two-/four-GPU tests with local import paths.
  The distributed checkpoint test uses PyTorch
  `get_optimizer_state_dict` / `set_optimizer_state_dict` with
  `StateDictOptions(flatten_optimizer_state_dict=True)` and a small test-only
  module exposing the same parameters by FQN. This replaces TorchTitan optimizer
  utility imports without adding a production checkpoint-helper dependency.
- Include this provenance file and the upstream license in source and wheel
  distributions. Preserve the FlexShard Python and dependency requirements.

Future upstream fixes should be imported as reviewed changes, with numerical,
redistribution, and checkpoint regression checks and updated source provenance.
