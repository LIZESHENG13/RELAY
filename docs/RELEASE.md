# Release contents

This public source release includes the installable PyTorch/NumPy implementation,
portable data preparation, training and evaluation commands, checkpoint
conversion, parameter and inference-cost measurement, experiment configurations,
CPU tests, and a synthetic functional example. It is licensed under MIT; see
`LICENSE`.

Benchmark datasets and trained checkpoints are distributed separately. The
input contract and chronological preparation are documented in `DATA.md`; users
must obtain source data under its original terms and provide the matching SID
mapping and splits. Training runs, caches, checkpoints, and machine-specific
paths are excluded from this repository.

The paper's signed main configuration is `configs/relay.json`; per-domain
settings are under `configs/datasets/`. The nonnegative controls and multi-query
extensions are included as separate configurations. Run the CPU smoke workflow
and test suite from the repository root before adapting the implementation.
