<p align="center">
  <img src="assets/logo.png" alt="RELAY logo" width="680">
</p>

# RELAY

PyTorch implementation of RELAY for sequential recommendation with semantic
item IDs (SIDs). One shared attention readout supplies both the semantic score
and the routing weights for sparse history and transition memory. The paper's
main configuration uses one query with candidate-conditioned signed memory
correction. Nonnegative correction and two/four-query configurations are also
provided for controlled comparisons and model extensions.

Anonymous repository: [RELAY](https://anonymous.4open.science/r/RELAY-F98C).

Model implementation: [src/relay/model.py](src/relay/model.py). The command-line
entry points are in [src/relay/cli.py](src/relay/cli.py), with default and per-domain
configurations under [configs](configs).

## Install

Python 3.10+ is required. Install a PyTorch build suitable for your hardware, then
run from this directory:

    python -m pip install -e .
    python -m relay --help

The runtime depends only on PyTorch and NumPy. It does not import a separate
recommendation repository, Transformers, or a local reference-model directory.

## CPU smoke run

    python scripts/make_toy_data.py --output /tmp/relay-demo-data
    python -m relay train --data /tmp/relay-demo-data --config configs/demo.json --output /tmp/relay-demo-run
    python -m relay evaluate --data /tmp/relay-demo-data --checkpoint /tmp/relay-demo-run/best.pt
    python -m relay benchmark --data /tmp/relay-demo-data --checkpoint /tmp/relay-demo-run/best.pt --batch-size 16

The generated dataset is synthetic and checks execution only. Its metrics are
not research results. Use new output directories for each run.

## Prepare a dataset

Supply two JSON objects:

    sequences.json: {"user_1": ["item_a", "item_b", "item_c", "item_d", "item_e"]}
    sids.json: {"item_a": [0, 7, 19, 2], "item_b": [1, 4, 3, 9], ...}

Each sequence must already follow chronological order. Each item in a sequence
must have a SID, every SID must have the same number of codes, and local codes
must lie in [0, codebook_size). Users need at least four interactions. The last
two interactions become validation and test targets; all earlier nonempty
prefixes become training examples. Histories are left padded and truncated to
the most recent 20 interactions by default.

    python -m relay prepare --sequences sequences.json --sids sids.json --output data/toys

The model consumes integer item indices in histories. The preparation command
creates a consistent item-index mapping, reserves index zero for padding, and
applies a separate token offset to each SID position. Items with identical SIDs
remain distinct in the history mapping and share one candidate in the SID
catalog. See [the data contract](docs/DATA.md) for the portable file format.

This package starts from chronological sequences and precomputed SIDs. Raw
Amazon downloads, text embedding, and OPQ fitting are not included. To reproduce
the paper, use the same filtering, item mapping, and four-code SIDs as the
released benchmark inputs; arbitrary newly fitted SIDs define a different input.

## Train and evaluate

    python -m relay train --data data/toys --config configs/relay.json --output runs/toys-seed2024 --device cuda:0
    python -m relay evaluate --data data/toys --checkpoint runs/toys-seed2024/best.pt --device cuda:0 --output runs/toys-seed2024/test.json

Training reads only the training and validation files. It builds transition
counts from training records, removes the current record's count during
training, and selects the best checkpoint using validation score@10. Evaluation
uses exact matching of complete SIDs against the full distinct-SID catalog.
Reported recall and NDCG values are fractions, not percentages.

Each run records its configuration, hashes of training inputs, epoch history,
best weights, and a resumable epoch checkpoint. Resume with the same data,
configuration, device type, and run directory:

    python -m relay train --data data/toys --config configs/relay.json --output runs/toys-seed2024 --device cuda:0 --resume

For different seeds, edit training.seed in a copied configuration and use a new
run directory. The paper protocol uses seeds 2024–2028; the default file selects
2024. Per-domain files are provided under configs/datasets; Instruments uses
temperature 0.1 and the other five domains use 0.01. Native initialization is
deterministic for a fixed setup, but its random
draw order differs from the development training stack. Importing trained
weights preserves the model computation; a new native training run is not a
bitwise replay of a historical training trajectory.

## Parameters and inference cost

    python -m relay benchmark --data data/toys --checkpoint runs/toys-seed2024/best.pt --device cuda:0 --batch-size 256 --repeats 6 --cuda-graph --output runs/toys-seed2024/cost.json

The command reports trainable parameter count and the median elapsed time of six
complete passes by default. Timing includes history encoding, complete-catalog
scores, memory correction, and Top-10 SID lookup with cached catalog vectors.
Data loading, host-to-device transfers, preparation, and warmup are excluded.
CUDA Graph mode verifies ordered Top-10 equality against eager execution on
every batch and checks input updates and output ownership before timing. Omit
--cuda-graph for eager CPU/GPU timing. FP32 and disabled TF32 match the experiment
protocol; compare costs on the same hardware with the same execution mode.

The exact candidate-union helper is available for one-query nonnegative models.
The signed main configuration uses complete-catalog ranking, as does the
benchmark for every variant.

## Import existing experiment artifacts

The optional converter reads supported development RELAY tensor checkpoints
without importing their source code. Install the optional dependency only if
converting Arrow dataset caches:

    python -m pip install -e '.[legacy-data]'
    python scripts/import_legacy.py --checkpoint experiment/best.pth --output imported/model.pt --cache experiment/cache --data-output imported/data
    python -m relay evaluate --data imported/data --checkpoint imported/model.pt

The converter defaults to four attention heads, dropout 0.1, RMSNorm epsilon
1e-5, and temperature 0.01; supply the corresponding flags if the source run used
different values. These settings cannot all be recovered from tensor shapes.
Pass --source-config experiment/task.json to read runtime settings from the
original JSON configuration; in particular, the Instruments temperature is 0.1.

## Verification and release status

    python -m unittest discover -s tests -v

Tests cover exact record exclusion, SID collisions, shared routing, signed
initialization, multi-query gradients, checkpoint round trips, and exact resume
after an interrupted epoch. See [reproducibility notes](docs/REPRODUCIBILITY.md)
and [release notes](docs/RELEASE.md). Benchmark datasets and learned checkpoints
are distributed separately from source code under their respective terms.

## License

This project is released under the [MIT License](LICENSE).
