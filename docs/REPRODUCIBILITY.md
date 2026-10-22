# Reproducibility

## Default research configuration

The paper's main configuration uses one query and candidate-conditioned signed
correction. It has dimension 256, four attention heads, one history encoder
layer, one shared readout layer, feed-forward width 1024, and history length 20.
Dropout is 0.1, RMSNorm epsilon is 1e-5, semantic-score temperature is 0.01
(0.1 for Instruments), and transition degree is eight.
Per-domain configurations retain this difference. There are 2,043,650 learned
parameters for four-code SIDs. Catalog entries and transition counts are
buffers and are excluded from this parameter count.

The optimizer is AdamW with learning rate 1e-4, weight decay zero, batch size
1024, gradient norm clipping at one, a 10,000-step linear warmup, and cosine
decay over at most 150 epochs. Validation batch size is 256; patience is ten
epochs. The selected metric is 0.8 NDCG@10 + 0.2 Recall@10. The default seed is
2024; the paper's main evaluation averages five seeds 2024–2028.

Evaluation uses FP32 scores over every distinct legal complete SID and returns
Recall/NDCG at 1, 5, and 10. NDCG accumulation in this package uses float64;
development reports accumulated FP32 discounts, so metric values can differ at
the final few decimal places even for identical ranked predictions.

## Initial weights and resumed runs

An imported tensor checkpoint retains its weights, SID catalog, memory buffers,
and scoring equations. Conversion is followed by strict state loading. Native
initialization is self-contained and deterministic but consumes random numbers
in a different order from the development implementation. Do not interpret a
matching integer seed as a guarantee of identical fresh-training weights.

The epoch checkpoint stores model, AdamW state, learning-rate scheduler, shuffle
generator, CPU/CUDA RNG, Python RNG, best validation record, and input hashes.
Resume restores these states and discards history entries after the checkpoint
epoch. The CPU regression test interrupts during the second epoch and verifies
bitwise equal final tensors against an uninterrupted run with dropout enabled.
Exact replay across different PyTorch versions, devices, or CUDA kernels is not
promised. Do not resume an interrupted run on a different device type.

## Correction variants and extensions

The released main configuration uses one query and candidate-conditioned signed
correction. Its gate is initialized as 0.5 x + log(2). Candidate-conditioned
nonnegative correction uses the same gate parameters with softplus; the two
gates have equal value and derivative at zero initialization. The simpler
nonnegative configuration uses a shared nonnegative scale. These alternatives
are available in `configs/nonnegative.json` and
`configs/candidate_nonnegative.json`.

Two/four-query configurations share the encoder and readout and add learned
query vectors. Each query produces complete corrected scores, and the model
takes their maximum per candidate. These configurations retain shared routing
and are available as extensions in `configs/query2.json` and `configs/query4.json`.
Candidate-conditioned correction currently requires one query. Signed scoring
uses the full catalog; the exact candidate-union property applies to one-query
nonnegative scoring only.

## Verification scope

The preparation audit runs on CPU while research jobs use the GPU. It compares
loaded real weights against the development model, checks parameter counts,
scores, ordered Top-10 predictions, training loss, and gradients, and rechecks
the full Toys validation set. The machine-readable summary is recorded in
verification.json: all eleven loaded configurations have identical CPU scores,
ordered Top-10, training loss, and gradients on the audited batches. The six
domains' training-memory tables are also reconstructed exactly from the input
records. The complete 19,412-user Toys validation rankings match, with maximum
metric difference 5.3e-10 from the development report. CPU smoke timings are execution
checks only, not replacement measurements for paper inference cost. The release
CUDA Graph path requires its own on-device validation when used.
