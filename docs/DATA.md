# Portable input format

The data directory contains:

| File | Contents |
| --- | --- |
| item_tokens.npy | int64 array [items + 1, SID digits], zero row reserved for padding |
| train.npz | input_ids [windows, history length]; labels [windows, 1] with item indices |
| val.npz | input_ids [users, history length]; labels [users, SID digits] |
| test.npz | Same representation as validation |
| item_ids.json | Optional original item IDs indexed in the same order; first entry null |
| metadata.json | Preparation settings and sizes |

All arrays use integer indices. Histories are left padded with zero and must
contain at least one nonpadding item. No user ID or timestamp is consumed by the
model. Their role is to establish the chronological split before preparation.
The same item-to-SID mapping must be used by all three splits and checkpoints.

For four 256-entry codebooks, the preparation command maps the four local codes
to token ranges 3–258, 259–514, 515–770, and 771–1026. The model vocabulary is
1028; entries 0–2 and 1027 are reserved. Supply matching model vocabulary and
history-length settings if changing the preparation options.

Candidates are sorted unique complete SIDs from all nonpadding catalog items.
SID collisions are preserved. Labels for validation and test are complete SIDs,
not item indices or partially matched codes. Evaluation ranks the entire
catalog, keeps previously seen items eligible, and uses no sampled negatives.

Transition memory counts the last-history-SID to next-item-SID pair in every
training window. Diagonal transitions are omitted. Serving retains the eight
largest counts per source, with ascending SID-catalog index breaking ties.
Training stores one additional neighbor so removing the current record can
recover the correct eight neighbors. The decrement applies at every occurrence
of that source SID in the current history. Memory strengths use log(1 + count).
Validation and test records never update the memory.

Domain statistics for the paper inputs are in datasets.json. They describe the
research inputs rather than the synthetic smoke dataset. Raw preprocessing and
SID artifacts need to be supplied separately to recreate those inputs.
