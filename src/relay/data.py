"""Portable NPZ data contract and chronological sequence preparation."""
from pathlib import Path
import json
import numpy as np
import torch


def save_split(path, histories, labels):
    np.savez_compressed(path, input_ids=np.asarray(histories, dtype=np.int64),
                        labels=np.asarray(labels, dtype=np.int64))


def load_split(folder, split, device="cpu"):
    with np.load(Path(folder) / (split + ".npz"), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in ("input_ids", "labels")}
    ids, labels = arrays["input_ids"], arrays["labels"]
    if not all(np.issubdtype(v.dtype, np.integer) for v in arrays.values()):
        raise ValueError("History and label arrays must contain integer indices")
    if ids.ndim != 2 or len(ids) == 0 or labels.ndim != 2 or len(ids) != len(labels):
        raise ValueError("Split must contain nonempty aligned rank-two arrays")
    if ids.min() < 0 or not (ids != 0).any(-1).all():
        raise ValueError("Histories must contain nonnegative item indices and at least one real item")
    return {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in arrays.items()}


def load_tokens(folder):
    array = np.load(Path(folder) / "item_tokens.npy", allow_pickle=False)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("item_tokens must be a rank-two integer array")
    return torch.from_numpy(array).long()


def validate_split(data, tokens, split, max_length):
    ids, labels = data["input_ids"], data["labels"]
    if ids.shape[1] != max_length or ids.max().item() >= len(tokens):
        raise ValueError("History width or item indices differ from the catalog")
    if split == "train":
        if labels.shape[1] != 1 or labels.min().item() < 1 or labels.max().item() >= len(tokens):
            raise ValueError("Training targets must be valid nonpadding item indices")
    else:
        if labels.shape[1] != tokens.shape[1]:
            raise ValueError("Evaluation targets must be complete semantic IDs")
        legal = {tuple(row) for row in tokens[1:].cpu().tolist()}
        if any(tuple(row) not in legal for row in labels.cpu().tolist()):
            raise ValueError("Evaluation target SID is missing from the item catalog")


def prepare_sequences(sequence_file, sid_file, output, max_length=20, codebook_size=256, offset=3):
    """Sequences must already be ordered chronologically and filtered as intended.

    sequences.json: {user: [raw_item_id, ...]}; sids.json: {raw_item_id: [local_code, ...]}.
    Last two interactions are validation and test targets. Earlier prefixes train.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    sequences = json.loads(Path(sequence_file).read_text())
    codes = {str(k): v for k, v in json.loads(Path(sid_file).read_text()).items()}
    if not sequences or not codes or max_length < 1 or codebook_size < 1 or offset < 1:
        raise ValueError("Nonempty sequence/SID mappings and positive max_length required")
    digits = len(next(iter(codes.values())))
    if digits < 1:
        raise ValueError("Each SID must have at least one code")
    for code in codes.values():
        if len(code) != digits or not all(isinstance(v, int) and 0 <= v < codebook_size for v in code):
            raise ValueError("SIDs must have equal length and valid local code indices")
    item_ids = list(codes)
    mapping = {item: i + 1 for i, item in enumerate(item_ids)}
    offsets = offset + codebook_size * np.arange(digits)
    tokens = np.zeros((len(item_ids) + 1, digits), dtype=np.int64)
    tokens[1:] = np.asarray([codes[item] for item in item_ids]) + offsets
    histories = {split: [] for split in ("train", "val", "test")}
    labels = {split: [] for split in histories}
    for sequence in sequences.values():
        if len(sequence) < 4:
            raise ValueError("Each user must have at least four chronological interactions")
        try:
            ids = [mapping[str(item)] for item in sequence]
        except KeyError as error:
            raise ValueError("Sequence item is missing from SID mapping") from error
        for split, ends in (("train", range(1, len(ids) - 2)),
                            ("val", [len(ids) - 2]), ("test", [len(ids) - 1])):
            for end in ends:
                history = np.zeros(max_length, dtype=np.int64)
                prefix = ids[max(0, end - max_length):end]
                history[-len(prefix):] = prefix
                histories[split].append(history)
                labels[split].append([ids[end]] if split == "train" else tokens[ids[end]].tolist())
    output.mkdir(parents=True)
    np.save(output / "item_tokens.npy", tokens, allow_pickle=False)
    for split in histories:
        save_split(output / (split + ".npz"), histories[split], labels[split])
    (output / "item_ids.json").write_text(json.dumps([None] + item_ids, ensure_ascii=False) + "\n")
    (output / "metadata.json").write_text(json.dumps(dict(max_length=max_length,
        digits=digits, codebook_size=codebook_size, sid_offset=offset,
        vocab_size=offset + digits * codebook_size + 1,
        users=len(sequences), items=len(item_ids),
        windows={split: len(histories[split]) for split in histories}), indent=2) + "\n")
