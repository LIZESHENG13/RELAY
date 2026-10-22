"""Training-only transition counts with one backup for exact record exclusion."""
from collections import Counter
import torch


def catalog_from_tokens(item_tokens):
    if item_tokens.ndim != 2 or len(item_tokens) < 2:
        raise ValueError("item_tokens must contain a padding row and real item SID rows")
    if item_tokens[0].ne(0).any() or item_tokens[1:].le(0).any():
        raise ValueError("Row zero is padding; real SID tokens must be positive")
    catalog, inverse = torch.unique(item_tokens[1:], dim=0, return_inverse=True)
    mapping = torch.nn.functional.pad(inverse, (1, 0), value=-1)
    return catalog, mapping


def build_memory(item_tokens, histories, targets, neighbors=8):
    """Count each supplied training window's last SID -> target SID once.

    Self-transitions are excluded. Tie order is increasing catalog index.
    No validation/test input is accepted by this function.
    """
    catalog, mapping = catalog_from_tokens(item_tokens)
    histories, targets = histories.cpu().long(), targets.cpu().long().reshape(-1)
    if histories.ndim != 2 or len(histories) != len(targets) or not len(targets):
        raise ValueError("Expected nonempty aligned histories and training item targets")
    if histories.min() < 0 or histories.max() >= len(item_tokens):
        raise ValueError("History item index outside item_tokens")
    if targets.min() <= 0 or targets.max() >= len(item_tokens):
        raise ValueError("Training targets must be real item indices")
    valid = histories.ne(0)
    if not valid.any(-1).all():
        raise ValueError("Every history must contain a real item")
    position = torch.arange(histories.shape[1])[None].expand_as(histories)
    last = position.masked_fill(~valid, -1).amax(-1)
    sources = mapping[histories[torch.arange(len(histories)), last]]
    destinations = mapping[targets]
    counts = [Counter() for _ in range(len(catalog))]
    for source, destination in zip(sources.tolist(), destinations.tolist()):
        if source != destination:
            counts[source][destination] += 1
    indices = torch.zeros(len(catalog), neighbors + 1, dtype=torch.long)
    values = torch.zeros_like(indices)
    for source, row in enumerate(counts):
        ordered = sorted(row.items(), key=lambda entry: (-entry[1], entry[0]))[:neighbors + 1]
        for column, (destination, count) in enumerate(ordered):
            indices[source, column], values[source, column] = destination, count
    return {"neighbors": indices, "counts": values}
