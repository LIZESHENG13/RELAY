"""Generate a small deterministic functional demo; not research benchmark data."""
import argparse
import json
from pathlib import Path
import tempfile
from relay.data import prepare_sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    items = [f"item{i}" for i in range(32)]
    sids = {item: [i % 16, i // 16, (i * 3) % 16, (i * 7) % 16] for i, item in enumerate(items)}
    sequences = {f"user{u}": [items[(u + j * (1 + u % 3)) % 32] for j in range(8)] for u in range(24)}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "sequences.json").write_text(json.dumps(sequences))
        (root / "sids.json").write_text(json.dumps(sids))
        prepare_sequences(root / "sequences.json", root / "sids.json", args.output,
                          max_length=20, codebook_size=16)


if __name__ == "__main__":
    main()
