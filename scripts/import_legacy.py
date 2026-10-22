"""Convert an original RELAY state_dict and optional Arrow caches to portable files.

Only tensor state is read; no historical model/runtime is imported. Data and
weights stay in a user-selected output directory, never in the source package.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from relay import ModelConfig, RELAY
from relay.checkpoint import pack, save_checkpoint
from relay.data import save_split


def convert(checkpoint, output, heads=4, dropout=0.1, norm_eps=1e-5, temperature=0.01):
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = dict(original["model"] if "model" in original else original)
    signature = state.pop("signature").tolist()
    if signature[0] == 3:
        if signature != [3, 0, 0, 2, signature[4], 1]:
            raise ValueError("Expected shared-routing RELAY with both memory channels")
        correction = "nonnegative"
    elif signature[0] == 7:
        correction = ("nonnegative", "candidate_nonnegative", "candidate_signed")[signature[2]]
    else:
        raise ValueError("Checkpoint is not a supported RELAY architecture")
    config = ModelConfig(dim=state["queries"].shape[1], heads=heads,
        inner=state["encoder.0.mlp.c_fc.weight"].shape[0],
        encoder_layers=len({k.split(".")[1] for k in state if k.startswith("encoder.")}),
        readout_layers=len({k.split(".")[1] for k in state if k.startswith("readout.")}),
        max_length=len(state["positions.weight"]), dropout=dropout, norm_eps=norm_eps,
        temperature=temperature, neighbors=state["transition_neighbors"].shape[1],
        queries=len(state["queries"]), correction=correction,
        vocab_size=len(state["embedding.weight"]))
    graph = dict(neighbors=state["exclusion_neighbors"], counts=state["exclusion_counts"])
    model = RELAY(config, state["item_tokens"], graph)
    model.load_state_dict(state, strict=True)
    artifact = pack(model, source_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                    imported_tensor_state=True)
    save_checkpoint(output, artifact)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Portable .pt file")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--source-config", help="Original JSON config or task JSON, for runtime settings")
    parser.add_argument("--cache", help="Optional directory with train/val/test Arrow caches")
    parser.add_argument("--data-output", help="New directory for portable data files")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if bool(args.cache) != bool(args.data_output):
        parser.error("--cache and --data-output must be supplied together")
    if args.data_output and Path(args.data_output).exists():
        raise FileExistsError(args.data_output)
    if args.source_config:
        source = json.loads(Path(args.source_config).read_text())
        source = source.get("reference_config", source)
        args.heads = source.get("n_heads", source.get("heads", args.heads))
        args.dropout = source.get("dropout_rate", source.get("dropout", args.dropout))
        args.norm_eps = source.get("norm_eps", args.norm_eps)
        args.temperature = source.get("di4c_item_tau", source.get("temperature", args.temperature))
    model = convert(args.checkpoint, args.output, args.heads, args.dropout,
                    args.norm_eps, args.temperature)
    if args.cache:
        from datasets import load_from_disk
        target = Path(args.data_output)
        target.mkdir(parents=True, exist_ok=False)
        np.save(target / "item_tokens.npy", model.item_tokens.numpy(), allow_pickle=False)
        sizes = {}
        for split in ("train", "val", "test"):
            arrays = load_from_disk(str(Path(args.cache) / split)).with_format(
                "numpy", columns=["input_ids", "labels"])[:]
            save_split(target / (split + ".npz"), arrays["input_ids"], arrays["labels"])
            sizes[split] = len(arrays["input_ids"])
        (target / "metadata.json").write_text(json.dumps(dict(max_length=model.config.max_length,
            vocab_size=model.config.vocab_size, windows=sizes, imported=True), indent=2) + "\n")
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
                          model=model.config.to_dict()), indent=2))


if __name__ == "__main__":
    main()
