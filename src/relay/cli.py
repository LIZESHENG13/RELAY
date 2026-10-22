"""Portable train/evaluate/benchmark/prepare commands."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import torch
from .benchmark import benchmark
from .checkpoint import load_model
from .config import ModelConfig, TrainConfig
from .data import load_split, load_tokens, prepare_sequences, validate_split
from .metrics import evaluate
from .training import seed_everything, train


def config_file(path):
    data = json.loads(Path(path).read_text())
    for key, cls in (("model", ModelConfig), ("training", TrainConfig)):
        unknown = set(data.get(key, {})) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {key} options: {sorted(unknown)}")
    return ModelConfig(**data.get("model", {})), TrainConfig(**data.get("training", {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare chronological sequences with precomputed SIDs")
    prepare.add_argument("--sequences", required=True)
    prepare.add_argument("--sids", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--max-length", type=int, default=20)
    prepare.add_argument("--codebook-size", type=int, default=256)
    prepare.add_argument("--sid-offset", type=int, default=3)
    training = commands.add_parser("train", help="Train using only train/validation data")
    training.add_argument("--data", required=True)
    training.add_argument("--config", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--device", default="cpu")
    training.add_argument("--initial", help="Optional portable initialization checkpoint")
    training.add_argument("--resume", action="store_true")
    for name in ("evaluate", "benchmark"):
        sub = commands.add_parser(name)
        sub.add_argument("--data", required=True)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--split", choices=("val", "test"), default="test")
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--batch-size", type=int, default=256)
        sub.add_argument("--output")
        if name == "benchmark":
            sub.add_argument("--repeats", type=int, default=6)
            sub.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(2024)
    if args.command == "prepare":
        prepare_sequences(args.sequences, args.sids, args.output, args.max_length,
                          args.codebook_size, args.sid_offset)
        result = dict(stage="prepared", output=args.output)
    elif args.command == "train":
        model_config, training_config = config_file(args.config)
        result = train(args.data, args.output, model_config, training_config,
                       args.device, args.initial, args.resume)
    else:
        model, artifact = load_model(args.checkpoint, args.device)
        data = load_split(args.data, args.split, args.device)
        if not torch.equal(load_tokens(args.data), model.item_tokens.cpu()):
            raise ValueError("Dataset item-to-SID mapping differs from the checkpoint")
        validate_split(data, model.item_tokens, args.split, model.config.max_length)
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        if args.command == "evaluate":
            result, _ = evaluate(model, data, args.batch_size)
            result = dict(split=args.split, users=len(data["input_ids"]), metrics=result)
        else:
            result = benchmark(model, data, args.batch_size, args.repeats, args.cuda_graph)
            result["split"] = args.split
        if args.output:
            output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
