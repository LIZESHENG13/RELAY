"""Full-catalog CE, validation selection, and resumable optimizer/RNG state."""
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch
from .checkpoint import load_model, pack, save_checkpoint
from .config import ModelConfig, TrainConfig
from .data import load_split, load_tokens, validate_split
from .memory import build_memory
from .metrics import evaluate
from .model import RELAY


def fingerprint(folder):
    result = {}
    for name in ("item_tokens.npy", "train.npz", "val.npz"):
        digest = hashlib.sha256()
        with (Path(folder) / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = digest.hexdigest()
    return result


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def train(folder, output, model_config, config, device="cpu", initial=None, resume=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Run directory is not empty; use a new directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    files = fingerprint(folder)
    data = {split: load_split(folder, split, device) for split in ("train", "val")}
    tokens = load_tokens(folder)
    for split in data:
        validate_split(data[split], tokens, split, model_config.max_length)
    if data["train"]["labels"].shape[1] != 1 or data["val"]["labels"].shape[1] != tokens.shape[1]:
        raise ValueError("Train labels are item indices; validation labels are complete SIDs")
    graph = build_memory(tokens, data["train"]["input_ids"].cpu(),
                         data["train"]["labels"].cpu(), model_config.neighbors)
    resume_path = output / "last.pt"
    artifact = None
    if resume:
        model, artifact = load_model(resume_path, device)
        if (artifact["cuda_rng"] is not None) != str(device).startswith("cuda"):
            raise ValueError("Resume requires the same CPU/CUDA device type")
        if artifact["model_config"] != model_config.to_dict():
            raise ValueError("Resume model config differs")
        if artifact["train_config"] != config.to_dict() or artifact["input_sha256"] != files:
            raise ValueError("Resume training configuration or inputs differ")
    elif initial:
        model, artifact = load_model(initial, device)
        if model.config.to_dict() != model_config.to_dict():
            raise ValueError("Initial checkpoint and requested model configurations differ")
    else:
        model = RELAY(model_config, tokens, graph).to(device)
    if not torch.equal(model.item_tokens.cpu(), tokens):
        raise ValueError("Checkpoint SID mapping differs from data")
    if not (torch.equal(model.exclusion_neighbors.cpu(), graph["neighbors"])
            and torch.equal(model.exclusion_counts.cpu(), graph["counts"])):
        raise ValueError("Checkpoint memory differs from training-only counts")
    if data["train"]["input_ids"].shape[1] != model_config.max_length:
        raise ValueError("History width differs from model max_length")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, fused=str(device).startswith("cuda"))
    steps = math.ceil(len(data["train"]["input_ids"]) / config.batch_size)
    total = config.epochs * steps
    def schedule(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total - config.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    generator = torch.Generator().manual_seed(config.seed)
    start, best_score, best_epoch, best_metrics = 0, -1.0, 0, {}
    if resume:
        optimizer.load_state_dict(artifact["optimizer"])
        scheduler.load_state_dict(artifact["scheduler"])
        generator.set_state(artifact["data_rng"])
        torch.set_rng_state(artifact["cpu_rng"])
        random.setstate(artifact["python_rng"])
        if str(device).startswith("cuda"):
            if artifact["cuda_rng"] is None:
                raise ValueError("Cannot resume CPU optimizer state as an identical CUDA run")
            torch.cuda.set_rng_state(artifact["cuda_rng"], device=device)
        start, best_score, best_epoch = artifact["epoch"], artifact["best_score"], artifact["best_epoch"]
        best_metrics = artifact["best_validation"]
    (output / "config.json").write_text(json.dumps(dict(model=model_config.to_dict(),
        training=config.to_dict(), input_sha256=files), indent=2) + "\n")
    history_path = output / "history.jsonl"
    if resume and history_path.exists():
        rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        history_path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["epoch"] <= start))
    if resume and start - best_epoch >= config.patience:
        print("Run already met the early stopping criterion.", flush=True)
    else:
        for epoch in range(start, config.epochs):
            model.train()
            order = torch.randperm(len(data["train"]["input_ids"]), generator=generator).to(device)
            started, running = time.perf_counter(), 0.0
            for indices in order.split(config.batch_size):
                batch = {k: v.index_select(0, indices) for k, v in data["train"].items()}
                optimizer.zero_grad(set_to_none=True)
                loss = model(batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training loss is not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, foreach=True)
                optimizer.step(); scheduler.step()
                running += loss.detach().item()
            validation, _ = evaluate(model, data["val"], config.eval_batch_size)
            if validation["score@10"] > best_score:
                best_score, best_epoch, best_metrics = validation["score@10"], epoch + 1, validation
                save_checkpoint(output / "best.pt", pack(model, seed=config.seed,
                    best_epoch=best_epoch, validation=best_metrics))
            row = dict(epoch=epoch + 1, train_loss=running / steps, validation=validation,
                       best_epoch=best_epoch, elapsed_seconds=time.perf_counter() - started)
            with history_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            artifact = pack(model)
            artifact.update(train_config=config.to_dict(), input_sha256=files, epoch=epoch + 1,
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                data_rng=generator.get_state(), cpu_rng=torch.get_rng_state(), python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None,
                best_score=best_score, best_epoch=best_epoch, best_validation=best_metrics)
            save_checkpoint(resume_path, artifact)
            if epoch + 1 - best_epoch >= config.patience:
                break
    summary = dict(best_epoch=best_epoch, validation=best_metrics,
        parameters=sum(p.numel() for p in model.parameters()), test_evaluations=0)
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
