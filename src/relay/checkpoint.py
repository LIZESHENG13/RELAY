"""Versioned portable checkpoints; no reference model or absolute paths required."""
from pathlib import Path
import torch
from .config import ModelConfig
from .model import RELAY

FORMAT = "relay-v1"


def pack(model, **metadata):
    return dict(format=FORMAT, model_config=model.config.to_dict(),
                model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                metadata=metadata)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_model(path, device="cpu"):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format") != FORMAT:
        raise ValueError("Expected a portable relay-v1 checkpoint")
    state = artifact["model"]
    graph = dict(neighbors=state["exclusion_neighbors"], counts=state["exclusion_counts"])
    model = RELAY(ModelConfig(**artifact["model_config"]), state["item_tokens"], graph)
    model.load_state_dict(state, strict=True)
    return model.to(device), artifact
