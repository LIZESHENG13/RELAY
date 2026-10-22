"""Serializable model and training settings."""
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    dim: int = 256
    heads: int = 4
    inner: int = 1024
    encoder_layers: int = 1
    readout_layers: int = 1
    max_length: int = 20
    dropout: float = 0.1
    norm_eps: float = 1e-5
    temperature: float = 0.01
    neighbors: int = 8
    queries: int = 1
    correction: str = "candidate_signed"
    vocab_size: int = 1028

    def __post_init__(self):
        if self.dim <= 0 or self.heads <= 0 or self.dim % self.heads:
            raise ValueError("dim must be positive and divisible by heads")
        if self.queries not in (1, 2, 4):
            raise ValueError("queries must be 1, 2, or 4")
        if self.correction not in ("nonnegative", "candidate_nonnegative", "candidate_signed"):
            raise ValueError("Unknown memory correction")
        if self.correction != "nonnegative" and self.queries != 1:
            raise ValueError("Candidate-conditioned extensions currently use one query")
        if min(self.encoder_layers, self.readout_layers, self.max_length, self.neighbors) < 1:
            raise ValueError("Layer counts, max_length, and neighbors must be positive")
        if self.temperature <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid temperature or dropout")
        if self.inner < 1 or self.norm_eps <= 0 or self.vocab_size < 2:
            raise ValueError("Invalid feed-forward width, normalization epsilon, or vocabulary")

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    seed: int = 2024
    epochs: int = 150
    patience: int = 10
    batch_size: int = 1024
    eval_batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 10000
    max_grad_norm: float = 1.0

    def __post_init__(self):
        if min(self.epochs, self.patience, self.batch_size, self.eval_batch_size) < 1:
            raise ValueError("Epoch, patience, and batch sizes must be positive")
        if self.learning_rate <= 0 or self.warmup_steps < 0:
            raise ValueError("Invalid optimizer or warmup settings")
        if self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("Weight decay must be nonnegative and gradient norm positive")

    def to_dict(self):
        return asdict(self)
