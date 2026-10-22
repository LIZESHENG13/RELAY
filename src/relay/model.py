"""Self-contained RELAY with checkpoint-compatible names and scoring operations."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig
from .memory import catalog_from_tokens


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.inner)
        self.c_proj = nn.Linear(config.inner, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.dim = config.heads, config.dim
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.proj = nn.Linear(config.dim, config.dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, valid):
        batch, length, dim = x.shape
        query, key, value = self.qkv(x).chunk(3, -1)
        query, key, value = [v.view(batch, length, self.heads, -1).transpose(1, 2)
                             for v in (query, key, value)]
        attention = (query @ key.transpose(-2, -1)) * ((dim // self.heads) ** -0.5)
        attention = attention.masked_fill(~valid[:, None, None, :], -float("inf"))
        attention = self.attn_dropout(attention.softmax(-1))
        value = (attention @ value).transpose(1, 2).contiguous().view(batch, length, dim)
        return self.resid_dropout(self.proj(value))


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x, valid):
        x = x + self.attn(self.ln_1(x), valid)
        return x + self.mlp(self.ln_2(x))


class ReadoutBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.query = nn.Linear(config.dim, config.dim, bias=False)
        self.projection = nn.Linear(config.dim, config.dim)
        self.norm1 = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.norm2 = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn = FeedForward(config)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.attention_dropout = config.dropout

    def forward(self, hidden, memory, valid):
        batch, count, dim = hidden.shape
        query = self.query(self.norm1(hidden)).view(batch, count, self.heads, -1).transpose(1, 2)
        key = memory.view(batch, memory.shape[1], self.heads, -1).transpose(1, 2)
        context = F.scaled_dot_product_attention(query, key, key,
            attn_mask=valid[:, None, None, :],
            dropout_p=self.attention_dropout if self.training else 0.0)
        logits = (query @ key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        attention = logits.masked_fill(~valid[:, None, None, :], -float("inf")).softmax(-1)
        context = context.transpose(1, 2).contiguous().reshape(batch, count, dim)
        hidden = hidden + self.residual_dropout(self.projection(context))
        hidden = hidden + self.ffn(self.norm2(hidden))
        return hidden, attention.mean(1)


@dataclass
class Output:
    loss: torch.Tensor
    metrics: dict


class RELAY(nn.Module):
    def __init__(self, config: ModelConfig, item_tokens, graph):
        super().__init__()
        self.config = config
        item_tokens = torch.as_tensor(item_tokens, dtype=torch.long).cpu()
        catalog, mapping = catalog_from_tokens(item_tokens)
        if item_tokens.max() >= config.vocab_size:
            raise ValueError("SID token outside configured embedding vocabulary")
        self.digits = item_tokens.shape[1]
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.item_mlp = nn.Sequential(nn.Linear(self.digits * config.dim, config.dim),
                                     nn.ReLU(), nn.Linear(config.dim, config.dim))
        self.positions = nn.Embedding(config.max_length, config.dim)
        self.encoder = nn.ModuleList(EncoderBlock(config) for _ in range(config.encoder_layers))
        self.encoder_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.readout = nn.ModuleList(ReadoutBlock(config) for _ in range(config.readout_layers))
        self.readout_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.queries = nn.Parameter(torch.empty(config.queries, config.dim))
        self.memory_scale = nn.Linear(config.dim, 2)
        self.register_buffer("item_tokens", item_tokens)
        self.register_buffer("catalog_tokens", catalog)
        self.register_buffer("item_to_catalog", mapping)
        neighbors = torch.as_tensor(graph["neighbors"], dtype=torch.long).cpu()
        counts = torch.as_tensor(graph["counts"], dtype=torch.long).cpu()
        if neighbors.shape != (len(catalog), config.neighbors + 1) or counts.shape != neighbors.shape:
            raise ValueError("Graph must contain B+1 neighbors for every distinct SID")
        if counts.min() < 0 or neighbors.min() < 0 or neighbors.max() >= len(catalog):
            raise ValueError("Invalid memory counts or destination indices")
        self.register_buffer("transition_neighbors", neighbors[:, :config.neighbors].clone())
        self.register_buffer("transition_strength", counts[:, :config.neighbors].float().log1p())
        self.register_buffer("exclusion_neighbors", neighbors.clone())
        self.register_buffer("exclusion_counts", counts.clone())
        self._catalog_cache = None
        self.apply(self._initialize)
        nn.init.normal_(self.queries, std=0.02)
        nn.init.zeros_(self.memory_scale.weight)
        nn.init.zeros_(self.memory_scale.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def train(self, mode=True):
        self._catalog_cache = None
        return super().train(mode)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._catalog_cache = None
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _apply(self, fn, recurse=True):
        self._catalog_cache = None
        return super()._apply(fn, recurse=recurse)

    def encode(self, ids):
        valid = ids.ne(0)
        positions = self.positions(torch.arange(ids.shape[1], device=ids.device))[None]
        memory = self.item_mlp(self.embedding(self.item_tokens[ids]).flatten(-2)) + positions
        memory = self.dropout(memory * valid[:, :, None])
        for block in self.encoder:
            memory = block(memory, valid)
        return self.encoder_norm(memory), valid

    def readout_evidence(self, ids):
        memory, valid = self.encode(ids)
        hidden = self.dropout(self.queries[None].expand(len(ids), -1, -1))
        for block in self.readout:
            hidden, attention = block(hidden, memory, valid)
        state = self.readout_norm(hidden)
        repeated = state[:, :, None].expand(-1, -1, self.digits, -1).reshape(
            len(ids) * self.config.queries, -1)
        user = F.normalize(self.item_mlp(repeated), dim=-1).reshape(len(ids), self.config.queries, -1)
        return user, state, attention

    def item_vectors(self):
        if self._catalog_cache is not None and not self.training:
            return self._catalog_cache
        embedded = self.embedding(self.catalog_tokens).detach()
        vectors = F.normalize(self.item_mlp(embedded.flatten(-2)), dim=-1)
        # Do not retain a stale autograd graph across gradient-enabled calls.
        if not self.training and not torch.is_grad_enabled():
            self._catalog_cache = vectors
        return vectors

    def transition_table(self, ids, excluded_target=None):
        classes = self.item_to_catalog[ids].clamp_min(0)
        if excluded_target is None:
            return self.transition_neighbors[classes], self.transition_strength[classes]
        valid = ids.ne(0)
        positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        last = positions.masked_fill(~valid, -1).amax(-1)
        last_class = classes[torch.arange(len(ids), device=ids.device), last]
        neighbors, counts = self.exclusion_neighbors[classes], self.exclusion_counts[classes]
        own_edge = ((classes == last_class[:, None])[:, :, None]
                    & (neighbors == excluded_target[:, None, None]) & counts.gt(0))
        adjusted = counts - own_edge.long()
        size = len(self.catalog_tokens)
        order = (adjusted * (size + 1) + size - neighbors).argsort(-1, descending=True)
        order = order[..., :self.config.neighbors]
        return neighbors.gather(-1, order), adjusted.gather(-1, order).float().log1p()

    def candidate_gate(self, state, indices, vectors, channel):
        projected = state * self.memory_scale.weight[channel]
        logits = projected.sum(-1, keepdim=True) + self.memory_scale.bias[channel]
        logits = logits + math.sqrt(state.shape[-1]) * torch.einsum(
            "brd,bnd->brn", projected, vectors[indices])
        if self.config.correction == "candidate_nonnegative":
            return F.softplus(logits)
        return 0.5 * logits + math.log(2.0)

    def score_components(self, ids, excluded_target=None):
        user, state, attention = self.readout_evidence(ids)
        vectors = self.item_vectors()
        dense = (user.flatten(0, 1) @ vectors.T / self.config.temperature).reshape(
            len(ids), self.config.queries, len(vectors))
        neighbors, strength = self.transition_table(ids, excluded_target)
        history = self.item_to_catalog[ids].clamp_min(0)
        transition = neighbors.flatten(1)
        indices = torch.cat((history, transition), -1)
        if self.config.correction == "nonnegative":
            scale = F.softplus(self.memory_scale(state))
            direct = attention * scale[:, :, :1]
            routed = (attention[:, :, :, None] * scale[:, :, 1:2, None]
                      * strength[:, None]).flatten(2)
        else:
            direct = attention * self.candidate_gate(state, history, vectors, 0)
            routed = ((attention[:, :, :, None] * strength[:, None]).flatten(2)
                      * self.candidate_gate(state, transition, vectors, 1))
        values = torch.cat((direct, routed), -1)
        corrected = dense.scatter_add(2, indices[:, None].expand(-1, self.config.queries, -1), values)
        return dense, corrected, indices, values, attention

    def scores(self, ids):
        return self.score_components(ids)[1].amax(1)

    def forward(self, batch):
        target = self.item_to_catalog[batch["labels"].reshape(-1)]
        scores = self.score_components(batch["input_ids"], target if self.training else None)[1].amax(1)
        loss = F.cross_entropy(scores, target)
        return Output(loss, {"catalog_nll": loss.detach()})

    def exact_union_topk(self, ids, k=10):
        """Exact union verification for the default single-query nonnegative model.

        Semantic catalog scoring is dense; this is not an ANN implementation.
        """
        if self.config.queries != 1 or self.config.correction == "candidate_signed":
            raise ValueError("This union guarantee requires one nonnegative query")
        dense, corrected, indices, _, _ = self.score_components(ids)
        candidates = torch.cat((dense[:, 0].topk(k, -1).indices, indices), -1)
        ordered = candidates.sort(-1).values
        scores = corrected[:, 0].gather(1, ordered)
        keep = torch.ones_like(ordered, dtype=torch.bool)
        keep[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        selected = scores.masked_fill(~keep, -float("inf")).topk(k, -1).indices
        return ordered.gather(1, selected)
