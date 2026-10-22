"""Exact four-code SID relevance over the complete distinct-SID catalog."""
import torch


def ranking_metrics(predicted, labels):
    if predicted.ndim != 3 or labels.ndim != 2 or predicted.shape[0] != len(labels):
        raise ValueError("Expected [users,K,digits] predictions and [users,digits] labels")
    hits = predicted.eq(labels[:, None]).all(-1)
    if hits.sum(-1).gt(1).any():
        raise ValueError("Duplicate target SID in ranked output")
    discount = torch.arange(2, predicted.shape[1] + 2, dtype=torch.float64,
                            device=predicted.device).log2().reciprocal()
    output = {}
    for k in (1, 5, 10):
        if k <= predicted.shape[1]:
            output[f"recall@{k}"] = hits[:, :k].any(-1).double().mean().item()
            output[f"ndcg@{k}"] = (hits[:, :k] * discount[:k]).sum(-1).mean().item()
    if "ndcg@10" in output:
        output["score@10"] = 0.8 * output["ndcg@10"] + 0.2 * output["recall@10"]
    return output


@torch.inference_mode()
def evaluate(model, data, batch_size=256):
    model.eval()
    if len(model.catalog_tokens) < 10:
        raise ValueError("Evaluation at K=10 needs at least ten distinct SIDs")
    predictions = []
    for ids in data["input_ids"].split(batch_size):
        indices = model.scores(ids).topk(10, -1).indices
        predictions.append(model.catalog_tokens[indices].cpu())
    predictions = torch.cat(predictions)
    return ranking_metrics(predictions, data["labels"].cpu()), predictions
