"""CPU checks for exact exclusion, score support, persistence, and data splits."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io
from unittest.mock import patch
import numpy as np
import torch
from relay import ModelConfig, RELAY
from relay.checkpoint import load_model, pack, save_checkpoint
from relay.data import load_split, load_tokens, prepare_sequences, save_split
from relay.memory import build_memory
from relay.metrics import ranking_metrics


def fixture(correction="nonnegative", queries=1):
    config = ModelConfig(dim=16, heads=4, inner=32, dropout=0, max_length=4,
                         neighbors=2, vocab_size=32, correction=correction, queries=queries)
    tokens = torch.zeros(13, 2, dtype=torch.long)
    tokens[1:, 0] = torch.arange(1, 13)
    tokens[1:, 1] = torch.arange(13, 25)
    histories = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1],
                             [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 2]])
    targets = torch.tensor([[2], [3], [4], [1], [1], [2]])
    graph = build_memory(tokens, histories, targets, config.neighbors)
    return RELAY(config, tokens, graph), histories, targets


class RelayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)
        torch.set_num_threads(2)

    def test_exclusion_matches_rebuilding_counts_and_ties(self):
        model, histories, targets = fixture()
        target_classes = model.item_to_catalog[targets.flatten()]
        ids, strengths = model.transition_table(histories, target_classes)
        # Every occurrence of the last source receives the same adjusted row.
        all_counts = [Counter() for _ in model.catalog_tokens]
        for history, target in zip(histories, targets.flatten()):
            last = int(model.item_to_catalog[history[history != 0][-1]])
            dest = int(model.item_to_catalog[target])
            if last != dest:
                all_counts[last][dest] += 1
        for i, history in enumerate(histories):
            last = int(model.item_to_catalog[history[history != 0][-1]])
            target = int(target_classes[i])
            for j, item in enumerate(history):
                if item == 0:
                    continue
                source = int(model.item_to_catalog[item])
                counts = all_counts[source].copy()
                if source == last and source != target:
                    counts[target] -= 1
                expected = sorted(((d, c) for d, c in counts.items() if c > 0),
                                  key=lambda entry: (-entry[1], entry[0]))[:2]
                actual = [(int(d), int(c)) for d, c in zip(ids[i, j], strengths[i, j].expm1().round().long()) if c > 0]
                self.assertEqual(actual, expected)

    def test_nonnegative_control_and_union(self):
        model, histories, _ = fixture("nonnegative")
        model.eval()
        with torch.inference_mode():
            dense, corrected, _, values, attention = model.score_components(histories)
            self.assertTrue((values >= 0).all())
            self.assertTrue((corrected >= dense).all())
            self.assertTrue((attention.masked_select(histories.eq(0)[:, None]) == 0).all())
            torch.testing.assert_close(attention.sum(-1), torch.ones(len(histories), 1))
            # Compare scores rather than arbitrary tie order for equal-score entries.
            union = model.exact_union_topk(histories, 5)
            torch.testing.assert_close(corrected[:, 0].gather(1, union), corrected[:, 0].topk(5).values)

    def test_default_uses_candidate_conditioned_signed_correction(self):
        self.assertEqual(ModelConfig().correction, "candidate_signed")
        model, histories, _ = fixture("candidate_signed")
        self.assertEqual(model.config.correction, "candidate_signed")
        self.assertEqual(model.config.queries, 1)
        with self.assertRaises(ValueError):
            ModelConfig(queries=2, correction="candidate_signed")

    def test_signed_pair_matches_initial_loss_and_gradients(self):
        positive, histories, targets = fixture("candidate_nonnegative")
        signed, _, _ = fixture("candidate_signed")
        signed.load_state_dict(positive.state_dict())
        batch = dict(input_ids=histories, labels=targets)
        p, s = positive(batch).loss, signed(batch).loss
        p.backward(); s.backward()
        torch.testing.assert_close(p, s)
        for left, right in zip(positive.parameters(), signed.parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
        with torch.no_grad():
            signed.memory_scale.weight.normal_()
            values = signed.score_components(histories)[3]
            self.assertTrue((values < 0).any() and (values > 0).any())
        with self.assertRaises(ValueError):
            signed.exact_union_topk(histories)

    def test_all_queries_receive_gradients(self):
        model, histories, targets = fixture(queries=4)
        model(dict(input_ids=histories, labels=targets)).loss.backward()
        self.assertTrue((model.queries.grad.abs().sum(-1) > 0).all())

    def test_checkpoint_roundtrip_and_cache_reset(self):
        model, histories, _ = fixture()
        model.eval()
        with torch.inference_mode():
            expected = model.scores(histories)
            self.assertIsNotNone(model._catalog_cache)
        model.train()
        self.assertIsNone(model._catalog_cache)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, pack(model, seed=2024))
            other, artifact = load_model(path)
            with torch.inference_mode():
                torch.testing.assert_close(other.eval().scores(histories), expected, rtol=0, atol=0)

    def test_chronological_split_and_sid_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "seq.json").write_text(json.dumps({"u": ["a", "b", "c", "d", "e"]}))
            (root / "sid.json").write_text(json.dumps({"a": [0, 0], "b": [1, 1], "c": [1, 1], "d": [2, 2], "e": [3, 3]}))
            prepare_sequences(root / "seq.json", root / "sid.json", root / "data", 4, 8)
            train, val, test = [load_split(root / "data", split) for split in ("train", "val", "test")]
            self.assertEqual(train["labels"].flatten().tolist(), [2, 3])
            self.assertEqual(val["input_ids"].tolist(), [[0, 1, 2, 3]])
            self.assertEqual(test["input_ids"].tolist(), [[1, 2, 3, 4]])
            tokens = load_tokens(root / "data")
            torch.testing.assert_close(val["labels"], tokens[[4]])
            from relay.memory import catalog_from_tokens
            catalog, mapping = catalog_from_tokens(tokens)
            self.assertEqual(len(catalog), 4)
            self.assertEqual(mapping[2], mapping[3])

    def test_metrics(self):
        labels = torch.tensor([[1, 2], [3, 4]])
        predicted = torch.arange(40).reshape(2, 10, 2) + 10
        predicted[0, 0], predicted[1, 4] = labels[0], labels[1]
        result = ranking_metrics(predicted, labels)
        self.assertEqual(result["recall@1"], 0.5)
        self.assertEqual(result["recall@5"], 1.0)
        self.assertAlmostEqual(result["ndcg@5"], (1 + 1 / torch.tensor(6.).log2().item()) / 2)

    def test_resume_restores_optimizer_dropout_and_data_rng_without_test(self):
        from relay.config import TrainConfig
        from relay.training import train
        from relay.metrics import evaluate
        model, histories, targets = fixture()
        model.config.dropout = 0.1
        training = TrainConfig(epochs=3, patience=10, batch_size=2,
            eval_batch_size=3, learning_rate=0.001, warmup_steps=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"; data.mkdir()
            np.save(data / "item_tokens.npy", model.item_tokens.numpy())
            save_split(data / "train.npz", histories, targets)
            save_split(data / "val.npz", histories, model.item_tokens[targets.flatten()])
            calls = 0
            def interrupt_after_first_epoch(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise InterruptedError("Simulated interruption before the second checkpoint")
                return evaluate(*args, **kwargs)
            with redirect_stdout(io.StringIO()):
                train(data, root / "complete", model.config, training)
                with patch("relay.training.evaluate", interrupt_after_first_epoch):
                    with self.assertRaises(InterruptedError):
                        train(data, root / "resumed", model.config, training)
                train(data, root / "resumed", model.config, training, resume=True)
            complete, _ = load_model(root / "complete" / "last.pt")
            resumed, _ = load_model(root / "resumed" / "last.pt")
            for key, value in complete.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            self.assertFalse((data / "test.npz").exists())
            self.assertEqual(len((root / "resumed" / "history.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
