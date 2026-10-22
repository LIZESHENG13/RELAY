"""Warmed-up eager or CUDA Graph timing for complete evaluation passes."""
import statistics
import time
import torch


class GraphRunner:
    def __init__(self, function, example):
        current = torch.cuda.current_stream(example.device)
        side = torch.cuda.Stream(device=example.device)
        self.input = example.clone()
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(5):
                function(self.input)
        current.wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side):
            self.output = function(self.input)
        current.wait_stream(side)

    def __call__(self, ids):
        self.input.copy_(ids)
        self.graph.replay()
        return self.output.clone()


@torch.inference_mode()
def benchmark(model, data, batch_size=256, repeats=6, cuda_graph=False):
    model.eval()
    if repeats < 1 or batch_size < 1:
        raise ValueError("Positive repeats and batch size required")
    device = next(model.parameters()).device
    if cuda_graph and device.type != "cuda":
        raise ValueError("CUDA Graph timing requires a CUDA device")
    if len(model.catalog_tokens) < 10:
        raise ValueError("Top-10 benchmark requires at least ten distinct SIDs")
    batches = list(data["input_ids"].split(batch_size))
    model.item_vectors()
    graphs = {}
    def native(ids):
        return model.catalog_tokens[model.scores(ids).topk(10, -1).indices]
    def function(ids):
        if not cuda_graph:
            return native(ids)
        shape = tuple(ids.shape)
        if shape not in graphs:
            graphs[shape] = GraphRunner(native, ids)
        return graphs[shape](ids)
    for ids in batches:
        actual = function(ids)
        if cuda_graph and not torch.equal(actual, native(ids)):
            raise AssertionError("CUDA Graph Top-10 differs from eager")
    if cuda_graph:
        owned = function(batches[0]); saved = owned.clone()
        changed = batches[0].roll(1, 0).contiguous()
        if not torch.equal(function(changed), native(changed)) or not torch.equal(owned, saved):
            raise AssertionError("CUDA Graph input/output ownership check failed")
    for _ in range(4):
        for ids in batches:
            function(ids)
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    values = []
    for _ in range(repeats):
        synchronize(); started = time.perf_counter()
        for ids in batches:
            function(ids)
        synchronize(); values.append(time.perf_counter() - started)
    return dict(users=len(data["input_ids"]), batch_size=batch_size, repeats=repeats,
        full_pass_seconds=values, median_full_pass_seconds=statistics.median(values),
        parameters=sum(p.numel() for p in model.parameters()),
        execution="cuda_graph" if cuda_graph else "eager", device=str(device),
        pytorch=str(torch.__version__), precision=str(next(model.parameters()).dtype),
        cuda_graph_verified=cuda_graph,
        scope="GPU/CPU-resident histories, cached catalog vectors, encoding, full-catalog scoring, memory correction, and Top-10 SID lookup; excludes loading, transfers, preparation, and warmup.")
