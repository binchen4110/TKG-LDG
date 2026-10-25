from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict

import dgl
import numpy as np
import torch


@dataclass(frozen=True)
class Snapshot:
    time: int
    triples: np.ndarray


def read_split(path, num_entities, num_relations):

    rows = np.loadtxt(path, dtype=np.int64, ndmin=2, usecols=(0, 1, 2, 3))
    if not len(rows):
        raise ValueError(f"Empty split: {path}")
    if (
        (rows[:, :3] < 0).any()
        or rows[:, [0, 2]].max() >= num_entities
        or rows[:, 1].max() >= num_relations
    ):
        raise ValueError(f"Entity/relation IDs outside stat.txt vocabulary: {path}")
    rows = rows[np.argsort(rows[:, 3], kind="stable")]
    times, starts = np.unique(rows[:, 3], return_index=True)
    return [
        Snapshot(int(t), block[:, :3].copy())
        for t, block in zip(times, np.split(rows, starts[1:]))
    ]


def load_dataset(root, name):
    path = Path(root) / name
    stats = [int(x) for x in (path / "stat.txt").read_text().split()]
    ne, nr = stats[:2]
    splits = {
        s: read_split(path / f"{s}.txt", ne, nr) for s in ("train", "valid", "test")
    }
    if not (
        splits["train"][-1].time < splits["valid"][0].time
        and splits["valid"][-1].time < splits["test"][0].time
    ):
        raise ValueError(
            "Split timestamps overlap: chronological extrapolation requires train < valid < test"
        )
    return ne, nr, splits


def history_windows(history, long_window, short_window):
    """Split observed history into disjoint long-term and short-term windows (§4.1–4.2)."""

    if long_window < 0 or short_window < 0:
        raise ValueError("Window sizes must be nonnegative")
    end = max(0, len(history) - short_window)
    return history[max(0, end - long_window) : end], history[end:]


def merge_triples(snapshots):
    """Construct the long-term unified graph as a union of historical facts (§4.1)."""

    if not snapshots:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.concatenate([s.triples for s in snapshots]), axis=0)


def augment_inverse(triples, num_relations):
    inverse = triples[:, [2, 1, 0]].clone()
    inverse[:, 1] += num_relations
    return torch.cat((triples, inverse))


@dataclass
class GraphInput:
    graph: object
    pool_relations: torch.Tensor
    pool_entities: torch.Tensor

    def to(self, device):
        return GraphInput(
            self.graph.to(device),
            self.pool_relations.to(device),
            self.pool_entities.to(device),
        )


def build_graph(triples, num_entities, num_relations):
    """Add inverse edges and collect each relation's unique endpoints for pooling (Eq. 5)."""

    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    s, r, o = triples.T
    graph = dgl.graph(
        (np.concatenate((s, o)), np.concatenate((o, s))), num_nodes=num_entities
    )
    graph.edata["type"] = torch.from_numpy(np.concatenate((r, r + num_relations)))
    graph.ndata["norm"] = (
        graph.in_degrees().clamp(min=1).float().reciprocal().unsqueeze(1)
    )

    pairs = np.unique(
        np.concatenate((np.stack((r, s), 1), np.stack((r, o), 1))), axis=0
    )
    inverse_pairs = pairs.copy()
    inverse_pairs[:, 0] += num_relations
    pairs = np.concatenate((pairs, inverse_pairs))
    return GraphInput(
        graph, torch.from_numpy(pairs[:, 0]), torch.from_numpy(pairs[:, 1])
    )


class HistoryGraphBuilder:

    def __init__(
        self, num_entities, num_relations, long_window, short_window, cache_size=64
    ):
        self.ne, self.nr = num_entities, num_relations
        self.long_window, self.short_window = long_window, short_window
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def __call__(self, history, device):
        old, recent = history_windows(history, self.long_window, self.short_window)
        long_graph = (
            build_graph(merge_triples(old), self.ne, self.nr).to(device)
            if old
            else None
        )
        short_graphs = []
        for snapshot in recent:

            key = (snapshot.time, id(snapshot))
            if key not in self.cache:
                self.cache[key] = build_graph(snapshot.triples, self.ne, self.nr)
            self.cache.move_to_end(key)
            short_graphs.append(self.cache[key].to(device))
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return long_graph, short_graphs
