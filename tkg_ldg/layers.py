import dgl.function as fn
import torch
from torch import nn
from torch.nn import functional as F


class UnionRGCNLayer(nn.Module):
    """Relation-aware message composition, normalized aggregation, and self loops (Eqs. 1, 3)."""

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.weight_neighbor = nn.Parameter(torch.empty(dim, dim))
        self.loop_weight = nn.Parameter(torch.empty(dim, dim))
        self.evolve_loop_weight = nn.Parameter(torch.empty(dim, dim))
        for parameter in (
            self.weight_neighbor,
            self.loop_weight,
            self.evolve_loop_weight,
        ):
            nn.init.xavier_uniform_(parameter, gain=nn.init.calculate_gain("relu"))
        self.conv1d = nn.Conv1d(2, 1, 1)
        self.gnn_bn0 = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, entities, relations):

        with graph.local_scope():
            active = graph.in_degrees() > 0
            loops = torch.where(
                active[:, None],
                entities @ self.loop_weight,
                entities @ self.evolve_loop_weight,
            )
            if graph.num_edges():
                graph.ndata["h"] = entities

                def message(edges):
                    stacked = torch.stack(
                        (edges.src["h"], relations[edges.data["type"]]), dim=1
                    )
                    composed = self.conv1d(self.gnn_bn0(stacked)).squeeze(1)
                    return {"msg": composed @ self.weight_neighbor}

                graph.update_all(message, fn.sum("msg", "aggregate"))
                neighbors = graph.ndata["aggregate"] * graph.ndata["norm"]
            else:
                neighbors = torch.zeros_like(entities)

            return self.dropout(F.rrelu(neighbors + loops, training=self.training))


class RGCNCell(nn.Module):
    def __init__(self, dim, layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList(
            [UnionRGCNLayer(dim, dropout) for _ in range(layers)]
        )

    def forward(self, graph, entities, relations):
        for layer in self.layers:
            entities = layer(graph, entities, relations)
        return entities
