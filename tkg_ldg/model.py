from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import ConvTransDecoder
from .layers import RGCNCell


@dataclass
class ModelConfig:
    dim: int = 200
    long_layers: int = 2
    short_layers: int = 2
    dropout: float = 0.2
    channels: int = 50
    kernel_size: int = 3
    normalize: bool = True


class EvolutionBlock(nn.Module):
    """Update relation and entity states using graph structure and dual GRUs (Eqs. 3–6)."""

    def __init__(self, config, layers):
        super().__init__()
        self.normalize = config.normalize
        self.rgcn = RGCNCell(config.dim, layers, config.dropout)
        self.relation_cell = nn.GRUCell(2 * config.dim, config.dim)
        self.entity_cell = nn.GRUCell(config.dim, config.dim)

    def norm(self, x):
        return F.normalize(x, dim=-1) if self.normalize else x

    def forward(self, graph_input, entities, relations, base_relations):
        # Relation evolution: mean-pool related entities and update the relation GRU.
        pooled = entities.new_zeros(relations.shape)
        counts = entities.new_zeros((len(relations), 1))
        rids, eids = graph_input.pool_relations, graph_input.pool_entities
        pooled.index_add_(0, rids, entities[eids])
        counts.index_add_(0, rids, entities.new_ones((len(rids), 1)))
        pooled = pooled / counts.clamp(min=1)

        relations = self.norm(
            self.relation_cell(torch.cat((base_relations, pooled), dim=1), relations)
        )
        # Entity evolution: aggregate graph messages, then update the entity GRU.
        convolved = self.norm(self.rgcn(graph_input.graph, entities, relations))
        entities = self.norm(self.entity_cell(convolved, entities))
        return entities, relations


class TKGLDG(nn.Module):
    """Long-term graph initialization, short-term evolution, and joint prediction."""

    def __init__(self, num_entities, num_relations, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.dim < 1 or min(c.long_layers, c.short_layers) < 1:
            raise ValueError("Invalid model dimensions or layer count")
        self.num_relations = num_relations
        self.dynamic_emb = nn.Parameter(torch.empty(num_entities, c.dim))
        self.emb_rel = nn.Parameter(torch.empty(2 * num_relations, c.dim))
        nn.init.normal_(self.dynamic_emb)
        nn.init.xavier_normal_(self.emb_rel)
        self.long_encoder = EvolutionBlock(c, c.long_layers)
        self.short_encoder = EvolutionBlock(c, c.short_layers)
        self.entity_decoder = ConvTransDecoder(
            c.dim, c.channels, c.kernel_size, c.dropout
        )
        self.relation_decoder = ConvTransDecoder(
            c.dim, c.channels, c.kernel_size, c.dropout
        )

    def encode(self, long_graph, short_graphs):
        entities = (
            F.normalize(self.dynamic_emb, dim=-1)
            if self.config.normalize
            else self.dynamic_emb
        )
        relations = self.emb_rel
        # Long-term module: encode the unified history graph to initialize short-term states.
        if long_graph is not None:
            entities, relations = self.long_encoder(
                long_graph, entities, relations, self.emb_rel
            )
        # Short-term module: evolve the initialized states through recent snapshots.
        for graph in short_graphs:
            entities, relations = self.short_encoder(
                graph, entities, relations, self.emb_rel
            )
        return entities, relations

    def scores(self, state, triples, task="entity"):
        entities, relations = state

        entities = torch.tanh(entities)
        if task == "entity":
            return self.entity_decoder(
                entities[triples[:, 0]], relations[triples[:, 1]], entities
            )
        if task == "relation":
            return self.relation_decoder(
                entities[triples[:, 0]], entities[triples[:, 2]], relations
            )
        raise ValueError(f"Unknown prediction task: {task}")

    def loss(self, state, triples, entity_weight=0.7):
        """Joint objective: λ times entity CE plus (1−λ) times relation CE (Eqs. 9–11)."""
        if not 0 <= entity_weight <= 1:
            raise ValueError("Entity weight must be in [0,1]")

        entity = F.cross_entropy(self.scores(state, triples), triples[:, 2])
        relation = F.cross_entropy(
            self.scores(state, triples, "relation"), triples[:, 1]
        )
        return entity_weight * entity + (1 - entity_weight) * relation, entity, relation
