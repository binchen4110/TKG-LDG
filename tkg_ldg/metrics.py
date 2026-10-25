from collections import defaultdict
import torch


def answers_for_timestamp(triples):
    answers = defaultdict(set)
    for s, r, o in triples.tolist():
        answers[(s, r)].add(o)
    return answers


def ranks(scores, targets):

    target_scores = scores.gather(1, targets[:, None])
    return (
        1
        + (scores > target_scores).sum(1).float()
        + ((scores == target_scores).sum(1).float() - 1) / 2
    )


def raw_and_filtered(scores, triples, answers):
    raw = ranks(scores, triples[:, 2])
    filtered = scores.clone()
    for i, (s, r, o) in enumerate(triples.tolist()):
        others = list(answers[(s, r)] - {o})
        if others:
            filtered[i, others] = -torch.inf
    return raw, ranks(filtered, triples[:, 2])


class RankingMetrics:
    def __init__(self):
        self.count = 0
        self.totals = torch.zeros(4, dtype=torch.float64)

    def add(self, values):
        values = values.detach().cpu().double()
        self.count += values.numel()
        self.totals += torch.stack(
            (
                values.reciprocal().sum(),
                (values <= 1).sum(),
                (values <= 3).sum(),
                (values <= 10).sum(),
            )
        )

    def compute(self):
        if not self.count:
            raise ValueError("Cannot evaluate an empty query set")
        return dict(
            zip(
                ("mrr", "hits@1", "hits@3", "hits@10"),
                (self.totals / self.count).tolist(),
            ),
            count=self.count,
        )
