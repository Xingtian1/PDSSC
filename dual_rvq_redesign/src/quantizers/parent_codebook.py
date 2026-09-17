from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ParentCodebook(nn.Module):
    def __init__(self, num_centroids: int, dim: int, normalize: bool = True):
        super().__init__()
        self.num_centroids = num_centroids
        self.dim = dim
        self.normalize = normalize
        self.centroids = nn.Parameter(torch.randn(num_centroids, dim) * 0.02)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        return F.normalize(x, dim=-1)

    def assign(self, features: torch.Tensor) -> torch.Tensor:
        x = features.permute(0, 2, 1)
        x = self._normalize(x)
        c = self._normalize(self.centroids)
        scores = torch.matmul(x, c.t())
        return scores.argmax(dim=-1)

    def decode(self, parent_ids: torch.Tensor) -> torch.Tensor:
        emb = F.embedding(parent_ids, self._normalize(self.centroids))
        return emb.permute(0, 2, 1).contiguous()

    def consistency_score(self, features: torch.Tensor, parent_ids: torch.Tensor) -> torch.Tensor:
        x = self._normalize(features.permute(0, 2, 1))
        p = self._normalize(F.embedding(parent_ids, self.centroids))
        return (x * p).sum(dim=-1)
