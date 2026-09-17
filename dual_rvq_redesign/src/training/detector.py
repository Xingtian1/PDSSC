from __future__ import annotations

import torch
from torch import nn


class ParentConsistencyDetector(nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        confidence: torch.Tensor,
        consistency: torch.Tensor,
        parent_similarity: torch.Tensor,
        code_norm: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack([confidence, consistency, parent_similarity, code_norm], dim=-1)
        return self.net(x).squeeze(-1)
