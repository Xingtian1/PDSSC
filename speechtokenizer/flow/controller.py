import torch
import torch.nn as nn


class BitrateController(nn.Module):
    """
    Tiny channel-adaptive transmission-depth selector.

    Input:
      - p_loss: packet loss rate in [0, 1]
      - r_max: target bitrate budget
      - obs_feat: lightweight statistics from the observed partial latent

    Output:
      - a score for each candidate transmission depth N
    """

    def __init__(self, num_actions: int, hidden_dim: int = 64, in_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(
        self,
        p_loss: torch.Tensor,
        r_max: torch.Tensor | None = None,
        obs_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if p_loss.dim() == 1:
            p_loss = p_loss.unsqueeze(-1)
        if r_max is None:
            r_max = torch.zeros_like(p_loss)
        elif r_max.dim() == 1:
            r_max = r_max.unsqueeze(-1)
        feats = [p_loss, r_max]
        if obs_feat is not None:
            if obs_feat.dim() == 1:
                obs_feat = obs_feat.unsqueeze(0)
            feats.append(obs_feat)
        x = torch.cat(feats, dim=-1)
        return self.net(x)
