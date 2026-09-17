from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from speechtokenizer.modules.seanet import SEANetDecoder, SEANetEncoder

from ..config import ModelConfig
from ..quantizers.parent_codebook import ParentCodebook
from ..quantizers.rvq_branch import RVQBranch


class DualRVQModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.encoder = SEANetEncoder(
            n_filters=config.n_filters,
            dimension=config.latent_dim,
            ratios=list(config.strides),
            lstm=config.lstm_layers,
            bidirectional=config.bidirectional,
            dilation_base=config.dilation_base,
            residual_kernel_size=config.residual_kernel_size,
            n_residual_layers=config.n_residual_layers,
            activation=config.activation,
        )
        self.decoder = SEANetDecoder(
            n_filters=config.n_filters,
            dimension=config.latent_dim,
            ratios=list(config.strides),
            lstm=config.lstm_layers,
            bidirectional=False,
            dilation_base=config.dilation_base,
            residual_kernel_size=config.residual_kernel_size,
            n_residual_layers=config.n_residual_layers,
            activation=config.activation,
        )

        self.semantic_proj = nn.Conv1d(config.latent_dim, config.semantic.branch_dim, kernel_size=1)
        self.acoustic_proj = nn.Conv1d(config.latent_dim, config.acoustic.branch_dim, kernel_size=1)
        self.semantic_fuse = nn.Conv1d(config.semantic.branch_dim, config.latent_dim, kernel_size=1)
        self.acoustic_fuse = nn.Conv1d(config.acoustic.branch_dim, config.latent_dim, kernel_size=1)

        self.semantic_branch = RVQBranch(config.semantic.branch_dim, config.semantic.n_q, config.semantic.codebook_sizes)
        self.acoustic_branch = RVQBranch(config.acoustic.branch_dim, config.acoustic.n_q, config.acoustic.codebook_sizes)

        self.semantic_parents = nn.ModuleList(
            [ParentCodebook(num_centroids, config.semantic.branch_dim) for num_centroids in config.semantic.parent_codebook_sizes]
        )
        self.acoustic_parents = nn.ModuleList(
            [ParentCodebook(num_centroids, config.acoustic.branch_dim) for num_centroids in config.acoustic.parent_codebook_sizes]
        )

    @staticmethod
    def _assign_parent_stack(layer_quantized: torch.Tensor, parents: nn.ModuleList) -> tuple[torch.Tensor, torch.Tensor]:
        parent_ids = []
        parent_latents = []
        for idx, parent in enumerate(parents):
            layer_feature = layer_quantized[idx].detach()
            ids = parent.assign(layer_feature)
            parent_ids.append(ids)
            parent_latents.append(parent.decode(ids))
        return torch.stack(parent_ids, dim=0), torch.stack(parent_latents, dim=0)

    @staticmethod
    def _sum_parent_fallback(
        branch_codes: torch.Tensor,
        branch_decode_fn,
        parent_ids: torch.Tensor | None,
        parent_mask: torch.Tensor | None,
        parents: nn.ModuleList,
    ) -> torch.Tensor:
        branch_layers = []
        num_layers = branch_codes.shape[0]
        for idx in range(num_layers):
            layer_codes = branch_codes[idx : idx + 1]
            layer_quantized = branch_decode_fn(layer_codes, st=idx)
            if parent_ids is not None and parent_mask is not None:
                parent_layer = parents[idx].decode(parent_ids[idx])
                layer_quantized = torch.where(parent_mask[idx][:, None, :], parent_layer, layer_quantized)
            branch_layers.append(layer_quantized)
        return torch.stack(branch_layers, dim=0).sum(dim=0)

    def encode_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h_sem = self.semantic_proj(h)
        h_ac = self.acoustic_proj(h)
        return h, h_sem, h_ac

    def decode_latent(self, q_sem: torch.Tensor, q_ac: torch.Tensor) -> torch.Tensor:
        fused = self.semantic_fuse(q_sem) + self.acoustic_fuse(q_ac)
        return self.decoder(fused)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, h_sem, h_ac = self.encode_latent(x)

        sem_out = self.semantic_branch(h_sem)
        ac_out = self.acoustic_branch(h_ac)

        sem_parent_ids, semantic_parent_latent = self._assign_parent_stack(sem_out["layer_quantized"], self.semantic_parents)
        ac_parent_ids, acoustic_parent_latent = self._assign_parent_stack(ac_out["layer_quantized"], self.acoustic_parents)

        wav_hat = self.decode_latent(sem_out["quantized"], ac_out["quantized"])

        return {
            "wav_hat": wav_hat,
            "semantic_quantized": sem_out["quantized"],
            "semantic_feature": sem_out["quantized"].permute(0, 2, 1),
            "semantic_codes": sem_out["codes"],
            "semantic_layer_quantized": sem_out["layer_quantized"],
            "semantic_commitment": sem_out["commitment"],
            "semantic_parent_ids": sem_parent_ids,
            "semantic_parent_latent": semantic_parent_latent,
            "acoustic_quantized": ac_out["quantized"],
            "acoustic_codes": ac_out["codes"],
            "acoustic_layer_quantized": ac_out["layer_quantized"],
            "acoustic_commitment": ac_out["commitment"],
            "acoustic_parent_ids": ac_parent_ids,
            "acoustic_parent_latent": acoustic_parent_latent,
            "full_latent": self.semantic_fuse(sem_out["quantized"]) + self.acoustic_fuse(ac_out["quantized"]),
            "commitment": sem_out["commitment"] + ac_out["commitment"],
        }

    @torch.no_grad()
    def estimate_bitrate_kbps(self) -> float:
        return self.config.target_bitrate_kbps

    def reconstruct_with_parent_fallback(
        self,
        semantic_codes: torch.Tensor,
        acoustic_codes: torch.Tensor,
        semantic_parent_ids: torch.Tensor | None = None,
        acoustic_parent_ids: torch.Tensor | None = None,
        use_semantic_parent: bool = False,
        use_acoustic_parent: bool = False,
    ) -> torch.Tensor:
        q_sem = self.semantic_branch.decode(semantic_codes)
        q_ac = self.acoustic_branch.decode(acoustic_codes)
        if use_semantic_parent:
            if semantic_parent_ids is None:
                raise ValueError("semantic_parent_ids is required when semantic parent fallback is enabled")
            q_sem = torch.stack(
                [self.semantic_parents[idx].decode(semantic_parent_ids[idx]) for idx in range(len(self.semantic_parents))],
                dim=0,
            ).sum(dim=0)
        if use_acoustic_parent:
            if acoustic_parent_ids is None:
                raise ValueError("acoustic_parent_ids is required when acoustic parent fallback is enabled")
            q_ac = torch.stack(
                [self.acoustic_parents[idx].decode(acoustic_parent_ids[idx]) for idx in range(len(self.acoustic_parents))],
                dim=0,
            ).sum(dim=0)
        return self.decode_latent(q_sem, q_ac)

    def reconstruct_latent_with_parent_fallback(
        self,
        semantic_codes: torch.Tensor,
        acoustic_codes: torch.Tensor,
        semantic_parent_ids: torch.Tensor | None = None,
        acoustic_parent_ids: torch.Tensor | None = None,
        semantic_parent_mask: torch.Tensor | None = None,
        acoustic_parent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_sem = self._sum_parent_fallback(
            semantic_codes,
            self.semantic_branch.decode,
            semantic_parent_ids,
            semantic_parent_mask,
            self.semantic_parents,
        )
        q_ac = self._sum_parent_fallback(
            acoustic_codes,
            self.acoustic_branch.decode,
            acoustic_parent_ids,
            acoustic_parent_mask,
            self.acoustic_parents,
        )
        fused = self.semantic_fuse(q_sem) + self.acoustic_fuse(q_ac)
        return q_sem, q_ac, fused

    @staticmethod
    def confidence_to_parent_mask(confidence: torch.Tensor, threshold: float) -> torch.Tensor:
        if confidence.dim() == 3:
            confidence = confidence.mean(dim=0)
        if confidence.dim() == 2:
            confidence = confidence.mean(dim=0)
        return confidence < threshold

    def extra_repr(self) -> str:
        return str(asdict(self.config))
