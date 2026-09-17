from __future__ import annotations

import torch
from torch import nn

from speechtokenizer.quantization.core_vq import VectorQuantization


class RVQBranch(nn.Module):
    def __init__(self, branch_dim: int, n_q: int, codebook_sizes: tuple[int, ...]):
        super().__init__()
        if len(codebook_sizes) != n_q:
            raise ValueError(f"Expected {n_q} codebook sizes, got {len(codebook_sizes)}")
        self.branch_dim = branch_dim
        self.n_q = n_q
        self.codebook_sizes = tuple(codebook_sizes)
        self.layers = nn.ModuleList(
            [
                VectorQuantization(
                    dim=branch_dim,
                    codebook_size=codebook_size,
                    kmeans_init=True,
                    kmeans_iters=50,
                    decay=0.99,
                    threshold_ema_dead_code=2,
                    commitment_weight=1.0,
                )
                for codebook_size in self.codebook_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = x
        quantized_sum = torch.zeros_like(x)
        codes = []
        commitments = []
        layer_quantized = []

        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized
            codes.append(indices)
            commitments.append(loss)
            layer_quantized.append(quantized)

        return {
            "quantized": quantized_sum,
            "codes": torch.stack(codes, dim=0),
            "commitment": torch.stack(commitments).mean(),
            "layer_quantized": torch.stack(layer_quantized, dim=0),
        }

    def encode(self, x: torch.Tensor, st: int | None = None, n_q: int | None = None) -> torch.Tensor:
        start = st or 0
        end = n_q if n_q is not None else self.n_q
        residual = x
        out_codes = []
        for idx, layer in enumerate(self.layers):
            if idx < start:
                with torch.no_grad():
                    indices = layer.encode(residual)
                    quantized = layer.decode(indices)
                residual = residual - quantized
                continue
            if idx >= end:
                break
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            out_codes.append(indices)
        return torch.stack(out_codes, dim=0)

    def decode(self, codes: torch.Tensor, st: int = 0) -> torch.Tensor:
        quantized_sum = torch.zeros(
            codes.shape[1],
            self.branch_dim,
            codes.shape[2],
            device=codes.device,
        )
        for offset, indices in enumerate(codes):
            quantized_sum = quantized_sum + self.layers[st + offset].decode(indices)
        return quantized_sum
