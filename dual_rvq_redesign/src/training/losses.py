from __future__ import annotations

import torch
import torch.nn.functional as F

from speechtokenizer.trainer.loss import (
    adversarial_loss,
    d_axis_distill_loss,
    discriminator_loss,
    feature_loss,
    mel_loss,
    recon_loss,
)


def waveform_l1(wav_hat: torch.Tensor, wav_ref: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(wav_hat, wav_ref)


def branch_energy_balance_loss(semantic_quantized: torch.Tensor, acoustic_quantized: torch.Tensor) -> torch.Tensor:
    sem_energy = semantic_quantized.pow(2).mean()
    ac_energy = acoustic_quantized.pow(2).mean()
    return (sem_energy - ac_energy).abs()


def parent_consistency_loss(consistency_score: torch.Tensor) -> torch.Tensor:
    return 1.0 - consistency_score.mean()


def confidence_penalty(confidence: torch.Tensor) -> torch.Tensor:
    return 1.0 / (confidence.mean() + 1e-6)


__all__ = [
    "adversarial_loss",
    "branch_energy_balance_loss",
    "confidence_penalty",
    "d_axis_distill_loss",
    "discriminator_loss",
    "feature_loss",
    "mel_loss",
    "parent_consistency_loss",
    "recon_loss",
    "waveform_l1",
]
