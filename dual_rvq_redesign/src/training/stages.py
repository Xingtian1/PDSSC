from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StageConfig:
    name: str
    train_encoder: bool
    train_decoder: bool
    train_fine_codebooks: bool
    train_parent_codebooks: bool
    train_flow: bool
    inject_channel_errors: bool


def default_stage_configs() -> list[StageConfig]:
    return [
        StageConfig(
            name="stage1_ed_fine",
            train_encoder=True,
            train_decoder=True,
            train_fine_codebooks=True,
            train_parent_codebooks=False,
            train_flow=False,
            inject_channel_errors=False,
        ),
        StageConfig(
            name="stage2_channel_parent_flow",
            train_encoder=False,
            train_decoder=False,
            train_fine_codebooks=False,
            train_parent_codebooks=True,
            train_flow=True,
            inject_channel_errors=True,
        ),
    ]
