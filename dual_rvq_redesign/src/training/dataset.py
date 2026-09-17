from __future__ import annotations

import os
import torch
from torch.utils.data import DataLoader

from speechtokenizer.flow.dataset import LibriSpeechFlowDataset
from speechtokenizer.trainer.dataset import audioDataset, get_dataloader


def collate_waveforms(batch):
    wavs = torch.stack([item[0] for item in batch], dim=0)
    ref_wavs = torch.stack([item[1] for item in batch], dim=0)
    return wavs, ref_wavs


def build_librispeech_loader(
    data_dir: str,
    split: str,
    segment_sec: float,
    sample_rate: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = LibriSpeechFlowDataset(
        data_dir=data_dir,
        split=split,
        segment_sec=segment_sec,
        sample_rate=sample_rate,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_waveforms,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def build_codec_feature_loader(
    train_files: str,
    segment_size: int,
    sample_rate: int,
    downsample_rate: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
):
    dataset = audioDataset(
        file_list=open(train_files, encoding="utf-8").readlines(),
        segment_size=segment_size,
        sample_rate=sample_rate,
        downsample_rate=downsample_rate,
        valid=False,
    )
    return get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
    )
