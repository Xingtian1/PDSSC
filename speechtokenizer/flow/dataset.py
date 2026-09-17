"""
LibriSpeech 同说话人数据集，用于 Flow 模型训练。

LibriSpeech 目录结构:
  data/LibriSpeech/{split}/{speaker_id}/{chapter_id}/*.flac

每条数据返回:
  (wav, ref_wav) — 同说话人的两段音频，均裁剪/填充到 segment_sec 秒。
信道模拟在训练循环中在线完成（见 train_flow.py）。
"""

import os
import random
from collections import defaultdict

from typing import Optional

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LibriSpeechFlowDataset(Dataset):
    """
    参数
    ----
    data_dir    : 数据根目录（data/LibriSpeech/{split} 在此目录下）
    split       : 数据集分割，默认 'train-clean-100'
    segment_sec : 裁剪长度（秒），训练时固定长度便于批次化
    sample_rate : 采样率，默认 16000
    min_dur_sec : 跳过短于此时长的文件
    """

    def __init__(
        self,
        data_dir    : str,
        split       : str   = "train-clean-100",
        segment_sec : float = 4.0,
        sample_rate : int   = 16000,
        min_dur_sec : float = 2.0,
    ):
        self.segment_len = int(segment_sec * sample_rate)
        self.sample_rate = sample_rate
        self.min_samples = int(min_dur_sec * sample_rate)

        # 构建 speaker → [file_path, ...] 映射
        split_dir  = os.path.join(data_dir, "LibriSpeech", split)
        if not os.path.isdir(split_dir):
            # 兼容 data_dir 直接是 split 目录的情况
            split_dir = os.path.join(data_dir, split)

        spk_files: dict = defaultdict(list)
        for spk_id in sorted(os.listdir(split_dir)):
            spk_path = os.path.join(split_dir, spk_id)
            if not os.path.isdir(spk_path):
                continue
            for chap_id in sorted(os.listdir(spk_path)):
                chap_path = os.path.join(spk_path, chap_id)
                if not os.path.isdir(chap_path):
                    continue
                for fname in sorted(os.listdir(chap_path)):
                    if fname.endswith(".flac") or fname.endswith(".wav"):
                        spk_files[spk_id].append(os.path.join(chap_path, fname))

        # 只保留有 ≥2 条音频的说话人（用于参考采样）
        self.spk_files = {k: v for k, v in spk_files.items() if len(v) >= 2}
        self.speakers  = sorted(self.spk_files.keys())

        # 展平为 (speaker_id, audio_path) 列表
        self.pairs = []
        for spk, files in self.spk_files.items():
            for path in files:
                self.pairs.append((spk, path))

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"未找到音频文件，请先下载 {split} 到 {split_dir}"
            )
        print(
            f"[FlowDataset] {split}: "
            f"{len(self.speakers)} 个说话人, {len(self.pairs)} 条音频"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_crop(self, path: str) -> Optional[torch.Tensor]:
        """加载音频，转为单声道，随机裁剪到 segment_len。"""
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception:
            return None
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        wav = torch.from_numpy(audio)
        if len(wav) < self.min_samples:
            return None
        if len(wav) >= self.segment_len:
            start = random.randint(0, len(wav) - self.segment_len)
            wav   = wav[start : start + self.segment_len]
        else:
            wav = F.pad(wav, (0, self.segment_len - len(wav)))
        return wav.unsqueeze(0)  # (1, T)

    def __getitem__(self, idx: int):
        spk, path = self.pairs[idx]

        # 加载主音频
        wav = self._load_crop(path)
        if wav is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 采样同说话人的参考音频（尽量不同文件）
        cands    = self.spk_files[spk]
        ref_path = path
        if len(cands) > 1:
            while ref_path == path:
                ref_path = random.choice(cands)
        ref_wav = self._load_crop(ref_path)
        if ref_wav is None:
            ref_wav = wav.clone()

        return wav, ref_wav   # both (1, T)
