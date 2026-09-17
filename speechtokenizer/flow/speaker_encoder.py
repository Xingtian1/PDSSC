"""
说话人编码器

使用 resemblyzer（Google GE2E）预训练说话人编码器。
  - 预训练于 VoxCeleb，输出 256 维嵌入
  - 骨干冻结，只训练一个 Linear 投影层（域适配）
  - 无复杂依赖，pip install resemblyzer 即可

安装：
  pip install resemblyzer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainedSpeakerEncoder(nn.Module):
    """
    resemblyzer GE2E 预训练说话人编码器。

    骨干（冻结）：GE2E，VoxCeleb 预训练，输出 256 维
    投影层（可训练）：Linear(256, emb_dim) + BN

    前向传播
    --------
    wav : (B, T) 或 (B, 1, T)，16kHz 单声道 float32 tensor
    返回: (B, emb_dim) L2 归一化说话人嵌入
    """

    BACKBONE_DIM = 256

    def __init__(
        self,
        emb_dim  : int = 256,
        save_dir : str = "",    # resemblyzer 自动管理权重缓存，此参数保留兼容性
    ):
        super().__init__()
        self.emb_dim  = emb_dim
        self._encoder = None    # 延迟加载

        # 可训练投影层
        self.proj = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def _get_encoder(self):
        """首次调用时加载（自动下载权重，约 17MB）。"""
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
            self._encoder.eval()
        return self._encoder

    @torch.no_grad()
    def _extract_backbone(self, wav: torch.Tensor) -> torch.Tensor:
        """
        用冻结骨干批量提取说话人嵌入。
        wav : (B, T) float32 tensor（任意设备）
        返回: (B, 256) float32 tensor（同设备）
        """
        device  = wav.device
        encoder = self._get_encoder()
        wavs_np = wav.cpu().float().numpy()   # (B, T) numpy

        embeds = np.stack([
            encoder.embed_utterance(w) for w in wavs_np
        ])  # (B, 256) numpy float32

        return torch.from_numpy(embeds).to(device)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav.squeeze(1)   # (B, 1, T) → (B, T)

        emb = self._extract_backbone(wav)   # (B, 256)，无梯度
        emb = self.proj(emb)                # (B, emb_dim)，有梯度
        return F.normalize(emb, p=2, dim=-1)

    def train(self, mode: bool = True):
        """骨干始终保持 eval 模式。"""
        super().train(mode)
        if self._encoder is not None:
            self._encoder.eval()
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 自训练版本（备用）
# ─────────────────────────────────────────────────────────────────────────────

class TDNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, in_ch, bottleneck=256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_ch * 3, bottleneck, 1), nn.Tanh(),
            nn.Conv1d(bottleneck, in_ch, 1), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        mu  = x.mean(dim=-1, keepdim=True).expand_as(x)
        std = x.std(dim=-1, keepdim=True).expand_as(x)
        w   = self.attn(torch.cat([x, mu, std], dim=1))
        mean = (w * x).sum(dim=-1)
        std_ = ((w * x**2).sum(dim=-1) - mean**2).clamp(1e-8).sqrt()
        return torch.cat([mean, std_], dim=1)


class SpeakerEncoder(nn.Module):
    """自训练 TDNN 说话人编码器（备用方案）。"""

    def __init__(self, emb_dim=256, n_mels=80):
        super().__init__()
        self.emb_dim     = emb_dim
        self.n_mels      = n_mels
        self._mel_fn     = None
        self._mel_device = ""

        self.tdnn = nn.Sequential(
            TDNNBlock(n_mels, 512, 5, 1), TDNNBlock(512, 512, 3, 2),
            TDNNBlock(512, 512, 3, 3),    TDNNBlock(512, 512, 1, 1),
            nn.Conv1d(512, 1536, 1), nn.ReLU(inplace=True),
        )
        self.asp  = AttentiveStatisticsPooling(1536, 256)
        self.proj = nn.Sequential(
            nn.BatchNorm1d(3072), nn.Linear(3072, emb_dim), nn.BatchNorm1d(emb_dim),
        )

    def _get_mel(self, device):
        if self._mel_fn is None or self._mel_device != str(device):
            import torchaudio.transforms as T
            self._mel_fn = T.MelSpectrogram(
                sample_rate=16000, n_fft=512, hop_length=160,
                n_mels=self.n_mels, f_min=20.0, f_max=7600.0,
            ).to(device)
            self._mel_device = str(device)
        return self._mel_fn

    def forward(self, wav):
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        mel = (self._get_mel(wav.device)(wav) + 1e-6).log()
        return F.normalize(self.proj(self.asp(self.tdnn(mel))), p=2, dim=-1)
