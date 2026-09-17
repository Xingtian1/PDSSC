"""
完整的发送-信道-接收流水线

发送端:
  1. SpeechTokenizer encode → codes (4层, T帧)
  2. Q1 单独可靠传输（p_loss_q1 可配置，默认 0）
  3. Q2/Q3/Q4 → BlockInterleaver → 逻辑包序列 → GilbertElliot 信道

接收端:
  1. Q1 丢失帧 → 相邻帧 latent 线性插值恢复
  2. Q2/Q3/Q4 → 解交织 → 得到 (codes_recv, layer_mask)
  3. 用收到的 codes 重建 partial latent（Flow 模型的输入）
"""

import torch
import numpy as np
from typing import Tuple, Optional

from speechtokenizer import SpeechTokenizer
from .interleaver import BlockInterleaver
from .channel import GilbertElliotChannel, BernoulliChannel, BaseChannel


class TransmissionPipeline:
    def __init__(
        self,
        model         : SpeechTokenizer,
        interleave_N  : int   = 6,          # 交织深度，支持 3 或 6
        p_loss        : float = 0.05,       # Q2-Q4 信道丢包率
        mean_burst    : float = 3.0,        # 平均突发长度（包数）
        p_loss_q1     : float = 0.0,        # Q1 信道丢包率（默认可靠）
        seed          : int   = 42,
    ):
        self.model = model
        self.device = next(model.parameters()).device

        self.interleaver = BlockInterleaver(N=interleave_N, n_supp=3)

        # Q2-Q4 突发丢包信道
        self.supp_channel = GilbertElliotChannel(
            p_loss=p_loss, mean_burst=mean_burst, seed=seed
        )
        # Q1 信道（独立，默认无损）
        if p_loss_q1 == 0.0:
            self.q1_channel = None  # 完全可靠，不模拟
        else:
            self.q1_channel = BernoulliChannel(p_loss=p_loss_q1, seed=seed + 1)

    # ─────────────────────────────────────────────────────────
    # 编码
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> dict:
        """
        wav : (1, T) 单声道音频
        返回编码结果字典（供 transmit 使用）
        """
        x     = wav.unsqueeze(0).to(self.device)   # (1, 1, T)
        codes = self.model.encode(x, n_q=4)       # (4, 1, T_enc)
        codes = codes.squeeze(1)                   # (4, T_enc)

        q1_codes   = codes[0]                      # (T_enc,)
        supp_codes = codes[1:]                     # (3, T_enc)

        return {"q1_codes": q1_codes, "supp_codes": supp_codes}

    # ─────────────────────────────────────────────────────────
    # 发送端打包 + 信道传输
    # ─────────────────────────────────────────────────────────
    def transmit(self, encoded: dict) -> dict:
        """
        模拟打包 → 信道 → 解包全流程。
        返回接收端的 codes 和 mask。
        """
        q1_codes   = encoded["q1_codes"]    # (T_enc,)
        supp_codes = encoded["supp_codes"]  # (3, T_enc)
        T_enc      = q1_codes.shape[0]

        # ── Q2-Q4 交织打包 ──────────────────────────────────
        packets, assignment, T_orig = self.interleaver.interleave(supp_codes.cpu())
        n_packets = packets.shape[0]

        # ── 信道传输 Q2-Q4 ──────────────────────────────────
        self.supp_channel.reset()
        supp_received = self.supp_channel.transmit(n_packets)  # (n_packets,) bool

        # ── 信道传输 Q1 ─────────────────────────────────────
        if self.q1_channel is not None:
            q1_received = self.q1_channel.transmit(T_enc)  # (T_enc,) bool
        else:
            q1_received = np.ones(T_enc, dtype=bool)       # 全部收到

        # ── 解交织 Q2-Q4 ─────────────────────────────────────
        codes_recv, layer_mask = self.interleaver.deinterleave(
            packets, assignment, supp_received, T_orig
        )

        # ── 统计 ────────────────────────────────────────────
        supp_loss_rate = 1.0 - supp_received.mean()
        q1_loss_rate   = 1.0 - q1_received.mean()

        return {
            "q1_codes"      : q1_codes,           # (T_enc,)       原始 Q1
            "q1_received"   : q1_received,         # (T_enc,) bool
            "supp_codes_recv": codes_recv,         # (3, T_enc)     收到的 Q2-Q4
            "layer_mask"    : layer_mask,          # (3, T_enc) bool
            "supp_loss_rate": supp_loss_rate,
            "q1_loss_rate"  : q1_loss_rate,
        }

    # ─────────────────────────────────────────────────────────
    # 通用：对任意一层的丢失帧做相邻帧 latent 插值
    # ─────────────────────────────────────────────────────────
    def _interpolate_missing(
        self,
        latents  : torch.Tensor,  # (T, D) 已解码的 latent，丢失帧为零
        received : np.ndarray,    # (T,) bool
    ) -> torch.Tensor:
        """
        对 received=False 的帧，用同层前后最近有效帧线性插值填充。
        向量化实现，避免 Python 逐帧循环。
        """
        if np.all(received):
            return latents

        missing_idx = np.where(~received)[0]
        valid_idx   = np.where(received)[0]

        if len(valid_idx) == 0 or len(missing_idx) == 0:
            return latents

        # searchsorted 找每个丢失帧在 valid_idx 中的插入位置
        ins      = np.searchsorted(valid_idx, missing_idx)
        has_prev = ins > 0
        has_next = ins < len(valid_idx)

        prev_vi = np.clip(ins - 1, 0, len(valid_idx) - 1)
        next_vi = np.clip(ins,     0, len(valid_idx) - 1)
        prev_t  = valid_idx[prev_vi]   # 每个丢失帧的前一有效帧时间索引
        next_t  = valid_idx[next_vi]   # 每个丢失帧的后一有效帧时间索引

        denom = np.maximum(next_t - prev_t, 1).astype(np.float32)
        alpha = np.where(has_prev & has_next,
                         (missing_idx - prev_t) / denom, 0.0).astype(np.float32)

        dev = latents.device
        alpha_t  = torch.from_numpy(alpha).to(dev).unsqueeze(1)           # (n_miss, 1)
        prev_t_t = torch.from_numpy(prev_t).long().to(dev)
        next_t_t = torch.from_numpy(next_t).long().to(dev)
        miss_t   = torch.from_numpy(missing_idx).long().to(dev)

        interp = (1 - alpha_t) * latents[prev_t_t] + alpha_t * latents[next_t_t]

        only_prev = torch.from_numpy(has_prev & ~has_next).to(dev)
        only_next = torch.from_numpy(~has_prev & has_next).to(dev)
        if only_prev.any():
            interp[only_prev] = latents[prev_t_t[only_prev]]
        if only_next.any():
            interp[only_next] = latents[next_t_t[only_next]]

        latents[miss_t] = interp
        return latents

    @torch.no_grad()
    def recover_layer_latent(
        self,
        codes    : torch.Tensor,  # (T_enc,) 某层的 code 索引
        received : np.ndarray,    # (T_enc,) bool
        layer_idx: int,           # RVQ 层编号（0=Q1, 1=Q2, 2=Q3, 3=Q4）
    ) -> torch.Tensor:
        """
        将某层的 code 解码到 latent 向量，对丢失帧相邻帧插值恢复。
        批量 decode 所有已收到帧，避免逐帧调用。
        返回: (T_enc, D)
        """
        T        = codes.shape[0]
        D        = self.model.quantizer.dimension
        vq_layer = self.model.quantizer.vq.layers[layer_idx]

        latents     = torch.zeros(T, D, device=self.device)
        recv_idx    = np.where(received)[0]

        if len(recv_idx) > 0:
            batch_codes = codes[torch.from_numpy(recv_idx).long()].unsqueeze(0).to(self.device)  # (1, n_recv)
            batch_vecs  = vq_layer.decode(batch_codes)                                            # (1, D, n_recv) or (1, n_recv, D)
            bv = batch_vecs.squeeze(0)  # (D, n_recv) or (n_recv, D)
            if bv.shape[0] == D:        # (D, n_recv) → 转置为 (n_recv, D)
                bv = bv.T
            latents[torch.from_numpy(recv_idx).long().to(self.device)] = bv

        return self._interpolate_missing(latents, received)

    # ─────────────────────────────────────────────────────────
    # 接收端重建 partial latent（Flow 模型输入）
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def build_partial_latent(self, rx: dict) -> dict:
        """
        用接收到的 codes 重建 partial latent。
        未收到的层用零向量填充，mask 指示哪些位置有效。

        返回:
          latent_q1    : (1, D, T_enc)  Q1 latent（已插值恢复）
          latent_partial:(1, D, T_enc)  Q1+存活Q2-Q4 的累加 latent
          full_mask    : (4, T_enc) bool  第0行=Q1 mask，后3行=Q2-Q4 mask
        """
        q1_codes       = rx["q1_codes"].to(self.device)
        q1_received    = rx["q1_received"]
        supp_codes_recv= rx["supp_codes_recv"].to(self.device)  # (3, T_enc)
        layer_mask     = rx["layer_mask"]                        # (3, T_enc) bool
        T_enc          = q1_codes.shape[0]

        D = self.model.quantizer.dimension

        # Q1：解码 + 插值恢复丢失帧 → (T, D) → (1, D, T)
        q1_latent = self.recover_layer_latent(q1_codes, q1_received, layer_idx=0)
        q1_latent = q1_latent.T.unsqueeze(0)  # (1, D, T)

        # Q2-Q4：每层独立解码 + 插值恢复丢失帧
        # 无论是否丢失，所有层都做插值补全，保证 latent_partial 无空洞
        supp_latent = torch.zeros(3, D, T_enc, device=self.device)
        for l in range(3):
            layer_received = layer_mask[l].numpy()   # (T_enc,) bool
            lat = self.recover_layer_latent(
                supp_codes_recv[l], layer_received, layer_idx=l + 1
            )                                        # (T_enc, D)
            supp_latent[l] = lat.T                   # (D, T_enc)

        # 累加所有收到层的 latent
        latent_partial = q1_latent + supp_latent.sum(0, keepdim=True)  # (1, D, T)

        # 合并 mask（Q1 行 + Q2-Q4 行）
        q1_mask_tensor  = torch.from_numpy(q1_received).unsqueeze(0)  # (1, T)
        full_mask       = torch.cat([q1_mask_tensor, layer_mask], dim=0)  # (4, T)

        return {
            "latent_q1"     : q1_latent,       # (1, D, T)
            "latent_partial": latent_partial,   # (1, D, T)  Flow 模型先验
            "full_mask"     : full_mask,        # (4, T)     指示哪些层丢失
        }

    # ─────────────────────────────────────────────────────────
    # 解码（无 Flow 模型的基线）
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def decode_partial(self, latent_partial: torch.Tensor) -> torch.Tensor:
        """
        直接用 partial latent 解码，作为无 Flow 模型的基线。
        latent_partial : (1, D, T_enc)
        返回           : (1, T) 重建音频
        """
        wav_out = self.model.decoder(latent_partial)  # (1, 1, T)
        return wav_out.squeeze(0)                      # (1, T)

    # ─────────────────────────────────────────────────────────
    # 完整端到端流水线（单条音频）
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def run(self, wav: torch.Tensor) -> dict:
        """
        wav : (1, T) 单声道 16kHz 音频

        返回字典包含:
          wav_4layer      : 4层完整解码（无信道损伤，上界参考）
          wav_8layer      : 8层完整解码（质量上界）
          wav_partial     : 信道损伤后直接解码（无 Flow 补全，基线下界）
          latent_partial  : (1, D, T_enc) Flow 模型输入
          full_mask       : (4, T_enc)    层丢失 mask
          supp_loss_rate  : Q2-Q4 实际丢包率
          q1_loss_rate    : Q1 实际丢包率
        """
        # 编码
        encoded = self.encode(wav)
        q1      = encoded["q1_codes"]
        supp    = encoded["supp_codes"]

        # 4层/8层参考解码（无信道）
        codes_4 = torch.stack([q1] + [supp[i] for i in range(3)]).unsqueeze(1)
        codes_8 = self.model.encode(wav.unsqueeze(0).to(self.device)).squeeze(1)

        # 需要重新 encode 获取全8层
        x       = wav.unsqueeze(0).to(self.device)
        codes_all = self.model.encode(x)
        wav_4   = self.model.decode(codes_all[:4]).squeeze(0).cpu()
        wav_8   = self.model.decode(codes_all).squeeze(0).cpu()

        # 信道传输
        rx      = self.transmit(encoded)

        # 接收端重建
        partial = self.build_partial_latent(rx)
        wav_partial = self.decode_partial(
            partial["latent_partial"]
        ).cpu()

        return {
            "wav_4layer"     : wav_4,
            "wav_8layer"     : wav_8,
            "wav_partial"    : wav_partial,
            "latent_partial" : partial["latent_partial"],
            "full_mask"      : partial["full_mask"],
            "supp_loss_rate" : rx["supp_loss_rate"],
            "q1_loss_rate"   : rx["q1_loss_rate"],
        }
