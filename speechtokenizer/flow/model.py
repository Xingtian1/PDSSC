"""
Rectified Flow 生成模型（1D U-Net）

训练目标:
  x_0 = latent_received  (接收端重建的 partial latent，已插值)
  x_1 = latent_8         (完整 8 层反量化之和)
  x_t = (1-t)*x_0 + t*x_1
  Loss = || v_θ(x_t, t, spk_emb) - (x_1 - x_0) ||^2

推理（ODE 求解）:
  x_0 → x_1 via  dx/dt = v_θ(x_t, t, spk_emb),  t: 0→1
  用 N 步欧拉法即可（Rectified Flow 的直线轨迹）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    正弦位置编码，用于时间步 t ∈ [0, 1]。
    t   : (B,)
    返回: (B, dim)
    """
    assert dim % 2 == 0
    device = t.device
    half   = dim // 2
    freq   = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=device) / (half - 1)
    )
    emb = t[:, None] * freq[None, :]          # (B, half)
    return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


# ─────────────────────────────────────────────────────────────────────────────
# 基础模块
# ─────────────────────────────────────────────────────────────────────────────

class FiLM(nn.Module):
    """Feature-wise Linear Modulation — 用 cond 对特征做 scale+shift"""
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, cond_dim)
        g_b = self.proj(cond)                # (B, 2C)
        gamma, beta = g_b.chunk(2, dim=-1)   # each (B, C)
        return x * (1 + gamma[:, :, None]) + beta[:, :, None]


class ResBlock1D(nn.Module):
    """1D 残差块，含 GroupNorm + SiLU + FiLM 条件注入"""
    def __init__(self, channels: int, cond_dim: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.film  = FiLM(cond_dim, channels)
        self.act   = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.film(h, cond)
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class DownBlock1D(nn.Module):
    """编码器块：残差堆叠 → 步长2下采样"""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, n_res: int = 2):
        super().__init__()
        self.proj_in = nn.Conv1d(in_ch, out_ch, 1)
        self.blocks  = nn.ModuleList([
            ResBlock1D(out_ch, cond_dim) for _ in range(n_res)
        ])
        self.down = nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x, cond)
        skip = x
        x    = self.down(x)
        return x, skip


class UpBlock1D(nn.Module):
    """解码器块：转置卷积上采样 → 跳接拼接 → 残差堆叠"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int, n_res: int = 2):
        super().__init__()
        self.up       = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.proj_in  = nn.Conv1d(out_ch + skip_ch, out_ch, 1)
        self.blocks   = nn.ModuleList([
            ResBlock1D(out_ch, cond_dim) for _ in range(n_res)
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 补齐长度差（stride=2 时可能差 1）
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x, cond)
        return x


class MiddleAttention(nn.Module):
    """瓶颈注意力块（序列自注意力）"""
    def __init__(self, channels: int, n_heads: int = 8):
        super().__init__()
        self.norm  = nn.GroupNorm(min(8, channels), channels)
        self.attn  = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.norm(x).permute(0, 2, 1)   # (B, T, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────

class FlowMatchingModel(nn.Module):
    """
    Rectified Flow 速度场估计器（1D U-Net）。

    参数
    ----
    latent_dim : 输入/输出 latent 维度（SpeechTokenizer 默认 1024）
    base_ch    : U-Net 基础通道数
    ch_mults   : 各层通道倍数（决定下采样层数）
    cond_dim   : 条件嵌入维度（时间 + 说话人）
    spk_dim    : 说话人嵌入维度（SpeakerEncoder 输出）
    time_dim   : 时间步正弦编码维度
    n_res      : 每个 DownBlock / UpBlock 中的 ResBlock 数量
    n_mid_res  : 瓶颈 ResBlock 数量
    n_heads    : 瓶颈注意力头数

    前向传播
    --------
    x       : (B, D, T) — 插值后接收到的 partial latent (x_t at time t)
    t       : (B,)      — 流动时间步，∈ [0, 1]
    spk_emb : (B, spk_dim) — 说话人嵌入

    返回
    ----
    v : (B, D, T) — 速度场估计，目标 ≈ x_1 - x_0
    """

    def __init__(
        self,
        latent_dim : int   = 1024,
        base_ch    : int   = 512,
        ch_mults   : tuple = (1, 1, 2),   # 3 级，最大 1024 通道
        cond_dim   : int   = 512,
        spk_dim    : int   = 256,
        time_dim   : int   = 128,
        n_res      : int   = 2,
        n_mid_res  : int   = 2,
        n_heads    : int   = 8,
        max_n_layers : int = 8,           # 最大传输层数（RVQ 层数上限）
    ):
        super().__init__()
        chs      = [base_ch * m for m in ch_mults]
        n_levels = len(chs) - 1   # 下采样次数
        self.time_dim = time_dim

        # ── 条件编码 ────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.spk_proj = nn.Sequential(
            nn.Linear(spk_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # N 层数条件：离散嵌入（index 1..max_n_layers）
        self.n_emb = nn.Embedding(max_n_layers + 1, cond_dim)
        nn.init.zeros_(self.n_emb.weight)   # 初始为零，等效于旧模型行为

        # ── 输入投影 ─────────────────────────────────────────
        self.input_proj = nn.Conv1d(latent_dim, chs[0], 1)

        # ── 编码器 ───────────────────────────────────────────
        self.down_blocks = nn.ModuleList([
            DownBlock1D(chs[i], chs[i + 1], cond_dim, n_res)
            for i in range(n_levels)
        ])

        # ── 瓶颈 ─────────────────────────────────────────────
        self.mid_blocks = nn.ModuleList([
            ResBlock1D(chs[-1], cond_dim) for _ in range(n_mid_res)
        ])
        self.mid_attn = MiddleAttention(chs[-1], n_heads)

        # ── 解码器 ───────────────────────────────────────────
        self.up_blocks = nn.ModuleList([
            UpBlock1D(chs[i + 1], chs[i + 1], chs[i], cond_dim, n_res)
            for i in range(n_levels - 1, -1, -1)
        ])

        # ── 输出投影 ─────────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, chs[0]), chs[0]),
            nn.SiLU(),
            nn.Conv1d(chs[0], latent_dim, 1),
        )

    def forward(
        self,
        x       : torch.Tensor,             # (B, D, T)
        t       : torch.Tensor,             # (B,)
        spk_emb : torch.Tensor,             # (B, spk_dim)
        n_layers: torch.Tensor = None,      # (B,) int64, 传输层数 1..8
    ) -> torch.Tensor:

        # 条件向量：时间 + 说话人 + 传输层数
        t_emb  = sinusoidal_embedding(t, self.time_dim)         # (B, time_dim)
        cond   = self.time_mlp(t_emb) + self.spk_proj(spk_emb) # (B, cond_dim)
        if n_layers is not None:
            cond = cond + self.n_emb(n_layers.long())            # (B, cond_dim)

        # 输入投影
        h = self.input_proj(x)   # (B, chs[0], T)

        # 编码器（收集 skip）
        skips = []
        for down in self.down_blocks:
            h, skip = down(h, cond)
            skips.append(skip)

        # 瓶颈
        for mid in self.mid_blocks:
            h = mid(h, cond)
        h = self.mid_attn(h)

        # 解码器（消费 skip，逆序）
        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h, skip, cond)

        return self.output_proj(h)   # (B, D, T)

    @torch.no_grad()
    def sample(
        self,
        x_0     : torch.Tensor,        # (B, D, T) — received latent (x_0)
        spk_emb : torch.Tensor,        # (B, spk_dim)
        n_steps : int = 10,
        n_layers: torch.Tensor = None, # (B,) int64, 传输层数（可选）
    ) -> torch.Tensor:
        """
        欧拉法 ODE 求解：从 x_0 出发，沿速度场前进到 x_1。
        n_steps 步即可（Rectified Flow 轨迹接近直线）。
        返回 x_1 : (B, D, T)
        """
        x  = x_0.clone()
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i * dt
            t     = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
            v     = self(x, t, spk_emb, n_layers)
            x     = x + v * dt
        return x
