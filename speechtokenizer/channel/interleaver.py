"""
块交织器（Block Interleaver）

将同一帧的 Q2/Q3/Q4 数据分散到不同逻辑包中，
使突发包丢失转化为帧内散布的层丢失，
让 Flow 生成模型可以帧内独立补全，无需跨帧插值。

交织规则：
  逻辑包编号 = (block_offset + local_frame + layer * stride) % N
  stride     = N // n_supp_layers

参数对比:
  N=3, stride=1: 突发 ≤1 包 → 每帧最多丢1层；突发 2 → 每帧最多丢2层
  N=6, stride=2: 突发 ≤2 包 → 每帧最多丢1层；突发 5 → 每帧最多丢3层（全丢）

延迟代价: N 帧 × 20ms/帧（编解码端各一次）
"""

import torch
import numpy as np
from typing import Tuple


class BlockInterleaver:
    def __init__(self, N: int, n_supp: int = 3):
        """
        N      : 交织深度（每块的帧数），必须是 n_supp 的整数倍
        n_supp : 补充层数量（Q2, Q3, Q4 → 3）
        """
        if N % n_supp != 0:
            raise ValueError(f"N ({N}) 必须是 n_supp ({n_supp}) 的整数倍")
        self.N      = N
        self.L      = n_supp
        self.stride = N // n_supp  # 同一帧各层之间的包间距

    # ── 核心映射函数 ──────────────────────────────────────────
    def _local_packet(self, local_frame: int, layer: int) -> int:
        """帧内局部编号 → 块内逻辑包编号"""
        return (local_frame + layer * self.stride) % self.N

    def _build_assignment(self, T_pad: int) -> torch.Tensor:
        """
        预计算全局包分配表。
        assignment[l, f] = 全局逻辑包编号（含块偏移）
        shape: (L, T_pad)
        """
        n_blocks   = T_pad // self.N
        assignment = torch.zeros(self.L, T_pad, dtype=torch.long)
        for b in range(n_blocks):
            for f_local in range(self.N):
                g_frame = b * self.N + f_local
                for l in range(self.L):
                    p_local  = self._local_packet(f_local, l)
                    p_global = b * self.N + p_local
                    assignment[l, g_frame] = p_global
        return assignment

    # ── 交织（发送端）────────────────────────────────────────
    def interleave(
        self, supp_codes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        supp_codes : (L, T_enc)  Q2/Q3/Q4 码书索引

        返回:
          packets    : (n_packets, L)  每行是一个逻辑包的内容（L 个 code）
          assignment : (L, T_pad)      每个 (layer, frame) 对应的逻辑包编号
          T_orig     : 原始帧数（填充前）
        """
        L, T   = supp_codes.shape
        assert L == self.L, f"层数不匹配: 期望 {self.L}, 得到 {L}"

        # 填充到 N 的整数倍
        n_blocks = (T + self.N - 1) // self.N
        T_pad    = n_blocks * self.N
        padded   = torch.zeros(L, T_pad, dtype=supp_codes.dtype)
        padded[:, :T] = supp_codes

        assignment  = self._build_assignment(T_pad)
        n_packets   = n_blocks * self.N

        # 按包聚合：packets[p, k] = 第 p 包第 k 个槽位的 code
        packets      = torch.zeros(n_packets, L, dtype=supp_codes.dtype)
        slot_counter = torch.zeros(n_packets, dtype=torch.long)

        for l in range(L):
            for f in range(T_pad):
                p    = assignment[l, f].item()
                slot = slot_counter[p].item()
                packets[p, slot] = padded[l, f]
                slot_counter[p] += 1

        return packets, assignment, T

    # ── 解交织（接收端）──────────────────────────────────────
    def deinterleave(
        self,
        packets    : torch.Tensor,   # (n_packets, L)
        assignment : torch.Tensor,   # (L, T_pad)
        received   : np.ndarray,     # (n_packets,) bool
        T_orig     : int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          codes_recv : (L, T_orig)  收到的 codes，丢失处为 0
          layer_mask : (L, T_orig)  bool，True 表示该 (layer, frame) 已收到
        """
        T_pad      = assignment.shape[1]
        codes_recv = torch.zeros(self.L, T_pad, dtype=packets.dtype)
        layer_mask = torch.zeros(self.L, T_pad, dtype=torch.bool)

        # 将逻辑包编号反查为 (layer, frame) 列表
        # 利用 assignment 逆映射
        for l in range(self.L):
            for f in range(T_pad):
                p = assignment[l, f].item()
                if received[p]:
                    codes_recv[l, f] = packets[p, self._slot_of(l, f, assignment)]
                    layer_mask[l, f] = True

        return codes_recv[:, :T_orig], layer_mask[:, :T_orig]

    def _slot_of(self, layer: int, frame: int, assignment: torch.Tensor) -> int:
        """
        找到 (layer, frame) 在其对应逻辑包中的槽位编号。
        同一包内的槽位按 (layer, frame) 遍历顺序排列（与 interleave 填包顺序一致）。
        """
        p_target = assignment[layer, frame].item()
        slot     = 0
        T_pad    = assignment.shape[1]
        for l in range(self.L):
            for f in range(T_pad):
                if assignment[l, f].item() == p_target:
                    if l == layer and f == frame:
                        return slot
                    slot += 1
        return 0  # should not reach

    # ── 调试工具 ─────────────────────────────────────────────
    def print_block_layout(self, n_blocks: int = 1):
        """打印前 n_blocks 块的分配示意"""
        print(f"\nBlockInterleaver N={self.N}, L={self.L}, stride={self.stride}")
        for b in range(n_blocks):
            print(f"\n  Block {b}:")
            header = "  Frame\\Layer " + "  ".join([f"Q{l+2}" for l in range(self.L)])
            print(header)
            for f_local in range(self.N):
                g_frame = b * self.N + f_local
                pkts = [f"P{b*self.N + self._local_packet(f_local, l)}" for l in range(self.L)]
                print(f"  F{g_frame:<10} " + "  ".join(pkts))

    def protection_analysis(self) -> dict:
        """
        分析不同突发长度下每帧最多丢失的层数。
        返回: {burst_len: max_layers_lost_per_frame}
        """
        result = {}
        for burst in range(1, self.N + 1):
            # 最坏情况：包 P0 ~ P{burst-1} 丢失
            max_lost = 0
            for f_local in range(self.N):
                layers_lost = sum(
                    1 for l in range(self.L)
                    if self._local_packet(f_local, l) < burst
                )
                max_lost = max(max_lost, layers_lost)
            result[burst] = max_lost
        return result
