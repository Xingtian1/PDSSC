"""
数字信道丢包模型

BernoulliChannel     : 独立随机丢包（基线对比用）
GilbertElliotChannel : 二阶马尔可夫突发丢包（主要仿真模型）

Gilbert-Elliott 参数说明:
  p_loss      : 平均丢包率（稳态），如 0.10 表示 10%
  mean_burst  : 平均连续丢包长度（Bad 状态平均持续帧数）

内部转移概率推导:
  β  = 1 / mean_burst          (Bad → Good 概率)
  α  = p_loss × β / (1-p_loss) (Good → Bad 概率)
  稳态丢包率验证: α / (α + β) ≈ p_loss
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseChannel(ABC):
    @abstractmethod
    def transmit(self, n_packets: int) -> np.ndarray:
        """返回 bool 数组，True=收到，False=丢失"""

    @abstractmethod
    def reset(self):
        pass


class BernoulliChannel(BaseChannel):
    """每个包独立以概率 p_loss 丢失，无突发"""

    def __init__(self, p_loss: float, seed: int = None):
        assert 0.0 <= p_loss < 1.0
        self.p_loss = p_loss
        self.rng = np.random.default_rng(seed)

    def transmit(self, n_packets: int) -> np.ndarray:
        return self.rng.random(n_packets) > self.p_loss

    def reset(self):
        pass  # 无状态，无需重置

    def __repr__(self):
        return f"BernoulliChannel(p_loss={self.p_loss:.3f})"


class GilbertElliotChannel(BaseChannel):
    """
    二阶马尔可夫链（Gilbert-Elliott）突发丢包信道

    状态:
      G (Good) : 包必定收到（p_loss_good ≈ 0）
      B (Bad)  : 包必定丢失（p_loss_bad  = 1）

    用 (目标丢包率, 平均突发长度) 初始化，自动推导转移概率。
    """

    def __init__(
        self,
        p_loss: float,
        mean_burst: float = 3.0,
        p_loss_good: float = 0.0,
        p_loss_bad: float = 1.0,
        seed: int = None,
    ):
        """
        p_loss      : 目标平均丢包率
        mean_burst  : Bad 状态平均持续包数（突发长度）
        p_loss_good : Good 状态丢包率（默认 0，即无损）
        p_loss_bad  : Bad 状态丢包率（默认 1，即全丢）
        """
        assert 0.0 < p_loss < 1.0, "p_loss 必须在 (0, 1) 之间"
        assert mean_burst >= 1.0, "mean_burst 不能小于 1"

        self.p_loss = p_loss
        self.mean_burst = mean_burst
        self.p_loss_good = p_loss_good
        self.p_loss_bad = p_loss_bad
        self.rng = np.random.default_rng(seed)

        # 由用户参数推导转移概率
        self.beta  = 1.0 / mean_burst                          # P(B → G)
        self.alpha = p_loss * self.beta / (1.0 - p_loss)       # P(G → B)

        self.state = 'G'  # 初始状态

    # ── 理论属性 ──────────────────────────────────────────────
    @property
    def theoretical_plr(self) -> float:
        """理论稳态丢包率"""
        pi_b = self.alpha / (self.alpha + self.beta)
        pi_g = self.beta  / (self.alpha + self.beta)
        return pi_g * self.p_loss_good + pi_b * self.p_loss_bad

    @property
    def theoretical_burst_len(self) -> float:
        """Bad 状态平均持续长度"""
        return 1.0 / self.beta

    # ── 仿真 ──────────────────────────────────────────────────
    def transmit(self, n_packets: int) -> np.ndarray:
        """
        模拟 n_packets 个逻辑包的传输。
        返回 bool 数组：True=收到，False=丢失
        """
        received = np.zeros(n_packets, dtype=bool)
        for i in range(n_packets):
            if self.state == 'G':
                received[i] = self.rng.random() > self.p_loss_good
                if self.rng.random() < self.alpha:   # G → B
                    self.state = 'B'
            else:
                received[i] = self.rng.random() > self.p_loss_bad
                if self.rng.random() < self.beta:    # B → G
                    self.state = 'G'
        return received

    def reset(self):
        self.state = 'G'

    def __repr__(self):
        return (
            f"GilbertElliotChannel("
            f"p_loss={self.p_loss:.3f}, "
            f"mean_burst={self.mean_burst:.1f}, "
            f"α={self.alpha:.4f}, β={self.beta:.4f}, "
            f"theoretical_plr={self.theoretical_plr:.3f})"
        )
