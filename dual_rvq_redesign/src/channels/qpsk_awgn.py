from __future__ import annotations

import math

import torch


class QPSKAWGNChannel:
    def __init__(self, ebno_db: float = 8.0, soft_output: bool = True):
        self.ebno_db = ebno_db
        self.soft_output = soft_output

    @staticmethod
    def _indices_to_bits(indices: torch.Tensor, bits_per_index: int) -> torch.Tensor:
        shifts = torch.arange(bits_per_index - 1, -1, -1, device=indices.device)
        return ((indices.unsqueeze(-1) >> shifts) & 1).to(torch.float32)

    @staticmethod
    def _bits_to_indices(bits: torch.Tensor) -> torch.Tensor:
        bits = bits.to(torch.int64)
        shifts = torch.arange(bits.shape[-1] - 1, -1, -1, device=bits.device)
        return (bits * (2 ** shifts)).sum(dim=-1)

    @staticmethod
    def _pack_qpsk(bits: torch.Tensor) -> torch.Tensor:
        if bits.shape[-1] % 2 != 0:
            bits = torch.cat([bits, torch.zeros_like(bits[..., :1])], dim=-1)
        pairs = bits.reshape(*bits.shape[:-1], -1, 2)
        real = 1.0 - 2.0 * pairs[..., 0]
        imag = 1.0 - 2.0 * pairs[..., 1]
        return torch.complex(real, imag) / math.sqrt(2.0)

    @staticmethod
    def _unpack_qpsk(symbols: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real = symbols.real
        imag = symbols.imag
        bits = torch.stack([(real < 0).to(torch.float32), (imag < 0).to(torch.float32)], dim=-1)
        confidence = torch.stack([real.abs(), imag.abs()], dim=-1)
        return bits.reshape(*bits.shape[:-2], -1), confidence.reshape(*confidence.shape[:-2], -1)

    def _noise_std(self) -> float:
        ebno = 10.0 ** (self.ebno_db / 10.0)
        esno = 2.0 * ebno
        return math.sqrt(1.0 / (2.0 * esno))

    def transmit_bits(self, bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tx = self._pack_qpsk(bits)
        sigma = self._noise_std()
        noise = torch.complex(
            torch.randn_like(tx.real) * sigma,
            torch.randn_like(tx.imag) * sigma,
        )
        rx = tx + noise
        rx_bits, confidence = self._unpack_qpsk(rx)
        return rx_bits, confidence

    def transmit_indices(self, indices: torch.Tensor, bits_per_index: int) -> dict[str, torch.Tensor]:
        bits = self._indices_to_bits(indices.to(torch.int64), bits_per_index)
        rx_bits, confidence = self.transmit_bits(bits)
        rx_bits = rx_bits[..., :bits_per_index]
        confidence = confidence[..., :bits_per_index]
        hard_indices = self._bits_to_indices(rx_bits)
        return {
            "tx_bits": bits,
            "rx_bits": rx_bits,
            "confidence": confidence,
            "hard_indices": hard_indices,
        }
