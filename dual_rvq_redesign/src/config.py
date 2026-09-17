from dataclasses import dataclass, field
from math import prod


@dataclass
class BranchConfig:
    name: str
    branch_dim: int
    n_q: int
    codebook_sizes: tuple[int, ...]
    frame_rate_hz: float = 40.0
    parent_codebook_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.codebook_sizes) != self.n_q:
            raise ValueError(
                f"{self.name} codebook_sizes must have length {self.n_q}, "
                f"got {len(self.codebook_sizes)}"
            )
        if len(self.parent_codebook_sizes) != self.n_q:
            raise ValueError(
                f"{self.name} parent_codebook_sizes must have length {self.n_q}, "
                f"got {len(self.parent_codebook_sizes)}"
            )

    @property
    def bitrate_kbps(self) -> float:
        return self.frame_rate_hz * sum(size.bit_length() - 1 for size in self.codebook_sizes) / 1000.0

    @property
    def bits_per_layer(self) -> tuple[int, ...]:
        return tuple(size.bit_length() - 1 for size in self.codebook_sizes)

    @property
    def parent_bitrate_kbps(self) -> float:
        return (
            self.frame_rate_hz
            * sum(size.bit_length() - 1 for size in self.parent_codebook_sizes)
            / 1000.0
        )

    @property
    def total_bitrate_kbps(self) -> float:
        return self.bitrate_kbps + self.parent_bitrate_kbps


@dataclass
class ChannelConfig:
    modulation: str = "qpsk"
    ebno_db: float = 8.0
    bits_per_symbol: int = 2
    soft_output: bool = True


@dataclass
class ModelConfig:
    sample_rate: int = 16000
    latent_dim: int = 1024
    n_filters: int = 64
    strides: tuple[int, ...] = (5, 5, 5, 4)
    lstm_layers: int = 2
    bidirectional: bool = True
    dilation_base: int = 2
    residual_kernel_size: int = 3
    n_residual_layers: int = 1
    activation: str = "ELU"
    semantic: BranchConfig = field(
        default_factory=lambda: BranchConfig(
            name="semantic",
            branch_dim=512,
            n_q=1,
            codebook_sizes=(1024,),
            parent_codebook_sizes=(16,),
        )
    )
    acoustic: BranchConfig = field(
        default_factory=lambda: BranchConfig(
            name="acoustic",
            branch_dim=512,
            n_q=2,
            codebook_sizes=(512, 256),
            parent_codebook_sizes=(8, 8),
        )
    )
    channel: ChannelConfig = field(default_factory=ChannelConfig)

    def __post_init__(self) -> None:
        frame_rate = self.frame_rate_hz
        self.semantic.frame_rate_hz = frame_rate
        self.acoustic.frame_rate_hz = frame_rate

    @property
    def downsample_ratio(self) -> int:
        return prod(self.strides)

    @property
    def frame_rate_hz(self) -> float:
        return self.sample_rate / float(self.downsample_ratio)

    @property
    def target_bitrate_kbps(self) -> float:
        return self.semantic.total_bitrate_kbps + self.acoustic.total_bitrate_kbps
