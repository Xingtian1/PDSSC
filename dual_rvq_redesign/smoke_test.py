import os
import sys


_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from dual_rvq_redesign.src.config import ModelConfig
from dual_rvq_redesign.src.models import DualRVQModel


def main() -> None:
    cfg = ModelConfig()
    model = DualRVQModel(cfg)
    print(f"estimated_bitrate_kbps={model.estimate_bitrate_kbps():.3f}")
    print(
        "semantic:",
        f"L={cfg.semantic.n_q}",
        f"K={list(cfg.semantic.codebook_sizes)}",
        f"rate={cfg.semantic.bitrate_kbps:.3f}kbps",
        f"parent_K={list(cfg.semantic.parent_codebook_sizes)}",
        f"parent_rate={cfg.semantic.parent_bitrate_kbps:.3f}kbps",
    )
    print(
        "acoustic:",
        f"L={cfg.acoustic.n_q}",
        f"K={list(cfg.acoustic.codebook_sizes)}",
        f"rate={cfg.acoustic.bitrate_kbps:.3f}kbps",
        f"parent_K={list(cfg.acoustic.parent_codebook_sizes)}",
        f"parent_rate={cfg.acoustic.parent_bitrate_kbps:.3f}kbps",
    )


if __name__ == "__main__":
    main()
