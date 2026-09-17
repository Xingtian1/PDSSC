import argparse
import os
import sys

import torch


_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from dual_rvq_redesign.src.channels import QPSKAWGNChannel
from dual_rvq_redesign.src.config import ModelConfig
from dual_rvq_redesign.src.models import DualRVQModel
from dual_rvq_redesign.src.training.detector import ParentConsistencyDetector
from dual_rvq_redesign.src.training.stages import default_stage_configs
from speechtokenizer.flow import FlowMatchingModel


def fake_batch(batch_size: int = 2, length: int = 16000) -> torch.Tensor:
    return torch.randn(batch_size, 1, length)


def test_stage1(model: DualRVQModel, x: torch.Tensor) -> None:
    out = model(x)
    assert out["wav_hat"].shape[:2] == x.shape[:2]
    assert torch.isfinite(out["commitment"])
    assert out["semantic_feature"].shape[0] == x.shape[0]
    print("[stage1] forward ok", flush=True)


def test_stage2(model: DualRVQModel, x: torch.Tensor, ebno_db: float) -> None:
    out = model(x)
    ch_sem = QPSKAWGNChannel(ebno_db=ebno_db, soft_output=True)
    sem_rx_layers = []
    sem_conf_layers = []
    for idx, codebook_size in enumerate(model.config.semantic.codebook_sizes):
        bits_per_sem = codebook_size.bit_length() - 1
        flat_sem = out["semantic_codes"][idx].reshape(-1)
        tx_sem = ch_sem.transmit_indices(flat_sem, bits_per_index=bits_per_sem)
        sem_rx_layers.append(tx_sem["hard_indices"].reshape(out["semantic_codes"].shape[1], out["semantic_codes"].shape[2]))
        sem_conf_layers.append(tx_sem["confidence"].mean(dim=-1).reshape(out["semantic_codes"].shape[1], out["semantic_codes"].shape[2]))

    ac_rx_layers = []
    ac_conf_layers = []
    ac_changed = []
    for idx, codebook_size in enumerate(model.config.acoustic.codebook_sizes):
        bits_per_ac = codebook_size.bit_length() - 1
        flat_ac = out["acoustic_codes"][idx].reshape(-1)
        tx_ac = ch_sem.transmit_indices(flat_ac, bits_per_index=bits_per_ac)
        ac_rx_layers.append(tx_ac["hard_indices"].reshape(out["acoustic_codes"].shape[1], out["acoustic_codes"].shape[2]))
        ac_conf_layers.append(tx_ac["confidence"].mean(dim=-1).reshape(out["acoustic_codes"].shape[1], out["acoustic_codes"].shape[2]))
        ac_changed.append((tx_ac["hard_indices"] != flat_ac).float().mean())

    rx_sem = torch.stack(sem_rx_layers, dim=0)
    rx_ac = torch.stack(ac_rx_layers, dim=0)
    sem_conf_raw = torch.stack(sem_conf_layers, dim=0)
    ac_conf_raw = torch.stack(ac_conf_layers, dim=0)

    det = ParentConsistencyDetector(in_dim=4, hidden_dim=32)
    sem_logits_list = []
    sem_masks = []
    for idx, parent in enumerate(model.semantic_parents):
        sem_layer = model.semantic_branch.decode(rx_sem[idx : idx + 1], st=idx)
        sem_score = parent.consistency_score(sem_layer, out["semantic_parent_ids"][idx])
        sem_parent_latent = parent.decode(out["semantic_parent_ids"][idx])
        sem_parent_similarity = torch.nn.functional.cosine_similarity(
            sem_layer.permute(0, 2, 1),
            sem_parent_latent.permute(0, 2, 1),
            dim=-1,
        )
        sem_code_norm = sem_layer.permute(0, 2, 1).norm(dim=-1)
        sem_conf = sem_conf_raw[idx]
        sem_logits = det(sem_conf, sem_score, sem_parent_similarity, sem_code_norm)
        sem_logits_list.append(sem_logits)
        sem_masks.append(torch.sigmoid(sem_logits) > 0.5)

    ac_logits_list = []
    ac_masks = []
    for idx, parent in enumerate(model.acoustic_parents):
        ac_layer = model.acoustic_branch.decode(rx_ac[idx : idx + 1], st=idx)
        ac_score = parent.consistency_score(ac_layer, out["acoustic_parent_ids"][idx])
        ac_parent_latent = parent.decode(out["acoustic_parent_ids"][idx])
        ac_parent_similarity = torch.nn.functional.cosine_similarity(
            ac_layer.permute(0, 2, 1),
            ac_parent_latent.permute(0, 2, 1),
            dim=-1,
        )
        ac_code_norm = ac_layer.permute(0, 2, 1).norm(dim=-1)
        ac_conf = ac_conf_raw[idx]
        ac_logits = det(ac_conf, ac_score, ac_parent_similarity, ac_code_norm)
        ac_logits_list.append(ac_logits)
        ac_masks.append(torch.sigmoid(ac_logits) > 0.5)

    sem_mask = torch.stack(sem_masks, dim=0)
    ac_mask = torch.stack(ac_masks, dim=0)
    _q_sem, _q_ac, latent_corr = model.reconstruct_latent_with_parent_fallback(
        semantic_codes=rx_sem,
        acoustic_codes=rx_ac,
        semantic_parent_ids=out["semantic_parent_ids"],
        acoustic_parent_ids=out["acoustic_parent_ids"],
        semantic_parent_mask=sem_mask,
        acoustic_parent_mask=ac_mask,
    )
    flow_model = FlowMatchingModel(latent_dim=model.config.latent_dim, base_ch=128, ch_mults=(1, 1), cond_dim=128, spk_dim=256, time_dim=64, n_res=1, n_mid_res=1)
    spk_emb = torch.randn(x.shape[0], 256)
    latent_restored = flow_model.sample(latent_corr, spk_emb, n_steps=2)
    wav_restored = model.decoder(latent_restored)
    assert wav_restored.shape[:2] == x.shape[:2]
    assert torch.isfinite(wav_restored).all()
    changed = torch.stack(ac_changed).mean().item()
    print(f"[stage2] channel+parent+flow ok changed_ratio={changed:.4f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Dual-RVQ stage smoke tests")
    p.add_argument("--stage", default="all", choices=["all"] + [s.name for s in default_stage_configs()])
    p.add_argument("--ebno_db", type=float, default=0.0)
    args = p.parse_args()

    cfg = ModelConfig()
    model = DualRVQModel(cfg)
    x = fake_batch()

    if args.stage in ("all", "stage1_ed_fine"):
        test_stage1(model, x)
    if args.stage in ("all", "stage2_channel_parent_flow"):
        test_stage2(model, x, args.ebno_db)


if __name__ == "__main__":
    main()
