from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_FILE_DIR)
_PKG_DIR = os.path.dirname(_SRC_DIR)
_ROOT_DIR = os.path.dirname(_PKG_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from speechtokenizer.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)
from speechtokenizer.flow import FlowMatchingModel, PretrainedSpeakerEncoder

from dual_rvq_redesign.src.channels import QPSKAWGNChannel
from dual_rvq_redesign.src.config import ModelConfig
from dual_rvq_redesign.src.models import DualRVQModel
from dual_rvq_redesign.src.training.dataset import build_codec_feature_loader, build_librispeech_loader
from dual_rvq_redesign.src.training.detector import ParentConsistencyDetector
from dual_rvq_redesign.src.training.losses import (
    adversarial_loss,
    branch_energy_balance_loss,
    confidence_penalty,
    d_axis_distill_loss,
    discriminator_loss,
    feature_loss,
    mel_loss,
    parent_consistency_loss,
    recon_loss,
    waveform_l1,
)
from dual_rvq_redesign.src.training.stages import default_stage_configs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flow_sample(flow_model: FlowMatchingModel, latent: torch.Tensor, spk_emb: torch.Tensor, n_steps: int) -> torch.Tensor:
    return flow_model.sample(latent, spk_emb, n_steps=n_steps)


class DualRVQTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cfg = ModelConfig()
        self.model = DualRVQModel(self.model_cfg).to(self.device)
        self.stage_map = {s.name: s for s in default_stage_configs()}
        self.stage = self.stage_map[args.stage]
        self.channel = QPSKAWGNChannel(ebno_db=args.ebno_db, soft_output=True)

        self.codec_loader = None
        self.channel_loader = None
        self.discriminators = None
        self.optim_d = None
        self.detector = None
        self.flow_model = None
        self.spk_encoder = None

        if self.stage.name == "stage1_ed_fine":
            self.codec_loader = build_codec_feature_loader(
                train_files=args.train_files,
                segment_size=args.segment_size,
                sample_rate=self.model_cfg.sample_rate,
                downsample_rate=int(np.prod(self.model_cfg.strides)),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            self.discriminators = {
                "mpd": MultiPeriodDiscriminator().to(self.device),
                "msd": MultiScaleDiscriminator().to(self.device),
                "mstftd": MultiScaleSTFTDiscriminator(32).to(self.device),
            }
        else:
            self.channel_loader = build_librispeech_loader(
                data_dir=args.data_dir,
                split=args.split,
                segment_sec=args.segment_sec,
                sample_rate=self.model_cfg.sample_rate,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            self.detector = ParentConsistencyDetector(in_dim=4, hidden_dim=args.detector_hidden_dim).to(self.device)
            self.spk_encoder = PretrainedSpeakerEncoder(emb_dim=args.spk_dim).to(self.device)
            self.flow_model = FlowMatchingModel(
                latent_dim=self.model_cfg.latent_dim,
                base_ch=args.flow_base_ch,
                ch_mults=tuple(args.flow_ch_mults),
                cond_dim=args.flow_cond_dim,
                spk_dim=args.spk_dim,
                time_dim=args.flow_time_dim,
                n_res=args.flow_n_res,
                n_mid_res=args.flow_n_mid_res,
            ).to(self.device)

        self.optim_g = torch.optim.AdamW(self._generator_parameters(), lr=args.lr, weight_decay=1e-4)
        if self.discriminators is not None:
            d_params = []
            for disc in self.discriminators.values():
                d_params.extend(list(disc.parameters()))
            self.optim_d = torch.optim.AdamW(d_params, lr=args.lr, weight_decay=1e-4)

        self.mel_kwargs_list = [
            {
                "n_fft": args.n_fft // mult,
                "num_mels": args.num_mels,
                "sample_rate": self.model_cfg.sample_rate,
                "hop_size": args.hop_size // mult,
                "win_size": args.win_size // mult,
                "fmin": args.fmin,
                "fmax": args.fmax_for_loss,
            }
            for mult in (1, 2, 4, 8)
        ]
        self.mel_loss_lambdas = args.mel_loss_lambdas

    def _generator_parameters(self):
        params = list(self.model.parameters())
        if self.detector is not None:
            params.extend(list(self.detector.parameters()))
        if self.flow_model is not None:
            params.extend(list(self.flow_model.parameters()))
        if self.spk_encoder is not None:
            params.extend(list(self.spk_encoder.proj.parameters()))
        return params

    def _stage1_generator_loss(self, x: torch.Tensor, semantic_feature: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        x = x.unsqueeze(1)
        out = self.model(x)
        x_hat = out["wav_hat"]

        disc_outputs = [disc(x, x_hat) for disc in self.discriminators.values()]
        loss_recon = recon_loss(x, x_hat)
        loss_mel = 0.0
        for lam, kwargs in zip(self.mel_loss_lambdas, self.mel_kwargs_list):
            loss_mel = loss_mel + lam * mel_loss(x, x_hat, **kwargs)
        loss_feature = sum(feature_loss(disc_out[2], disc_out[3]) for disc_out in disc_outputs)
        loss_adversarial = sum(adversarial_loss(disc_out[1]) for disc_out in disc_outputs)
        loss_distill = d_axis_distill_loss(out["semantic_feature"], semantic_feature)
        loss_commit = out["commitment"]
        loss_balance = branch_energy_balance_loss(out["semantic_quantized"], out["acoustic_quantized"])

        total = (
            loss_feature
            + loss_adversarial
            + loss_mel
            + self.args.commitment_loss_lambda * loss_commit
            + self.args.recon_loss_lambda * loss_recon
            + self.args.distill_loss_lambda * loss_distill
            + self.args.balance_loss_lambda * loss_balance
        )
        stats = {
            "recon": float(loss_recon.item()),
            "mel": float(loss_mel.item()),
            "feature": float(loss_feature.item()),
            "adv": float(loss_adversarial.item()),
            "distill": float(loss_distill.item()),
            "commit": float(loss_commit.item()),
            "balance": float(loss_balance.item()),
        }
        return total, stats, x_hat

    def _stage1_discriminator_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        disc_outputs = [disc(x, x_hat.detach()) for disc in self.discriminators.values()]
        return sum(discriminator_loss(disc_out[0], disc_out[1]) for disc_out in disc_outputs)

    def _transmit_codes(self, codes: torch.Tensor, codebook_sizes: tuple[int, ...]) -> dict[str, torch.Tensor]:
        rx_codes = []
        avg_conf = []
        for idx, codebook_size in enumerate(codebook_sizes):
            bits_per_index = codebook_size.bit_length() - 1
            flat = codes[idx].reshape(-1)
            tx = self.channel.transmit_indices(flat, bits_per_index=bits_per_index)
            rx_layer = tx["hard_indices"].reshape(codes.shape[1], codes.shape[2])
            conf_layer = tx["confidence"].mean(dim=-1).reshape(codes.shape[1], codes.shape[2])
            rx_codes.append(rx_layer)
            avg_conf.append(conf_layer)
        return {
            "rx_codes": torch.stack(rx_codes, dim=0),
            "confidence": torch.stack(avg_conf, dim=0),
        }

    def _branch_detector_logits(
        self,
        rx_codes: torch.Tensor,
        confidence: torch.Tensor,
        branch_decode_fn,
        parent_modules,
        parent_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_per_layer = []
        parent_scores = []
        branch_latents = []
        for idx, parent_module in enumerate(parent_modules):
            layer_codes = rx_codes[idx : idx + 1]
            branch_latent = branch_decode_fn(layer_codes, st=idx)
            layer_parent_ids = parent_ids[idx]
            parent_score = parent_module.consistency_score(branch_latent, layer_parent_ids)
            parent_latent = parent_module.decode(layer_parent_ids)
            parent_similarity = F.cosine_similarity(
                branch_latent.permute(0, 2, 1),
                parent_latent.permute(0, 2, 1),
                dim=-1,
            )
            code_norm = branch_latent.permute(0, 2, 1).norm(dim=-1)
            conf_frame = confidence[idx]
            logits = self.detector(conf_frame, parent_score, parent_similarity, code_norm)
            logits_per_layer.append(logits)
            parent_scores.append(parent_score)
            branch_latents.append(branch_latent)
        return (
            torch.stack(logits_per_layer, dim=0),
            torch.stack(parent_scores, dim=0),
            torch.stack(branch_latents, dim=0),
        )

    def _stage2_loss(self, wavs: torch.Tensor, ref_wavs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        out = self.model(wavs)
        sem_tx = self._transmit_codes(out["semantic_codes"], self.model_cfg.semantic.codebook_sizes)
        ac_tx = self._transmit_codes(out["acoustic_codes"], self.model_cfg.acoustic.codebook_sizes)

        sem_logits, sem_parent_score, sem_latent_rx = self._branch_detector_logits(
            sem_tx["rx_codes"],
            sem_tx["confidence"],
            self.model.semantic_branch.decode,
            self.model.semantic_parents,
            out["semantic_parent_ids"],
        )
        ac_logits, ac_parent_score, ac_latent_rx = self._branch_detector_logits(
            ac_tx["rx_codes"],
            ac_tx["confidence"],
            self.model.acoustic_branch.decode,
            self.model.acoustic_parents,
            out["acoustic_parent_ids"],
        )

        sem_mask = torch.sigmoid(sem_logits) > 0.5
        ac_mask = torch.sigmoid(ac_logits) > 0.5

        _q_sem, _q_ac, latent_corr = self.model.reconstruct_latent_with_parent_fallback(
            semantic_codes=sem_tx["rx_codes"],
            acoustic_codes=ac_tx["rx_codes"],
            semantic_parent_ids=out["semantic_parent_ids"],
            acoustic_parent_ids=out["acoustic_parent_ids"],
            semantic_parent_mask=sem_mask,
            acoustic_parent_mask=ac_mask,
        )

        wav_corr = self.model.decoder(latent_corr)
        rec_corr = waveform_l1(wav_corr, wavs)
        mel_corr = 0.0
        wavs_ch = wavs
        for lam, kwargs in zip(self.mel_loss_lambdas, self.mel_kwargs_list):
            mel_corr = mel_corr + lam * mel_loss(wavs_ch, wav_corr, **kwargs)

        parent_codebook_loss = (
            F.mse_loss(out["semantic_parent_latent"], out["semantic_layer_quantized"].detach())
            + F.mse_loss(out["acoustic_parent_latent"], out["acoustic_layer_quantized"].detach())
        )

        spk_emb = self.spk_encoder(ref_wavs.squeeze(1))
        latent_target = out["full_latent"].detach()
        t = torch.rand(latent_corr.shape[0], device=self.device)
        xt = (1.0 - t[:, None, None]) * latent_corr + t[:, None, None] * latent_target
        target_v = latent_target - latent_corr
        pred_v = self.flow_model(xt, t, spk_emb)
        flow_loss = F.mse_loss(pred_v, target_v)

        latent_restored = flow_sample(self.flow_model, latent_corr, spk_emb, n_steps=self.args.flow_steps)
        wav_restored = self.model.decoder(latent_restored)
        rec_flow = waveform_l1(wav_restored, wavs)
        mel_flow = 0.0
        for lam, kwargs in zip(self.mel_loss_lambdas, self.mel_kwargs_list):
            mel_flow = mel_flow + lam * mel_loss(wavs_ch, wav_restored, **kwargs)

        sem_parent_target = (sem_parent_score < self.args.parent_consistency_target).float()
        ac_parent_target = (ac_parent_score < self.args.parent_consistency_target).float()
        det_loss = F.binary_cross_entropy_with_logits(sem_logits, sem_parent_target) + F.binary_cross_entropy_with_logits(ac_logits, ac_parent_target)

        parent_cons = parent_consistency_loss(sem_parent_score) + parent_consistency_loss(ac_parent_score)
        conf_pen = confidence_penalty(sem_tx["confidence"]) + confidence_penalty(ac_tx["confidence"])

        total = (
            self.args.lambda_stage2_corr * rec_corr
            + self.args.lambda_stage2_mel_corr * mel_corr
            + self.args.lambda_stage2_flow * flow_loss
            + self.args.lambda_stage2_wave * rec_flow
            + self.args.lambda_stage2_mel_flow * mel_flow
            + self.args.lambda_stage2_parent * parent_cons
            + self.args.lambda_stage2_parent_codebook * parent_codebook_loss
            + self.args.lambda_stage2_conf * conf_pen
            + self.args.lambda_stage2_det * det_loss
        )
        return total, {
            "rec_corr": float(rec_corr.item()),
            "mel_corr": float(mel_corr.item()),
            "rec_flow": float(rec_flow.item()),
            "mel_flow": float(mel_flow.item()),
            "flow_loss": float(flow_loss.item()),
            "parent_cons": float(parent_cons.item()),
            "parent_codebook": float(parent_codebook_loss.item()),
            "det_loss": float(det_loss.item()),
            "sem_replace_ratio": float(sem_mask.float().mean().item()),
            "ac_replace_ratio": float(ac_mask.float().mean().item()),
        }

    def train(self) -> None:
        self.model.train()
        if self.flow_model is not None:
            self.flow_model.train()
        if self.spk_encoder is not None:
            self.spk_encoder.train()
        os.makedirs(self.args.save_dir, exist_ok=True)

        if self.stage.name == "stage1_ed_fine":
            loader = self.codec_loader
        else:
            loader = self.channel_loader

        for epoch in range(self.args.max_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(loader):
                if self.stage.name == "stage1_ed_fine":
                    x, semantic_feature = batch
                    x = x.to(self.device)
                    semantic_feature = semantic_feature.to(self.device)
                    gen_loss, stats, x_hat = self._stage1_generator_loss(x, semantic_feature)
                    self.optim_d.zero_grad()
                    d_loss = self._stage1_discriminator_loss(x, x_hat)
                    d_loss.backward()
                    self.optim_d.step()
                    self.optim_g.zero_grad()
                    gen_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._generator_parameters(), max_norm=1.0)
                    self.optim_g.step()
                    loss = gen_loss
                    stats["disc"] = float(d_loss.item())
                else:
                    wavs, ref_wavs = batch
                    wavs = wavs.to(self.device)
                    ref_wavs = ref_wavs.to(self.device)
                    self.optim_g.zero_grad()
                    loss, stats = self._stage2_loss(wavs, ref_wavs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._generator_parameters(), max_norm=1.0)
                    self.optim_g.step()

                epoch_loss += float(loss.item())
                if step % self.args.log_every == 0:
                    stat_str = " ".join(f"{k}={v:.4f}" for k, v in stats.items())
                    print(
                        f"[{self.stage.name}] epoch={epoch} step={step}/{len(loader)} "
                        f"loss={loss.item():.4f} {stat_str}",
                        flush=True,
                    )
                if step + 1 >= self.args.max_steps_per_epoch:
                    break

            avg_loss = epoch_loss / max(1, min(len(loader), self.args.max_steps_per_epoch))
            ckpt = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optim_g": self.optim_g.state_dict(),
                "stage": self.stage.name,
                "avg_loss": avg_loss,
                "args": vars(self.args),
                "model_cfg": asdict(self.model_cfg),
            }
            if self.optim_d is not None:
                ckpt["optim_d"] = self.optim_d.state_dict()
                ckpt["discriminators"] = {k: v.state_dict() for k, v in self.discriminators.items()}
            if self.detector is not None:
                ckpt["detector"] = self.detector.state_dict()
            if self.flow_model is not None:
                ckpt["flow_model"] = self.flow_model.state_dict()
            if self.spk_encoder is not None:
                ckpt["spk_encoder"] = self.spk_encoder.state_dict()
            torch.save(ckpt, os.path.join(self.args.save_dir, f"{self.stage.name}_last.pt"))
            print(f"[{self.stage.name}] epoch={epoch} avg_loss={avg_loss:.4f}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual-RVQ two-stage trainer")
    p.add_argument("--stage", default="stage1_ed_fine", choices=[s.name for s in default_stage_configs()])
    p.add_argument("--data_dir", default="data")
    p.add_argument("--split", default="train-clean-100")
    p.add_argument("--train_files", default="train_file_list.txt")
    p.add_argument("--save_dir", default="dual_rvq_redesign/output")
    p.add_argument("--segment_sec", type=float, default=2.0)
    p.add_argument("--segment_size", type=int, default=48000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--max_steps_per_epoch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--ebno_db", type=float, default=6.0)

    p.add_argument("--num_mels", type=int, default=80)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_size", type=int, default=240)
    p.add_argument("--win_size", type=int, default=1024)
    p.add_argument("--fmin", type=int, default=0)
    p.add_argument("--fmax_for_loss", type=int, default=8000)
    p.add_argument("--mel_loss_lambdas", type=float, nargs="+", default=[45.0, 1.0, 1.0, 1.0])
    p.add_argument("--recon_loss_lambda", type=float, default=500.0)
    p.add_argument("--commitment_loss_lambda", type=float, default=10.0)
    p.add_argument("--distill_loss_lambda", type=float, default=120.0)
    p.add_argument("--balance_loss_lambda", type=float, default=1.0)

    p.add_argument("--detector_hidden_dim", type=int, default=64)
    p.add_argument("--parent_consistency_target", type=float, default=0.4)
    p.add_argument("--spk_dim", type=int, default=256)
    p.add_argument("--flow_base_ch", type=int, default=256)
    p.add_argument("--flow_ch_mults", type=int, nargs="+", default=[1, 1, 2])
    p.add_argument("--flow_cond_dim", type=int, default=256)
    p.add_argument("--flow_time_dim", type=int, default=128)
    p.add_argument("--flow_n_res", type=int, default=2)
    p.add_argument("--flow_n_mid_res", type=int, default=2)
    p.add_argument("--flow_steps", type=int, default=6)

    p.add_argument("--lambda_stage2_corr", type=float, default=1.0)
    p.add_argument("--lambda_stage2_mel_corr", type=float, default=1.0)
    p.add_argument("--lambda_stage2_parent", type=float, default=0.1)
    p.add_argument("--lambda_stage2_parent_codebook", type=float, default=1.0)
    p.add_argument("--lambda_stage2_conf", type=float, default=0.01)
    p.add_argument("--lambda_stage2_flow", type=float, default=1.0)
    p.add_argument("--lambda_stage2_wave", type=float, default=1.0)
    p.add_argument("--lambda_stage2_mel_flow", type=float, default=1.0)
    p.add_argument("--lambda_stage2_det", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    trainer = DualRVQTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
