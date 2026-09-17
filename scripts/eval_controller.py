# -*- coding: utf-8 -*-
"""
Evaluate the trained bitrate controller.

For each (PLR, Rmax) operating point, this script:
  1) computes candidate reconstruction MSE for all N
  2) applies the controller to select one feasible N
  3) reports average selected bitrate, MSE, oracle MSE, and selection accuracy
"""

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speechtokenizer.model import SpeechTokenizer
from speechtokenizer.flow import BitrateController, FlowMatchingModel, PretrainedSpeakerEncoder
from speechtokenizer.flow.dataset import LibriSpeechFlowDataset
from scripts.eval_utils import _latent_linear_interp, flow_sample


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    wavs = torch.stack([item[0] for item in batch], dim=0)
    ref_wavs = torch.stack([item[1] for item in batch], dim=0)
    return wavs, ref_wavs


def extract_obs_features(latent: torch.Tensor) -> torch.Tensor:
    x = latent.squeeze(0).float()
    frame_energy = x.pow(2).mean(dim=0).sqrt()
    frame_var = x.var(dim=0, unbiased=False)
    if x.shape[-1] > 1:
        delta = (x[:, 1:] - x[:, :-1]).abs().mean()
    else:
        delta = x.new_tensor(0.0)
    feat = torch.stack([
        x.abs().mean(),
        x.std(unbiased=False),
        x.abs().amax(),
        frame_energy.mean(),
        frame_var.mean(),
        delta,
    ], dim=0)
    return feat


@torch.no_grad()
def build_candidate_latents(st_model, wav: torch.Tensor, candidate_layers, p_loss: float, device):
    x = wav.unsqueeze(0).to(device)
    codes_all = st_model.encode(x)
    t_enc = codes_all.shape[2]
    dim = st_model.quantizer.dimension

    def safe_decode(idx: int) -> torch.Tensor:
        vq_l = st_model.quantizer.vq.layers[idx]
        dec = vq_l.decode(codes_all[idx])
        if dec.shape[-1] == dim:
            dec = dec.permute(0, 2, 1)
        return dec.contiguous()

    decoded_layers = [safe_decode(i) for i in range(max(candidate_layers))]

    q1_recv = (torch.rand(t_enc, device=device) >= p_loss)
    latent_q1 = decoded_layers[0]
    if not bool(q1_recv.all()):
        latent_q1 = _latent_linear_interp(latent_q1, q1_recv)

    latent_running = latent_q1.clone()
    latent_by_n = {1: latent_running.clone()}
    for l in range(1, max(candidate_layers)):
        recv_mask = (torch.rand(t_enc, device=device) >= p_loss).float()
        latent_running = latent_running + decoded_layers[l] * recv_mask.unsqueeze(0).unsqueeze(0)
        latent_by_n[l + 1] = latent_running.clone()
    return latent_by_n


def build_backbones(args, device):
    st_model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
    st_model.eval().to(device)
    for p in st_model.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(args.flow_ckpt, map_location="cpu")
    saved_args = argparse.Namespace(**ckpt.get("args", {}))
    flow_model = FlowMatchingModel(
        latent_dim=st_model.quantizer.dimension,
        base_ch=getattr(saved_args, "base_ch", 512),
        ch_mults=tuple(getattr(saved_args, "ch_mults", [1, 1, 2])),
        cond_dim=getattr(saved_args, "cond_dim", 512),
        spk_dim=getattr(saved_args, "spk_dim", 256),
        time_dim=getattr(saved_args, "time_dim", 128),
        n_res=getattr(saved_args, "n_res", 2),
        n_mid_res=getattr(saved_args, "n_mid_res", 2),
    ).to(device)
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["flow_model"].items()})
    flow_model.eval()
    for p in flow_model.parameters():
        p.requires_grad_(False)

    spk_encoder = PretrainedSpeakerEncoder(
        emb_dim=getattr(saved_args, "spk_dim", 256),
        save_dir=os.path.join(os.path.dirname(args.flow_ckpt), "spkrec-ecapa"),
    ).to(device)
    spk_encoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["spk_encoder"].items()}, strict=False)
    spk_encoder.eval()
    for p in spk_encoder.parameters():
        p.requires_grad_(False)

    return st_model, flow_model, spk_encoder


def build_controller(args, device):
    ckpt = torch.load(args.controller_ckpt, map_location="cpu")
    saved_args = argparse.Namespace(**ckpt.get("args", {}))
    hidden_dim = getattr(saved_args, "hidden_dim", args.hidden_dim)
    obs_feat_dim = getattr(saved_args, "obs_feat_dim", args.obs_feat_dim)
    candidate_layers = list(getattr(saved_args, "n_layers_list", args.n_layers_list))
    controller = BitrateController(
        num_actions=len(candidate_layers),
        hidden_dim=hidden_dim,
        in_dim=2 + obs_feat_dim,
    ).to(device)
    controller.load_state_dict(ckpt["controller"], strict=True)
    controller.eval()
    return controller, candidate_layers


@torch.no_grad()
def evaluate_op_point(st_model, flow_model, spk_encoder, controller, candidate_layers, loader, plr_value, rmax_value, args, device):
    rate_vec = torch.tensor([n * 0.5 for n in candidate_layers], device=device, dtype=torch.float32)
    total = 0
    sum_mse = 0.0
    sum_oracle_mse = 0.0
    sum_rate = 0.0
    sum_oracle_rate = 0.0
    sum_acc = 0.0
    pred_hist = torch.zeros(len(candidate_layers), dtype=torch.long)

    for batch_idx, (wavs, ref_wavs) in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        wavs = wavs.to(device)
        ref_wavs = ref_wavs.to(device)
        plr = torch.full((wavs.size(0),), float(plr_value), device=device)
        r_max = torch.full((wavs.size(0),), float(rmax_value), device=device)

        spk_emb = spk_encoder(ref_wavs.squeeze(1))

        for b in range(wavs.size(0)):
            wav_b = wavs[b : b + 1]
            spk_b = spk_emb[b : b + 1]
            plr_b = float(plr[b].item())
            latent_by_n = build_candidate_latents(
                st_model,
                wav_b.squeeze(0),
                candidate_layers=candidate_layers,
                p_loss=plr_b,
                device=device,
            )
            obs_feat = extract_obs_features(latent_by_n[candidate_layers[0]]).unsqueeze(0)
            score = controller(plr[b : b + 1].unsqueeze(-1), r_max=r_max[b : b + 1].unsqueeze(-1), obs_feat=obs_feat)

            cand_mse = []
            for n_layers in candidate_layers:
                lat_in = latent_by_n[n_layers]
                lat_8 = flow_sample(flow_model, lat_in, spk_b, n_steps=args.n_steps, n_layers=n_layers)
                lat_8 = lat_8[..., :lat_in.shape[-1]]
                wav_rec = st_model.decoder(lat_8).squeeze(1)
                wav_ref = wav_b.squeeze(1)
                T = min(wav_rec.shape[-1], wav_ref.shape[-1])
                wav_rec = wav_rec[..., :T]
                wav_ref = wav_ref[..., :T]
                cand_mse.append(F.mse_loss(wav_rec, wav_ref, reduction="mean"))
            dist_mat = torch.stack(cand_mse, dim=0)

            feasible = rate_vec <= (r_max[b].item() + 1e-6)
            if feasible.any():
                masked_score = score.squeeze(0).masked_fill(~feasible, -1e9)
                pred_idx = int(masked_score.argmax(dim=0).item())
                oracle_idx = int(torch.where(feasible, dist_mat, torch.full_like(dist_mat, float("inf"))).argmin().item())
            else:
                pred_idx = int(rate_vec.argmin().item())
                oracle_idx = pred_idx

            sum_mse += float(dist_mat[pred_idx].item())
            sum_oracle_mse += float(dist_mat[oracle_idx].item())
            sum_rate += float(rate_vec[pred_idx].item())
            sum_oracle_rate += float(rate_vec[oracle_idx].item())
            sum_acc += float(pred_idx == oracle_idx)
            pred_hist[pred_idx] += 1
            total += 1

    return {
        "plr": float(plr_value),
        "rmax": float(rmax_value),
        "num_samples": int(total),
        "avg_mse": sum_mse / max(total, 1),
        "avg_oracle_mse": sum_oracle_mse / max(total, 1),
        "avg_rate": sum_rate / max(total, 1),
        "avg_oracle_rate": sum_oracle_rate / max(total, 1),
        "selection_acc": sum_acc / max(total, 1),
        "pred_hist": {f"N{candidate_layers[i]}": int(pred_hist[i].item()) for i in range(len(candidate_layers))},
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained bitrate controller")
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path", default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--flow_ckpt", default="output/flow_checkpoints_stage3/best_train.pt")
    p.add_argument("--controller_ckpt", default="output/controller_checkpoints/best_controller.pt")
    p.add_argument("--data_dir", default="LibriSpeech")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--save_dir", default="output/controller_eval")
    p.add_argument("--n_layers_list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--segment_sec", type=float, default=4.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--obs_feat_dim", type=int, default=6)
    p.add_argument("--plr_values", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    p.add_argument("--rmax_values", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    st_model, flow_model, spk_encoder = build_backbones(args, device)
    controller, candidate_layers = build_controller(args, device)

    dataset = LibriSpeechFlowDataset(
        data_dir=args.data_dir,
        split=args.split,
        segment_sec=args.segment_sec,
        sample_rate=st_model.sample_rate,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    results = []
    for plr in args.plr_values:
        for rmax in args.rmax_values:
            logger.info("evaluating plr=%.3f rmax=%.3f", plr, rmax)
            row = evaluate_op_point(
                st_model, flow_model, spk_encoder, controller, candidate_layers,
                loader, plr, rmax, args, device
            )
            results.append(row)
            logger.info(
                "plr=%.3f rmax=%.3f mse=%.6f oracle_mse=%.6f rate=%.3f oracle_rate=%.3f acc=%.3f",
                row["plr"], row["rmax"], row["avg_mse"], row["avg_oracle_mse"],
                row["avg_rate"], row["avg_oracle_rate"], row["selection_acc"]
            )

    out_path = os.path.join(args.save_dir, "controller_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("saved to %s", out_path)


if __name__ == "__main__":
    main()
