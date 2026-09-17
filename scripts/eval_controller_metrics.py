# -*- coding: utf-8 -*-
"""
Evaluate the trained controller with speech quality metrics and ablations.

Outputs:
  - VISQoL
  - PLCMOS
  - UTMOS (optional)
  - WER (optional)
  - ablation groups for controller variants
"""

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speechtokenizer.model import SpeechTokenizer
from speechtokenizer.flow import BitrateController, FlowMatchingModel, PretrainedSpeakerEncoder
from speechtokenizer.flow.dataset import LibriSpeechFlowDataset
from scripts.eval_utils import (
    _latent_linear_interp,
    flow_sample,
    calc_visqol_batch,
    calc_utmos,
    calc_plcmos,
    calc_wer,
    load_transcript,
)

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
    delta = (x[:, 1:] - x[:, :-1]).abs().mean() if x.shape[-1] > 1 else x.new_tensor(0.0)
    return torch.stack([
        x.abs().mean(),
        x.std(unbiased=False),
        x.abs().amax(),
        frame_energy.mean(),
        frame_var.mean(),
        delta,
    ], dim=0)


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
    candidate_layers = list(getattr(saved_args, "n_layers_list", args.n_layers_list))
    controller = BitrateController(
        num_actions=len(candidate_layers),
        hidden_dim=getattr(saved_args, "hidden_dim", args.hidden_dim),
        in_dim=2 + getattr(saved_args, "obs_feat_dim", args.obs_feat_dim),
    ).to(device)
    controller.load_state_dict(ckpt["controller"], strict=True)
    controller.eval()
    return controller, candidate_layers


@torch.no_grad()
def decode_sample(
    st_model,
    flow_model,
    spk_encoder,
    wav,
    ref_wav,
    candidate_layers,
    plr,
    rmax,
    controller,
    device,
    n_steps,
    use_controller: bool = True,
    use_flow: bool = True,
    fixed_n: int | None = None,
):
    latent_by_n = build_candidate_latents(st_model, wav, candidate_layers, plr, device)
    if use_controller:
        obs_feat = extract_obs_features(latent_by_n[candidate_layers[0]]).unsqueeze(0)
        score = controller(
            torch.tensor([plr], device=device),
            r_max=torch.tensor([rmax], device=device),
            obs_feat=obs_feat,
        )
        rate_vec = torch.tensor([n * 0.5 for n in candidate_layers], device=device, dtype=torch.float32)
        feasible = rate_vec <= (rmax + 1e-6)
        if feasible.any():
            pred_idx = int(score.squeeze(0).masked_fill(~feasible, -1e9).argmax().item())
        else:
            pred_idx = 0
        n_layers = candidate_layers[pred_idx]
    else:
        if fixed_n is None:
            raise ValueError("fixed_n must be provided when use_controller=False")
        n_layers = int(fixed_n)
        if n_layers not in candidate_layers:
            raise ValueError(f"fixed_n={fixed_n} not in candidate_layers={candidate_layers}")

    lat_in = latent_by_n[n_layers]
    if use_flow:
        spk_emb = spk_encoder(ref_wav.unsqueeze(0).to(device).squeeze(1))
        lat_out = flow_sample(flow_model, lat_in, spk_emb, n_steps=n_steps, n_layers=n_layers)
        lat_out = lat_out[..., :lat_in.shape[-1]]
        wav_rec = st_model.decoder(lat_out).squeeze(1).detach().cpu().numpy()
    else:
        wav_rec = st_model.decoder(lat_in).squeeze(1).detach().cpu().numpy()
    return wav_rec, n_layers


def evaluate_group(
    args,
    group_name,
    st_model,
    flow_model,
    spk_encoder,
    controller,
    candidate_layers,
    dataset,
    device,
    use_controller: bool = True,
    use_flow: bool = True,
    fixed_n: int | None = None,
):
    rows = []
    pred_rates = []
    pred_layers = []
    ref_list = []
    deg_list = []
    wer_list = []

    for idx in range(min(len(dataset), args.max_items if args.max_items > 0 else len(dataset))):
        wav, ref_wav = dataset[idx]
        _, fpath = dataset.pairs[idx]
        plr = float(random.choice(args.plr_values))
        rmax = float(random.choice(args.rmax_values))
        deg_np, n_layers = decode_sample(
            st_model, flow_model, spk_encoder, wav, ref_wav, candidate_layers, plr, rmax,
            controller, device, args.n_steps,
            use_controller=use_controller,
            use_flow=use_flow,
            fixed_n=fixed_n,
        )
        ref_np = wav.squeeze(0).numpy().astype(np.float64)
        deg_np = deg_np.astype(np.float64)
        ref_list.append(ref_np)
        deg_list.append(deg_np)
        pred_rates.append(n_layers * 0.5)
        pred_layers.append(n_layers)
        ref_text = load_transcript(fpath)
        row = {
            "file": os.path.basename(fpath),
            "plr": plr,
            "rmax": rmax,
            "n": n_layers,
            "rate_kbps": n_layers * 0.5,
            "utmos": calc_utmos(deg_np, st_model.sample_rate),
            "plcmos": calc_plcmos(deg_np, ref_np, st_model.sample_rate),
        }
        if not args.no_wer and ref_text:
            row["wer"] = calc_wer(deg_np, ref_text, st_model.sample_rate, model_size=args.whisper_model)
            wer_list.append(row["wer"])
        rows.append(row)

    visqol_list = calc_visqol_batch(
        ref_list, deg_list, st_model.sample_rate, n_workers=args.visqol_workers, mode=args.visqol_mode
    ) if not args.no_visqol else [float("nan")] * len(ref_list)
    for r, v in zip(rows, visqol_list):
        r["visqol"] = v

    summary = {
        "group": group_name,
        "mode": "controller" if use_controller else f"fixed_N{fixed_n}",
        "use_flow": bool(use_flow),
        "num_items": len(rows),
        "avg_rate_kbps": float(np.mean(pred_rates)) if pred_rates else float("nan"),
        "avg_visqol": float(np.nanmean(visqol_list)) if visqol_list else float("nan"),
        "avg_plcmos": float(np.nanmean([r["plcmos"] for r in rows])) if rows else float("nan"),
        "avg_utmos": float(np.nanmean([r["utmos"] for r in rows])) if rows else float("nan"),
        "avg_wer": float(np.nanmean(wer_list)) if wer_list else float("nan"),
        "layer_hist": {f"N{n}": int(sum(1 for x in pred_layers if x == n)) for n in candidate_layers},
        "layer_prob": {f"N{n}": float(sum(1 for x in pred_layers if x == n) / max(len(pred_layers), 1)) for n in candidate_layers},
        "items": rows,
    }
    return summary


def save_selection_heatmap(groups, candidate_layers, output_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("matplotlib unavailable, skip heatmap: %s", e)
        return

    if not groups:
        return

    groups = [g for g in groups if g.get("mode") == "controller" and g.get("use_flow", True)]
    labels = [f"PLR={g['group'].split('_')[0][3:]}, Rmax={g['group'].split('_')[1][4:]}" for g in groups]
    mat = np.array([
        [g["layer_prob"][f"N{n}"] for n in candidate_layers]
        for g in groups
    ], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(1.3 * len(candidate_layers) + 2.2, max(4.0, 0.45 * len(groups) + 2.2)))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=max(1e-6, float(mat.max())))
    ax.set_xticks(range(len(candidate_layers)))
    ax.set_xticklabels([f"N{n}" for n in candidate_layers])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Selected layer N")
    ax.set_ylabel("Operating point")
    ax.set_title("Controller selection distribution")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Selection probability")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "controller_selection_heatmap.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved heatmap to %s", out_path)


def parse_args():
    p = argparse.ArgumentParser(description="Controller metric evaluation")
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path", default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--flow_ckpt", default="output/flow_checkpoints_stage3/best_train.pt")
    p.add_argument("--controller_ckpt", default="output/controller_checkpoints/best_controller.pt")
    p.add_argument("--data_dir", default="LibriSpeech")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--output_dir", default="output/controller_metrics")
    p.add_argument("--n_layers_list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--plr_values", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    p.add_argument("--rmax_values", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--visqol_workers", type=int, default=4)
    p.add_argument("--visqol_mode", default="speech")
    p.add_argument("--no_visqol", action="store_true")
    p.add_argument("--no_wer", action="store_true")
    p.add_argument("--run_ablation", action="store_true")
    p.add_argument("--fixed_n_list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--whisper_model", default="base")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    st_model, flow_model, spk_encoder = build_backbones(args, device)
    controller, candidate_layers = build_controller(args, device)

    dataset = LibriSpeechFlowDataset(
        data_dir=args.data_dir,
        split=args.split,
        segment_sec=4.0,
        sample_rate=st_model.sample_rate,
    )

    groups = []
    ablations = []
    for plr in args.plr_values:
        for rmax in args.rmax_values:
            group_name = f"plr{plr:.2f}_rmax{rmax:.1f}"
            logger.info("evaluating %s", group_name)
            summary = evaluate_group(
                args, group_name, st_model, flow_model, spk_encoder, controller, candidate_layers, dataset, device
            )
            groups.append(summary)
            logger.info(
                "%s rate=%.3f visqol=%.3f plcmos=%.3f utmos=%.3f wer=%.3f",
                group_name,
                summary["avg_rate_kbps"],
                summary["avg_visqol"],
                summary["avg_plcmos"],
                summary["avg_utmos"],
                summary["avg_wer"],
            )
            if args.run_ablation:
                ablations.append(evaluate_group(
                    args, group_name, st_model, flow_model, spk_encoder, controller, candidate_layers, dataset, device,
                    use_controller=True, use_flow=False
                ))
                for fixed_n in args.fixed_n_list:
                    ablations.append(evaluate_group(
                        args, f"{group_name}_fixedN{fixed_n}_flow", st_model, flow_model, spk_encoder, controller,
                        candidate_layers, dataset, device, use_controller=False, use_flow=True, fixed_n=fixed_n
                    ))
                    ablations.append(evaluate_group(
                        args, f"{group_name}_fixedN{fixed_n}_noflow", st_model, flow_model, spk_encoder, controller,
                        candidate_layers, dataset, device, use_controller=False, use_flow=False, fixed_n=fixed_n
                    ))

    out_json = os.path.join(args.output_dir, "controller_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)
    if args.run_ablation:
        ablation_json = os.path.join(args.output_dir, "controller_ablations.json")
        with open(ablation_json, "w", encoding="utf-8") as f:
            json.dump(ablations, f, indent=2, ensure_ascii=False)
        logger.info("saved ablations to %s", ablation_json)
    save_selection_heatmap(groups, candidate_layers, args.output_dir)
    logger.info("saved to %s", out_json)


if __name__ == "__main__":
    main()
