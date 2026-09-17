# -*- coding: utf-8 -*-
"""
System-level evaluation for proposed controller vs baselines.

Outputs:
  - proposed system metrics with trained controller
  - baseline codec comparisons
  - ablation results
  - selection distribution heatmap
  - per-part CSV files
"""

import argparse
import csv
import json
import logging
import os
import random
from typing import Dict, List

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
    calc_visqol,
    calc_visqol_batch,
    calc_utmos,
    calc_plcmos,
    calc_wer,
    load_transcript,
    load_audio,
    build_spk2files,
    pick_ref_wav,
    channel_simulate,
    encodec_with_plr,
    encodec_with_lfrplc,
    ffmpeg_codec_with_plr,
    opus_lbrr_with_plr,
    HAS_ENCODEC,
    nanmean,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


VISQOL_OURS_SR = 48000
VISQOL_OURS_MODE = "audio"
VISQOL_COMP_SR = 16000
VISQOL_COMP_MODE = "speech"

PLR_LIST = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

BASELINE_RATE_COMPETITORS = {
    "AAC": {"type": "ffmpeg", "codec": "aac", "bitrates": [4, 6, 8, 12, 16, 20]},
    "Opus": {"type": "ffmpeg", "codec": "libopus", "bitrates": [6.0, 8.0, 12.0, 16.0]},
    "EnCodec": {"type": "encodec", "bitrates": [1.5, 3.0, 6.0]},
}

BASELINE_PLR_SYSTEMS = {
    "ours": {"label": "Proposed", "type": "ours", "n_layers": None},
    "encodec_1.5k": {"label": "EnCodec (1.5kbps)", "type": "encodec", "bw": 1.5},
    "encodec_3.0k": {"label": "EnCodec (3.0kbps)", "type": "encodec", "bw": 3.0},
    "opus_8k": {"label": "Opus (~8kbps)", "type": "opus", "bw": 8.0},
}


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


def save_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def decode_with_controller(st_model, flow_model, spk_encoder, controller, candidate_layers, wav, ref_wav,
                           plr, rmax, device, n_steps):
    latent_by_n = build_candidate_latents(st_model, wav, candidate_layers, p_loss=plr, device=device)
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
    lat_in = latent_by_n[n_layers]
    spk_emb = spk_encoder(ref_wav.unsqueeze(0).to(device).squeeze(1))
    lat_out = flow_sample(flow_model, lat_in, spk_emb, n_steps=n_steps, n_layers=n_layers)
    lat_out = lat_out[..., :lat_in.shape[-1]]
    wav_rec = st_model.decoder(lat_out).squeeze(1).detach().cpu().numpy()
    return wav_rec.astype(np.float64), n_layers, float(rate_vec[pred_idx].item())


@torch.no_grad()
def decode_fixed_n(st_model, flow_model, spk_encoder, wav, ref_wav, candidate_layers, n_layers, plr, device, n_steps):
    latent_by_n = build_candidate_latents(st_model, wav, candidate_layers, p_loss=plr, device=device)
    lat_in = latent_by_n[n_layers]
    spk_emb = spk_encoder(ref_wav.unsqueeze(0).to(device).squeeze(1))
    lat_out = flow_sample(flow_model, lat_in, spk_emb, n_steps=n_steps, n_layers=n_layers)
    lat_out = lat_out[..., :lat_in.shape[-1]]
    wav_rec = st_model.decoder(lat_out).squeeze(1).detach().cpu().numpy()
    return wav_rec.astype(np.float64)


def _safe_metric(value):
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def eval_proposed_system(args, st_model, flow_model, spk_encoder, controller, candidate_layers, dataset, device, out_dir):
    rows = []
    sr = st_model.sample_rate
    total = min(len(dataset), args.max_items if args.max_items > 0 else len(dataset))
    for idx in range(total):
        wav, ref_wav = dataset[idx]
        _, fpath = dataset.pairs[idx]
        plr = float(random.choice(args.plr_values))
        rmax = float(random.choice(args.rmax_values))
        deg_np, n_layers, rate = decode_with_controller(
            st_model, flow_model, spk_encoder, controller, candidate_layers,
            wav, ref_wav, plr, rmax, device, args.n_steps
        )
        ref_np = wav.squeeze(0).numpy().astype(np.float64)
        ref_text = load_transcript(fpath)
        row = {
            "file": os.path.basename(fpath),
            "plr": plr,
            "rmax": rmax,
            "n_layers": n_layers,
            "rate_kbps": rate,
            "visqol": _safe_metric(calc_visqol(ref_np, deg_np, sr, mode=VISQOL_OURS_MODE)) if not args.no_visqol else float("nan"),
            "plcmos": _safe_metric(calc_plcmos(deg_np, ref_np, sr)),
            "utmos": _safe_metric(calc_utmos(deg_np, sr)),
            "wer": _safe_metric(calc_wer(deg_np, ref_text, sr, model_size=args.whisper_model)) if not args.no_wer and ref_text else float("nan"),
        }
        rows.append(row)
    return rows


def summarize_rows(rows: List[Dict]) -> Dict[str, float]:
    return {
        "num_samples": len(rows),
        "avg_rate_kbps": float(np.nanmean([r["rate_kbps"] for r in rows])) if rows else float("nan"),
        "avg_visqol": float(np.nanmean([r["visqol"] for r in rows])) if rows else float("nan"),
        "avg_plcmos": float(np.nanmean([r["plcmos"] for r in rows])) if rows else float("nan"),
        "avg_utmos": float(np.nanmean([r["utmos"] for r in rows])) if rows else float("nan"),
        "avg_wer": float(np.nanmean([r["wer"] for r in rows])) if rows else float("nan"),
        "layer_hist": {f"N{n}": int(sum(1 for r in rows if r["n_layers"] == n)) for n in sorted(set(r["n_layers"] for r in rows))} if rows else {},
    }


def eval_baselines(args, st_model, flow_model, spk_encoder, dataset, device, out_dir):
    sr = st_model.sample_rate
    rows = {}
    selected_indices = list(range(min(len(dataset), args.max_items if args.max_items > 0 else len(dataset))))

    def collect(decode_fn, label):
        out = []
        for idx in selected_indices:
            wav, ref_wav = dataset[idx]
            _, fpath = dataset.pairs[idx]
            plr = float(random.choice(args.plr_values))
            rmax = float(random.choice(args.rmax_values))
            deg_np = decode_fn(wav, ref_wav, plr, rmax)
            ref_np = wav.squeeze(0).numpy().astype(np.float64)
            ref_text = load_transcript(fpath)
            out.append({
                "file": os.path.basename(fpath),
                "plr": plr,
                "rmax": rmax,
                "system": label,
                "visqol": _safe_metric(calc_visqol(ref_np, deg_np, sr, mode=VISQOL_COMP_MODE)) if not args.no_visqol else float("nan"),
                "plcmos": _safe_metric(calc_plcmos(deg_np, ref_np, sr)),
                "utmos": _safe_metric(calc_utmos(deg_np, sr)),
                "wer": _safe_metric(calc_wer(deg_np, ref_text, sr, model_size=args.whisper_model)) if not args.no_wer and ref_text else float("nan"),
            })
        return out

    def ours_fixed_decode(wav, ref_wav, plr, rmax):
        # use the controller to preserve the proposed system behavior
        return decode_with_controller(st_model, flow_model, spk_encoder, controller, candidate_layers,
                                      wav, ref_wav, plr, rmax, device, args.n_steps)[0]

    controller, candidate_layers = build_controller(args, device)
    rows["proposed"] = collect(ours_fixed_decode, "proposed")

    for bw, key in [(1.5, "encodec_1.5k"), (3.0, "encodec_3.0k")]:
        if not HAS_ENCODEC:
            continue
        def _dec_encodec(wav, _ref_wav, plr, _rmax, bw=bw):
            return encodec_with_lfrplc(wav.squeeze(0), sr, bw, plr, device)
        rows[key] = collect(_dec_encodec, key)

    def _dec_opus(wav, _ref_wav, plr, _rmax):
        return opus_lbrr_with_plr(wav.squeeze(0), sr, 8.0, plr)

    rows["opus_8k"] = collect(_dec_opus, "opus_8k")

    for key, entries in rows.items():
        save_csv(os.path.join(out_dir, f"{key}.csv"), entries)

    return rows


def save_heatmap(rows: List[Dict], candidate_layers, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    groups = {}
    for r in rows:
        g = (r["plr"], r["rmax"])
        groups.setdefault(g, []).append(r["n_layers"])
    keys = sorted(groups.keys())
    mat = np.zeros((len(keys), len(candidate_layers)), dtype=np.float32)
    for i, k in enumerate(keys):
        vals = groups[k]
        for j, n in enumerate(candidate_layers):
            mat[i, j] = float(sum(1 for x in vals if x == n) / max(1, len(vals)))
    fig, ax = plt.subplots(figsize=(1.4 * len(candidate_layers) + 2.2, max(4.0, 0.45 * len(keys) + 2.2)))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=max(1e-6, float(mat.max())))
    ax.set_xticks(range(len(candidate_layers)))
    ax.set_xticklabels([f"N{n}" for n in candidate_layers])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"PLR={p:.2f}, Rmax={r:.1f}" for p, r in keys])
    ax.set_xlabel("Selected layer N")
    ax.set_ylabel("Operating point")
    ax.set_title("Selection distribution")
    fig.colorbar(im, ax=ax, pad=0.02, label="Probability")
    plt.tight_layout()
    path = os.path.join(out_dir, "selection_heatmap.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="System-level evaluation for proposed controller")
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path", default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--flow_ckpt", default="output/flow_checkpoints_stage3/best_train.pt")
    p.add_argument("--controller_ckpt", default="output/controller_checkpoints/best_controller.pt")
    p.add_argument("--data_dir", default="LibriSpeech")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--out_dir", default="output/system_test")
    p.add_argument("--n_layers_list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--plr_values", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    p.add_argument("--rmax_values", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--segment_sec", type=float, default=4.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--obs_feat_dim", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--visqol_workers", type=int, default=4)
    p.add_argument("--whisper_model", default="base")
    p.add_argument("--no_visqol", action="store_true")
    p.add_argument("--no_wer", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
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

    proposed_rows = eval_proposed_system(args, st_model, flow_model, spk_encoder, controller, candidate_layers, dataset, device, args.out_dir)
    save_csv(os.path.join(args.out_dir, "proposed_system.csv"), proposed_rows)
    save_heatmap(proposed_rows, candidate_layers, args.out_dir)

    baseline_rows = eval_baselines(args, st_model, flow_model, spk_encoder, dataset, device, args.out_dir)
    for key, rows in baseline_rows.items():
        save_csv(os.path.join(args.out_dir, f"{key}.csv"), rows)

    summary = {
        "proposed": summarize_rows(proposed_rows),
        "baselines": {k: {
            "num_samples": len(v),
            "avg_visqol": float(np.nanmean([r["visqol"] for r in v])) if v else float("nan"),
            "avg_plcmos": float(np.nanmean([r["plcmos"] for r in v])) if v else float("nan"),
            "avg_utmos": float(np.nanmean([r["utmos"] for r in v])) if v else float("nan"),
            "avg_wer": float(np.nanmean([r["wer"] for r in v])) if v else float("nan"),
        } for k, v in baseline_rows.items()}
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("saved summary to %s", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
