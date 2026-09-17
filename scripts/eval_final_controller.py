# -*- coding: utf-8 -*-
"""
Comprehensive system evaluation with controller-based average bitrate.

Part 1 (rate):
  Quality vs average bitrate at fixed PLR.
Part 2 (plr):
  Quality vs PLR under several target Rmax budgets.
Part 3 (ablation):
  Proposed / w-o controller / w-o flow.
Part 4 (heatmap):
  Controller selection distribution over (PLR, Rmax).

Each part saves:
  - figure(s)
  - json
  - csv
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speechtokenizer import SpeechTokenizer
from speechtokenizer.flow import BitrateController, FlowMatchingModel, PretrainedSpeakerEncoder
from scripts.eval_utils import (
    set_seed,
    load_audio,
    load_filelist,
    build_spk2files,
    pick_ref_wav,
    calc_visqol_batch,
    calc_utmos,
    calc_plcmos,
    calc_wer,
    load_transcript,
    encodec_with_plr,
    encodec_with_lfrplc,
    ffmpeg_codec_with_plr,
    opus_lbrr_with_plr,
    flow_sample,
    HAS_ENCODEC,
    nanmean,
    _latent_linear_interp,
)


VISQOL_OURS_SR = 48000
VISQOL_OURS_MODE = "audio"
VISQOL_COMP_SR = 16000
VISQOL_COMP_MODE = "speech"

RATE_METRICS = ["visqol"]
PLR_METRICS = ["visqol", "plcmos"]
ABLATION_METRICS = ["visqol", "plcmos"]
ABLATION_PLOT_METRICS = ["visqol", "plcmos"]

METRIC_LABELS = {
    "visqol": "VISQoL",
    "utmos": "UTMOS",
    "plcmos": "PLCMOS",
    "wer": "WER",
}

RATE_COMPETITORS = {
    "AAC": {
        "type": "ffmpeg",
        "codec": "aac",
        "bitrates": [4, 5, 6, 7, 8, 10, 12, 16, 20],
        "color": "#9467bd",
        "marker": "s",
    },
    "Opus": {
        "type": "ffmpeg",
        "codec": "libopus",
        "bitrates": [5.0, 6.0, 7.0, 8.0],
        "color": "#2ca02c",
        "marker": "^",
    },
    "EnCodec": {
        "type": "encodec",
        "bitrates": [1.5, 3.0, 6.0],
        "color": "#17becf",
        "marker": "D",
    },
    "ESC": {
        "type": "esc",
        "bitrates": [1.5, 3.0, 4.5],
        "streams": [1, 2, 3],
        "color": "#e377c2",
        "marker": "P",
    },
}

RATE_LEGEND_LABELS = {
    "AAC": "AAC-LC",
    "Opus": "Opus",
    "EnCodec": "EnCodec",
    "ESC": "ESC",
    "ours": "Proposed",
}

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_esc_model_cache = {}

PLR_LIST = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def _slug(x: float) -> str:
    return str(x).replace(".", "p")


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _save_csv(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _make_grid_fig(n_metrics, ncols=3, cell_w=4.5, cell_h=4.5):
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, cell_h * nrows))
    if nrows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    for mi in range(n_metrics, len(axes_flat)):
        axes_flat[mi].set_visible(False)
    return fig, axes_flat


def _plot_line(ax, xs, ys, color, marker, label, lw=1.1, ls="-", zorder=3,
               markersize=3.5, markeredgewidth=0.9):
    pairs = [(x, y) for x, y in zip(xs, ys) if not np.isnan(float(y))]
    if not pairs:
        return
    px, py = zip(*pairs)
    ax.plot(
        px, py, color=color, marker=marker, linewidth=lw, linestyle=ls,
        markersize=markersize, markerfacecolor="none",
        markeredgewidth=markeredgewidth, label=label, zorder=zorder
    )


def _resample_np_for_visqol(wav_np: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav_np.astype(np.float64)
    import torchaudio.functional as TAF
    wav_t = torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)
    wav_rs = TAF.resample(wav_t, src_sr, dst_sr).squeeze(0).cpu().numpy()
    return wav_rs.astype(np.float64)


def _calc_visqol_batch_mixed(ref_list, deg_list, sr: int, n_workers: int, target_sr: int, mode: str) -> list:
    if not ref_list:
        return []
    ref_rs = [_resample_np_for_visqol(r, sr, target_sr) for r in ref_list]
    deg_rs = [_resample_np_for_visqol(d, sr, target_sr) for d in deg_list]
    return calc_visqol_batch(ref_rs, deg_rs, target_sr, n_workers=n_workers, mode=mode)


def _try_load_esc():
    global _esc_model_cache
    if "model" in _esc_model_cache:
        return _esc_model_cache["model"]
    try:
        import yaml
        import sys as _sys
        esc_src = os.path.join(_PROJ_ROOT, "efficient-speech-codec-main")
        if esc_src not in _sys.path:
            _sys.path.insert(0, esc_src)
        from esc.models.codecs import make_model

        esc_ckpt_dir = os.path.join(_PROJ_ROOT, "external_checkpoints", "esc")
        cfg_path = os.path.join(esc_ckpt_dir, "config.yaml")
        pth_path = os.path.join(esc_ckpt_dir, "model.pth")

        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model = make_model(cfg["model"], cfg["model_name"])
        ckpt = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _esc_model_cache["model"] = model
        print(f"[ESC] loaded {pth_path}", flush=True)
        return model
    except Exception as e:
        print(f"[ESC] load failed: {e}", flush=True)
        _esc_model_cache["model"] = None
        return None


def _esc_encode_decode(wav_np: np.ndarray, sr: int, n_streams: int, p_loss: float = 0.0):
    model = _try_load_esc()
    if model is None:
        return None
    try:
        import torchaudio.functional as TAF
        esc_sr = 16000
        wav_t = torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)
        if sr != esc_sr:
            wav_t = TAF.resample(wav_t, sr, esc_sr)
        orig_len = int(wav_t.shape[-1])
        align = int(round(0.01 * esc_sr))
        pad = (-orig_len) % align
        if pad:
            wav_t = F.pad(wav_t, (0, pad), mode="replicate" if orig_len > 1 else "constant")

        with torch.no_grad():
            codes, feat_shape = model.encode(wav_t, num_streams=n_streams)
            if p_loss > 0:
                codes = codes.clone()
                t_code = codes.size(-1)
                if t_code > 0:
                    last = codes[..., 0].clone()
                    for t in range(t_code):
                        if random.random() < p_loss:
                            codes[..., t] = last
                        else:
                            last = codes[..., t].clone()
            wav_out = model.decode(codes, feat_shape)

        wav_dec = wav_out.squeeze().cpu().numpy()
        wav_dec = wav_dec[:orig_len]
        if esc_sr != sr:
            wav_dec_t = torch.from_numpy(wav_dec.astype(np.float32)).unsqueeze(0)
            wav_dec = TAF.resample(wav_dec_t, esc_sr, sr).squeeze().numpy()
        return wav_dec.astype(np.float64)
    except Exception as e:
        print(f"[ESC] decode failed: {e}", flush=True)
        return None


def _strip_compile_prefix(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def load_models(args, device):
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
    flow_model.load_state_dict(_strip_compile_prefix(ckpt["flow_model"]))
    flow_model.eval()

    spk_encoder = PretrainedSpeakerEncoder(
        emb_dim=getattr(saved_args, "spk_dim", 256),
        save_dir=os.path.join(os.path.dirname(args.flow_ckpt), "spkrec-ecapa"),
    ).to(device)
    spk_encoder.load_state_dict(_strip_compile_prefix(ckpt["spk_encoder"]), strict=False)
    spk_encoder.eval()

    controller_ckpt = torch.load(args.controller_ckpt, map_location="cpu")
    controller_args = argparse.Namespace(**controller_ckpt.get("args", {}))
    candidate_layers = list(getattr(controller_args, "n_layers_list", args.n_layers_list))
    controller = BitrateController(
        num_actions=len(candidate_layers),
        hidden_dim=getattr(controller_args, "hidden_dim", args.hidden_dim),
        in_dim=2 + getattr(controller_args, "obs_feat_dim", args.obs_feat_dim),
    ).to(device)
    controller.load_state_dict(controller_ckpt["controller"], strict=True)
    controller.eval()

    return st_model, flow_model, spk_encoder, controller, candidate_layers


def extract_obs_features(latent: torch.Tensor) -> torch.Tensor:
    x = latent.squeeze(0).float()
    frame_energy = x.pow(2).mean(dim=0).sqrt()
    frame_var = x.var(dim=0, unbiased=False)
    if x.shape[-1] > 1:
        delta = (x[:, 1:] - x[:, :-1]).abs().mean()
    else:
        delta = x.new_tensor(0.0)
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


@torch.no_grad()
def decode_ours_controller(
    fpath, wav, plr, rmax, st_model, flow_model, spk_encoder, controller,
    candidate_layers, spk2f, device, sr, n_steps
):
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

    ref_wav = pick_ref_wav(fpath, spk2f, sr).to(device)
    spk_emb = spk_encoder(ref_wav)
    lat_flow = flow_sample(flow_model, lat_in, spk_emb, n_steps=n_steps, n_layers=n_layers)
    lat_flow = lat_flow[..., :lat_in.shape[-1]]
    wav_flow = st_model.decoder(lat_flow).squeeze(0).cpu()
    return wav_flow.squeeze().numpy().astype(np.float64), n_layers, float(rate_vec[pred_idx].item())


@torch.no_grad()
def decode_ours_fixed_n(
    fpath, wav, plr, n_layers, st_model, flow_model, spk_encoder, spk2f, device, sr, n_steps
):
    latent_by_n = build_candidate_latents(st_model, wav, [2, 3, 4, 5, 6, 7, 8], p_loss=plr, device=device)
    lat_in = latent_by_n[n_layers]
    ref_wav = pick_ref_wav(fpath, spk2f, sr).to(device)
    spk_emb = spk_encoder(ref_wav)
    lat_flow = flow_sample(flow_model, lat_in, spk_emb, n_steps=n_steps, n_layers=n_layers)
    lat_flow = lat_flow[..., :lat_in.shape[-1]]
    wav_flow = st_model.decoder(lat_flow).squeeze(0).cpu()
    return wav_flow.squeeze().numpy().astype(np.float64)


@torch.no_grad()
def decode_ours_fixed_n_noflow(wav, plr, n_layers, st_model, device):
    latent_by_n = build_candidate_latents(st_model, wav, [2, 3, 4, 5, 6, 7, 8], p_loss=plr, device=device)
    wav_out = st_model.decoder(latent_by_n[n_layers]).squeeze(0).cpu()
    return wav_out.squeeze().numpy().astype(np.float64)


def _nearest_feasible_n(rmax: float, candidate_layers: List[int]) -> int:
    feasible = [n for n in candidate_layers if n * 0.5 <= rmax + 1e-6]
    if feasible:
        return max(feasible)
    return min(candidate_layers)


def _evaluate_metric_lists(ref_list, deg_list, sr, visqol_workers, ours_mode: bool):
    if ours_mode:
        visqol_list = _calc_visqol_batch_mixed(
            ref_list, deg_list, sr, visqol_workers, VISQOL_OURS_SR, VISQOL_OURS_MODE
        )
    else:
        visqol_list = _calc_visqol_batch_mixed(
            ref_list, deg_list, sr, visqol_workers, VISQOL_COMP_SR, VISQOL_COMP_MODE
        )
    return visqol_list


def eval_rate_controller(args, st_model, flow_model, spk_encoder, controller, candidate_layers,
                         selected, spk2f, device, out_dir):
    sr = st_model.sample_rate
    rows = []
    details = []
    rate_plr = args.rate_plr

    print("\n[Part1] Proposed rate sweep with controller", flush=True)
    for rmax in args.rmax_values:
        ref_list, deg_list = [], []
        utmos_list, wer_list = [], []
        realized_rates = []
        layer_hist = {f"N{n}": 0 for n in candidate_layers}
        print(f"  [Ours] target Rmax={rmax:.1f}kbps", flush=True)

        for i, fpath in enumerate(selected, 1):
            print(f"    [{i}/{len(selected)}] {os.path.basename(fpath)}", flush=True)
            wav = load_audio(fpath, sr)[:, :int(args.max_sec * sr)]
            ref_np = wav.squeeze().numpy().astype(np.float64)
            deg_np, n_layers, realized_rate = decode_ours_controller(
                fpath, wav, rate_plr, rmax, st_model, flow_model, spk_encoder,
                controller, candidate_layers, spk2f, device, sr, args.n_steps
            )
            ref_text = load_transcript(fpath)

            ref_list.append(ref_np)
            deg_list.append(deg_np)
            utmos_list.append(calc_utmos(deg_np, sr))
            wer_list.append(calc_wer(deg_np, ref_text, sr, model_size=args.whisper_model) if (not args.no_wer and ref_text) else float("nan"))
            realized_rates.append(realized_rate)
            layer_hist[f"N{n_layers}"] += 1
            details.append({
                "system": "proposed",
                "target_rmax_kbps": float(rmax),
                "plr": float(rate_plr),
                "file": os.path.basename(fpath),
                "selected_n": int(n_layers),
                "rate_kbps": float(realized_rate),
            })

        vis_list = [float("nan")] * len(ref_list) if args.no_visqol else _evaluate_metric_lists(
            ref_list, deg_list, sr, args.visqol_workers, ours_mode=True
        )
        row = {
            "target_rmax_kbps": float(rmax),
            "bitrate_kbps": nanmean(realized_rates),
            "visqol": nanmean(vis_list),
            "utmos": nanmean(utmos_list),
            "wer": nanmean(wer_list),
        }
        for k, v in layer_hist.items():
            row[k] = int(v)
        rows.append(row)
        print(
            f"    avg_rate={row['bitrate_kbps']:.3f} VISQOL={row['visqol']:.3f} "
            f"UTMOS={row['utmos']:.3f} WER={row['wer']:.3f}",
            flush=True
        )

    _save_csv(rows, os.path.join(out_dir, "rate_data_ours.csv"))
    _save_csv(details, os.path.join(out_dir, "rate_data_ours_details.csv"))
    return rows


def _evaluate_rate_entry(model_name, bitrate_kbps, selected, sr, max_sec, no_visqol, visqol_workers,
                         decode_one, ours_mode=False, whisper_model="base", no_wer=False):
    ref_list, deg_list = [], []
    utmos_list, wer_list = [], []
    for i, fpath in enumerate(selected, 1):
        print(f"    [{i}/{len(selected)}] {model_name} {os.path.basename(fpath)}", flush=True)
        wav = load_audio(fpath, sr)[:, :int(max_sec * sr)]
        ref_np = wav.squeeze().numpy().astype(np.float64)
        deg_np = decode_one(fpath, wav, ref_np)
        if deg_np is None:
            continue
        ref_list.append(ref_np)
        deg_list.append(deg_np)
        utmos_list.append(calc_utmos(deg_np, sr))
        ref_text = load_transcript(fpath)
        wer_list.append(calc_wer(deg_np, ref_text, sr, model_size=whisper_model) if (not no_wer and ref_text) else float("nan"))

    if no_visqol:
        vis_list = [float("nan")] * len(ref_list)
    else:
        vis_list = _evaluate_metric_lists(ref_list, deg_list, sr, visqol_workers, ours_mode=ours_mode)

    return {
        "bitrate_kbps": float(bitrate_kbps),
        "visqol": nanmean(vis_list),
        "utmos": nanmean(utmos_list),
        "wer": nanmean(wer_list),
    }


def eval_rate_competitors(args, selected, sr, device, out_dir):
    rows_by_name = {}
    flat_rows = []
    for name, cfg in RATE_COMPETITORS.items():
        if cfg["type"] == "encodec" and not HAS_ENCODEC:
            print(f"[Part1] skip {name}: encodec not installed", flush=True)
            continue
        if cfg["type"] == "esc" and _try_load_esc() is None:
            print(f"[Part1] skip {name}: ESC not available", flush=True)
            continue
        rows = []
        print(f"\n[Part1] {name} rate sweep", flush=True)
        for idx, bw in enumerate(cfg["bitrates"]):
            if cfg["type"] == "encodec":
                def _decode_comp(_fpath, wav, _ref_np, bw=bw):
                    return encodec_with_plr(wav.squeeze(0), sr, bw, p_loss=args.rate_plr, device=device)
            elif cfg["type"] == "esc":
                n_st = int(cfg["streams"][idx])
                def _decode_comp(_fpath, _wav, ref_np, n_st=n_st):
                    return _esc_encode_decode(ref_np, sr, n_st, p_loss=args.rate_plr)
            else:
                codec = cfg["codec"]
                def _decode_comp(_fpath, wav, _ref_np, bw=bw, codec=codec):
                    return ffmpeg_codec_with_plr(wav.squeeze(0), sr, codec, bw, p_loss=args.rate_plr)

            row = _evaluate_rate_entry(
                name, bw, selected, sr, args.max_sec, args.no_visqol, args.visqol_workers,
                _decode_comp, ours_mode=False, whisper_model=args.whisper_model, no_wer=args.no_wer
            )
            rows.append(row)
            flat_rows.append({"system": name, **row})
            print(
                f"  {name} {bw:.1f}kbps VISQOL={row['visqol']:.3f} "
                f"UTMOS={row['utmos']:.3f} WER={row['wer']:.3f}",
                flush=True
            )
        rows_by_name[name] = rows

    _save_csv(flat_rows, os.path.join(out_dir, "rate_data_competitors.csv"))
    return rows_by_name


def plot_rate(ours_rows, comp_results, out_dir, rate_plr=0.05):
    def _plot_one(log_x: bool, out_name: str):
        fig, axes_flat = _make_grid_fig(len(RATE_METRICS))
        for mi, metric in enumerate(RATE_METRICS):
            ax = axes_flat[mi]
            for name, cfg in RATE_COMPETITORS.items():
                if name not in comp_results:
                    continue
                rows = comp_results[name]
                _plot_line(
                    ax,
                    [r["bitrate_kbps"] for r in rows],
                    [r.get(metric, float("nan")) for r in rows],
                    cfg["color"], cfg["marker"], RATE_LEGEND_LABELS.get(name, name)
                )
            _plot_line(
                ax,
                [r["bitrate_kbps"] for r in ours_rows],
                [r.get(metric, float("nan")) for r in ours_rows],
                "#d62728", "o", RATE_LEGEND_LABELS["ours"], lw=1.6, zorder=6
            )
            ax.set_xlabel("Average Bitrate (kbps)", fontsize=10, labelpad=16)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
            if log_x:
                ax.set_xscale("log")
                ax.set_xlim(left=0.8)
            else:
                ax.set_xlim(left=0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
            ax.set_title(f"({chr(97 + mi)})", fontsize=10, pad=20)

        suffix = "Log Rate" if log_x else "Linear Rate"
        fig.suptitle(f"Quality vs Average Bitrate ({suffix}, PLR={rate_plr*100:.0f}%)", fontsize=12)
        plt.tight_layout()
        path = os.path.join(out_dir, out_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure saved: {path}", flush=True)

    _plot_one(False, "fig_rate_final.png")
    _plot_one(True, "fig_rate_final_log.png")


def eval_plr(args, st_model, flow_model, spk_encoder, controller, candidate_layers,
             selected, spk2f, device, out_dir):
    sr = st_model.sample_rate
    systems = {}
    detail_rows = []

    competitor_specs = [
        ("encodec_1.5k", "EnCodec (1.5kbps)", lambda wav, plr: encodec_with_lfrplc(wav.squeeze(0), sr, 1.5, plr, device), False),
        ("encodec_3.0k", "EnCodec (3.0kbps)", lambda wav, plr: encodec_with_lfrplc(wav.squeeze(0), sr, 3.0, plr, device), False),
        ("esc_1.5k", "ESC (1.5kbps)", lambda wav, plr: _esc_encode_decode(wav.squeeze().numpy().astype(np.float64), sr, 1, plr), False),
        ("esc_3.0k", "ESC (3.0kbps)", lambda wav, plr: _esc_encode_decode(wav.squeeze().numpy().astype(np.float64), sr, 2, plr), False),
        ("esc_4.5k", "ESC (4.5kbps)", lambda wav, plr: _esc_encode_decode(wav.squeeze().numpy().astype(np.float64), sr, 3, plr), False),
        ("opus_8k", "Opus (~8kbps)", lambda wav, plr: opus_lbrr_with_plr(wav.squeeze(0), sr, 8.0, plr), False),
    ]
    competitor_specs = [x for x in competitor_specs if x[0] != "encodec_1.5k" or HAS_ENCODEC]
    competitor_specs = [x for x in competitor_specs if x[0] != "encodec_3.0k" or HAS_ENCODEC]
    competitor_specs = [x for x in competitor_specs if not x[0].startswith("esc_") or _try_load_esc() is not None]

    ours_specs = [(rmax, f"ours_rmax_{_slug(rmax)}", f"Proposed (Rmax={rmax:.1f}kbps)") for rmax in args.plr_rmax_values]
    for _, key, _ in ours_specs:
        systems[key] = {m: [] for m in PLR_METRICS}
        systems[key]["avg_rate_kbps"] = []
    for key, _, _, _ in competitor_specs:
        systems[key] = {m: [] for m in PLR_METRICS}
        systems[key]["avg_rate_kbps"] = []

    for plr in PLR_LIST:
        print(f"\n[Part2] PLR={plr*100:.0f}%", flush=True)

        for rmax, key, label in ours_specs:
            ref_list, deg_list = [], []
            plc_list, utmos_list, wer_list, rate_list = [], [], [], []
            for fpath in selected:
                wav = load_audio(fpath, sr)[:, :int(args.max_sec * sr)]
                ref_np = wav.squeeze().numpy().astype(np.float64)
                deg_np, n_layers, rate = decode_ours_controller(
                    fpath, wav, plr, rmax, st_model, flow_model, spk_encoder,
                    controller, candidate_layers, spk2f, device, sr, args.n_steps
                )
                ref_text = load_transcript(fpath)
                ref_list.append(ref_np)
                deg_list.append(deg_np)
                plc_list.append(calc_plcmos(deg_np, ref_np, sr))
                utmos_list.append(calc_utmos(deg_np, sr))
                wer_list.append(calc_wer(deg_np, ref_text, sr, model_size=args.whisper_model) if (not args.no_wer and ref_text) else float("nan"))
                rate_list.append(rate)
                detail_rows.append({
                    "part": "plr",
                    "system": label,
                    "plr": float(plr),
                    "target_rmax_kbps": float(rmax),
                    "rate_kbps": float(rate),
                    "selected_n": int(n_layers),
                    "file": os.path.basename(fpath),
                })
            vis_list = [float("nan")] * len(ref_list) if args.no_visqol else _evaluate_metric_lists(
                ref_list, deg_list, sr, args.visqol_workers, ours_mode=True
            )
            systems[key]["visqol"].append(nanmean(vis_list))
            systems[key]["plcmos"].append(nanmean(plc_list))
            systems[key]["utmos"].append(nanmean(utmos_list))
            systems[key]["wer"].append(nanmean(wer_list))
            systems[key]["avg_rate_kbps"].append(nanmean(rate_list))
            print(
                f"  {label:30s} avg_rate={systems[key]['avg_rate_kbps'][-1]:.3f} "
                f"VISQOL={systems[key]['visqol'][-1]:.3f} PLCMOS={systems[key]['plcmos'][-1]:.3f}",
                flush=True
            )

        for key, label, decode_fn, is_ours in competitor_specs:
            ref_list, deg_list = [], []
            plc_list, utmos_list, wer_list = [], [], []
            if "1.5" in key:
                fixed_rate = 1.5
            elif "3.0" in key:
                fixed_rate = 3.0
            elif "esc_4.5" in key:
                fixed_rate = 4.5
            else:
                fixed_rate = 8.0
            for fpath in selected:
                wav = load_audio(fpath, sr)[:, :int(args.max_sec * sr)]
                ref_np = wav.squeeze().numpy().astype(np.float64)
                deg_np = decode_fn(wav, plr)
                if deg_np is None:
                    continue
                ref_text = load_transcript(fpath)
                ref_list.append(ref_np)
                deg_list.append(deg_np)
                plc_list.append(calc_plcmos(deg_np, ref_np, sr))
                utmos_list.append(calc_utmos(deg_np, sr))
                wer_list.append(calc_wer(deg_np, ref_text, sr, model_size=args.whisper_model) if (not args.no_wer and ref_text) else float("nan"))
                detail_rows.append({
                    "part": "plr",
                    "system": label,
                    "plr": float(plr),
                    "target_rmax_kbps": float("nan"),
                    "rate_kbps": float(fixed_rate),
                    "selected_n": float("nan"),
                    "file": os.path.basename(fpath),
                })
            vis_list = [float("nan")] * len(ref_list) if args.no_visqol else _evaluate_metric_lists(
                ref_list, deg_list, sr, args.visqol_workers, ours_mode=is_ours
            )
            systems[key]["visqol"].append(nanmean(vis_list))
            systems[key]["plcmos"].append(nanmean(plc_list))
            systems[key]["utmos"].append(nanmean(utmos_list))
            systems[key]["wer"].append(nanmean(wer_list))
            systems[key]["avg_rate_kbps"].append(float(fixed_rate))
            print(
                f"  {label:30s} avg_rate={fixed_rate:.3f} "
                f"VISQOL={systems[key]['visqol'][-1]:.3f} PLCMOS={systems[key]['plcmos'][-1]:.3f}",
                flush=True
            )

    flat_rows = []
    for _, key, label in ours_specs:
        for i, plr in enumerate(PLR_LIST):
            flat_rows.append({
                "system": label,
                "plr": float(plr),
                "avg_rate_kbps": _safe_float(systems[key]["avg_rate_kbps"][i]),
                **{m: _safe_float(systems[key][m][i]) for m in PLR_METRICS},
            })
    for key, label, _, _ in competitor_specs:
        for i, plr in enumerate(PLR_LIST):
            flat_rows.append({
                "system": label,
                "plr": float(plr),
                "avg_rate_kbps": _safe_float(systems[key]["avg_rate_kbps"][i]),
                **{m: _safe_float(systems[key][m][i]) for m in PLR_METRICS},
            })

    _save_csv(flat_rows, os.path.join(out_dir, "plr_data.csv"))
    _save_csv(detail_rows, os.path.join(out_dir, "plr_data_details.csv"))
    _save_json({"plr_list": PLR_LIST, "systems": systems}, os.path.join(out_dir, "plr_data.json"))
    return systems, ours_specs, competitor_specs


def plot_plr(systems, ours_specs, competitor_specs, out_dir):
    fig, axes_flat = _make_grid_fig(len(PLR_METRICS), ncols=2)
    plr_pct = [p * 100 for p in PLR_LIST]

    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    ours_styles = {}
    for i, (_, key, label) in enumerate(ours_specs):
        ours_styles[key] = {"color": colors[i % len(colors)], "marker": "o", "ls": "-" if i == 0 else "--", "label": label}
    comp_styles = {
        "encodec_1.5k": {"color": "#17becf", "marker": "D", "ls": "--"},
        "encodec_3.0k": {"color": "#17becf", "marker": "D", "ls": "-"},
        "opus_8k": {"color": "#2ca02c", "marker": "^", "ls": "-"},
    }

    for mi, metric in enumerate(PLR_METRICS):
        ax = axes_flat[mi]
        for _, key, label in ours_specs:
            st = ours_styles[key]
            _plot_line(ax, plr_pct, systems[key][metric], st["color"], st["marker"], st["label"], lw=1.5, ls=st["ls"])
        for key, label, _, _ in competitor_specs:
            st = comp_styles.get(key, {"color": "#7f7f7f", "marker": "s", "ls": "-"})
            _plot_line(ax, plr_pct, systems[key][metric], st["color"], st["marker"], label, lw=1.1, ls=st["ls"])
        ax.set_xlabel("Packet Loss Probability (%)", fontsize=10, labelpad=16)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_xlim(-1, 32)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"({chr(97 + mi)})", fontsize=10, pad=20)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Quality vs Packet Loss Rate", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    path = os.path.join(out_dir, "fig_plr_final.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}", flush=True)


def eval_ablation(args, st_model, flow_model, spk_encoder, controller, candidate_layers,
                  selected, spk2f, device, out_dir):
    sr = st_model.sample_rate
    result = {}
    detail_rows = []

    for rmax in args.ablation_rmax_values:
        key = f"rmax_{_slug(rmax)}"
        result[key] = {
            "label_cf": f"w/ Controller + w/ Flow (Rmax={rmax:.1f})",
            "label_cn": f"w/ Controller + w/o Flow (Rmax={rmax:.1f})",
            "label_nf": f"w/o Controller + w/ Flow (Rmax={rmax:.1f})",
            "label_nn": f"w/o Controller + w/o Flow (Rmax={rmax:.1f})",
            "cf": {m: [] for m in ABLATION_METRICS},
            "cn": {m: [] for m in ABLATION_METRICS},
            "nf": {m: [] for m in ABLATION_METRICS},
            "nn": {m: [] for m in ABLATION_METRICS},
            "avg_rate_cf_kbps": [],
            "avg_rate_cn_kbps": [],
        }

        for plr in PLR_LIST:
            cf_ref, cf_deg = [], []
            cn_ref, cn_deg = [], []
            nf_ref, nf_deg = [], []
            nn_ref, nn_deg = [], []
            cf_plc, cf_ut, cf_wer = [], [], []
            cn_plc, cn_ut, cn_wer = [], [], []
            nf_plc, nf_ut, nf_wer = [], [], []
            nn_plc, nn_ut, nn_wer = [], [], []
            cf_rate, cn_rate = [], []
            fixed_n = _nearest_feasible_n(rmax, candidate_layers)

            for fpath in selected:
                wav = load_audio(fpath, sr)[:, :int(args.max_sec * sr)]
                ref_np = wav.squeeze().numpy().astype(np.float64)
                ref_text = load_transcript(fpath)

                deg_cf, n_sel, rate_sel = decode_ours_controller(
                    fpath, wav, plr, rmax, st_model, flow_model, spk_encoder,
                    controller, candidate_layers, spk2f, device, sr, args.n_steps
                )
                deg_cn = decode_ours_fixed_n_noflow(wav, plr, n_sel, st_model, device)
                deg_nf = decode_ours_fixed_n(
                    fpath, wav, plr, fixed_n, st_model, flow_model, spk_encoder, spk2f, device, sr, args.n_steps
                )
                deg_nn = decode_ours_fixed_n_noflow(wav, plr, fixed_n, st_model, device)

                cf_ref.append(ref_np)
                cf_deg.append(deg_cf)
                cn_ref.append(ref_np)
                cn_deg.append(deg_cn)
                nf_ref.append(ref_np)
                nf_deg.append(deg_nf)
                nn_ref.append(ref_np)
                nn_deg.append(deg_nn)

                cf_plc.append(calc_plcmos(deg_cf, ref_np, sr))
                cf_ut.append(calc_utmos(deg_cf, sr))
                cf_wer.append(calc_wer(deg_cf, ref_text, sr, model_size=args.whisper_model) if (not args.no_wer and ref_text) else float("nan"))
                cn_plc.append(calc_plcmos(deg_cn, ref_np, sr))
                cn_ut.append(calc_utmos(deg_cn, sr))
                cn_wer.append(calc_wer(deg_cn, ref_text, sr, model_size=args.whisper_model) if (not args.no_wer and ref_text) else float("nan"))
                nf_plc.append(calc_plcmos(deg_nf, ref_np, sr))
                nf_ut.append(calc_utmos(deg_nf, sr))
                nf_wer.append(calc_wer(deg_nf, ref_text, sr, model_size=args.whisper_model) if (not args.no_wer and ref_text) else float("nan"))
                nn_plc.append(calc_plcmos(deg_nn, ref_np, sr))
                nn_ut.append(calc_utmos(deg_nn, sr))
                nn_wer.append(calc_wer(deg_nn, ref_text, sr, model_size=args.whisper_model) if (not args.no_wer and ref_text) else float("nan"))
                cf_rate.append(rate_sel)
                cn_rate.append(rate_sel)

                detail_rows.append({
                    "target_rmax_kbps": float(rmax),
                    "plr": float(plr),
                    "file": os.path.basename(fpath),
                    "selected_n": int(n_sel),
                    "fixed_n": int(fixed_n),
                    "selected_rate_kbps": float(rate_sel),
                })

            cf_vis = [float("nan")] * len(cf_ref) if args.no_visqol else _evaluate_metric_lists(
                cf_ref, cf_deg, sr, args.visqol_workers, ours_mode=True
            )
            cn_vis = [float("nan")] * len(cn_ref) if args.no_visqol else _evaluate_metric_lists(
                cn_ref, cn_deg, sr, args.visqol_workers, ours_mode=True
            )
            nf_vis = [float("nan")] * len(nf_ref) if args.no_visqol else _evaluate_metric_lists(
                nf_ref, nf_deg, sr, args.visqol_workers, ours_mode=True
            )
            nn_vis = [float("nan")] * len(nn_ref) if args.no_visqol else _evaluate_metric_lists(
                nn_ref, nn_deg, sr, args.visqol_workers, ours_mode=True
            )

            result[key]["cf"]["visqol"].append(nanmean(cf_vis))
            result[key]["cf"]["plcmos"].append(nanmean(cf_plc))
            result[key]["cf"]["utmos"].append(nanmean(cf_ut))
            result[key]["cf"]["wer"].append(nanmean(cf_wer))

            result[key]["cn"]["visqol"].append(nanmean(cn_vis))
            result[key]["cn"]["plcmos"].append(nanmean(cn_plc))
            result[key]["cn"]["utmos"].append(nanmean(cn_ut))
            result[key]["cn"]["wer"].append(nanmean(cn_wer))

            result[key]["nf"]["visqol"].append(nanmean(nf_vis))
            result[key]["nf"]["plcmos"].append(nanmean(nf_plc))
            result[key]["nf"]["utmos"].append(nanmean(nf_ut))
            result[key]["nf"]["wer"].append(nanmean(nf_wer))

            result[key]["nn"]["visqol"].append(nanmean(nn_vis))
            result[key]["nn"]["plcmos"].append(nanmean(nn_plc))
            result[key]["nn"]["utmos"].append(nanmean(nn_ut))
            result[key]["nn"]["wer"].append(nanmean(nn_wer))

            result[key]["avg_rate_cf_kbps"].append(nanmean(cf_rate))
            result[key]["avg_rate_cn_kbps"].append(nanmean(cn_rate))
            print(
                f"[Part3] Rmax={rmax:.1f} PLR={plr:.2f} "
                f"cf={result[key]['cf']['visqol'][-1]:.3f} "
                f"cn={result[key]['cn']['visqol'][-1]:.3f} "
                f"nf={result[key]['nf']['visqol'][-1]:.3f} "
                f"nn={result[key]['nn']['visqol'][-1]:.3f}",
                flush=True
            )

    flat_rows = []
    for rmax in args.ablation_rmax_values:
        key = f"rmax_{_slug(rmax)}"
        for i, plr in enumerate(PLR_LIST):
            for group in ["cf", "cn", "nf", "nn"]:
                flat_rows.append({
                    "target_rmax_kbps": float(rmax),
                    "plr": float(plr),
                    "group": group,
                    "avg_rate_kbps": _safe_float(
                        result[key]["avg_rate_cf_kbps"][i] if group == "cf"
                        else (result[key]["avg_rate_cn_kbps"][i] if group == "cn" else rmax)
                    ),
                    **{m: _safe_float(result[key][group][m][i]) for m in ABLATION_METRICS},
                })

    _save_csv(flat_rows, os.path.join(out_dir, "ablation_data.csv"))
    _save_csv(detail_rows, os.path.join(out_dir, "ablation_data_details.csv"))
    _save_json({"plr_list": PLR_LIST, "result": result}, os.path.join(out_dir, "ablation_data.json"))
    return result


def plot_ablation(result, out_dir):
    fig, axes_flat = _make_grid_fig(len(ABLATION_PLOT_METRICS), ncols=2)
    plr_pct = [p * 100 for p in PLR_LIST]
    colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]

    for mi, metric in enumerate(ABLATION_PLOT_METRICS):
        ax = axes_flat[mi]
        for i, key in enumerate(sorted(result.keys())):
            color = colors[i % len(colors)]
            label_cf = result[key]["label_cf"]
            label_cn = result[key]["label_cn"]
            label_nf = result[key]["label_nf"]
            label_nn = result[key]["label_nn"]
            _plot_line(ax, plr_pct, result[key]["cf"][metric], color, "o", label_cf, lw=1.5, ls="-")
            _plot_line(ax, plr_pct, result[key]["cn"][metric], color, "s", label_cn, lw=1.2, ls="--")
            _plot_line(ax, plr_pct, result[key]["nf"][metric], color, "^", label_nf, lw=1.2, ls="-.")
            _plot_line(ax, plr_pct, result[key]["nn"][metric], color, "D", label_nn, lw=1.1, ls=":")
        ax.set_xlabel("Packet Loss Probability (%)", fontsize=10, labelpad=16)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_xlim(-1, 32)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"({chr(97 + mi)})", fontsize=10, pad=20)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ablation Study", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.23)
    path = os.path.join(out_dir, "fig_ablation_final.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}", flush=True)


def eval_heatmap(args, st_model, flow_model, spk_encoder, controller, candidate_layers,
                 selected, spk2f, device, out_dir):
    counts = {(plr, rmax): {n: 0 for n in candidate_layers} for plr in args.plr_values for rmax in args.rmax_values}
    sr = st_model.sample_rate

    for plr in args.plr_values:
        for rmax in args.rmax_values:
            print(f"[Part4] heatmap PLR={plr:.2f} Rmax={rmax:.1f}", flush=True)
            for fpath in selected:
                wav = load_audio(fpath, sr)[:, :int(args.max_sec * sr)]
                _, n_layers, _ = decode_ours_controller(
                    fpath, wav, plr, rmax, st_model, flow_model, spk_encoder,
                    controller, candidate_layers, spk2f, device, sr, args.n_steps
                )
                counts[(plr, rmax)][n_layers] += 1

    rows = []
    mat = []
    labels = []
    for plr in args.plr_values:
        for rmax in args.rmax_values:
            total = max(1, sum(counts[(plr, rmax)].values()))
            row = {"plr": float(plr), "rmax": float(rmax)}
            probs = []
            for n in candidate_layers:
                prob = counts[(plr, rmax)][n] / total
                row[f"N{n}"] = float(prob)
                probs.append(prob)
            rows.append(row)
            mat.append(probs)
            labels.append(f"PLR={plr:.2f}, Rmax={rmax:.1f}")

    _save_csv(rows, os.path.join(out_dir, "selection_heatmap.csv"))
    _save_json(rows, os.path.join(out_dir, "selection_heatmap.json"))

    fig, ax = plt.subplots(figsize=(1.3 * len(candidate_layers) + 2.2, max(4.0, 0.42 * len(labels) + 2.2)))
    mat = np.array(mat, dtype=np.float32)
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
    path = os.path.join(out_dir, "selection_heatmap.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {path}", flush=True)


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device} | part={args.part}", flush=True)

    st_model, flow_model, spk_encoder, controller, candidate_layers = load_models(args, device)
    sr = st_model.sample_rate

    files = load_filelist(args.data_dir, args.split)
    spk2f = build_spk2files(files)
    selected = random.sample(files, min(args.num_samples, len(files)))
    print(f"Samples: {len(selected)} | split={args.split}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_rate = os.path.join(args.out_dir, "part1_rate")
    out_plr = os.path.join(args.out_dir, "part2_plr")
    out_ablation = os.path.join(args.out_dir, "part3_ablation")
    out_heatmap = os.path.join(args.out_dir, "part4_heatmap")
    for d in (out_rate, out_plr, out_ablation, out_heatmap):
        os.makedirs(d, exist_ok=True)

    if args.part in ("rate", "all"):
        ours_rows = eval_rate_controller(
            args, st_model, flow_model, spk_encoder, controller, candidate_layers,
            selected, spk2f, device, out_rate
        )
        comp_results = eval_rate_competitors(args, selected, sr, device, out_rate)
        plot_rate(ours_rows, comp_results, out_rate, rate_plr=args.rate_plr)
        _save_json(
            {"ours": ours_rows, "competitors": comp_results, "rate_plr": args.rate_plr},
            os.path.join(out_rate, "rate_data.json"),
        )

    if args.part in ("plr", "all"):
        systems, ours_specs, competitor_specs = eval_plr(
            args, st_model, flow_model, spk_encoder, controller, candidate_layers,
            selected, spk2f, device, out_plr
        )
        plot_plr(systems, ours_specs, competitor_specs, out_plr)

    if args.part in ("ablation", "all"):
        ablation_result = eval_ablation(
            args, st_model, flow_model, spk_encoder, controller, candidate_layers,
            selected, spk2f, device, out_ablation
        )
        plot_ablation(ablation_result, out_ablation)

    if args.part in ("heatmap", "all"):
        eval_heatmap(
            args, st_model, flow_model, spk_encoder, controller, candidate_layers,
            selected, spk2f, device, out_heatmap
        )

    print(f"All results saved to {args.out_dir}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Controller-based comprehensive evaluation")
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path", default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--flow_ckpt", default="output/flow_checkpoints_stage3/best_train.pt")
    p.add_argument("--controller_ckpt", default="output/controller_checkpoints/best_controller.pt")
    p.add_argument("--data_dir", default="LibriSpeech")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--max_sec", type=float, default=10.0)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--rate_plr", type=float, default=0.05)
    p.add_argument("--n_layers_list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--rmax_values", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--plr_rmax_values", type=float, nargs="+", default=[1.5, 2.0, 3.0, 4.0])
    p.add_argument("--ablation_rmax_values", type=float, nargs="+", default=[1.5, 3.0])
    p.add_argument("--plr_values", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    p.add_argument("--out_dir", default="output/eval_results_controller")
    p.add_argument("--no_visqol", action="store_true")
    p.add_argument("--no_wer", action="store_true")
    p.add_argument("--visqol_workers", type=int, default=8)
    p.add_argument("--whisper_model", default="base")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--obs_feat_dim", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--part", default="all", choices=["all", "rate", "plr", "ablation", "heatmap"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
