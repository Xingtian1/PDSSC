# -*- coding: utf-8 -*-
"""
Minimal flow-only evaluation script for smoke testing the evaluation pipeline.

This script does not use the controller. It only verifies that:
1. part1 (rate) can run with VISQoL
2. part2 (plr) can run with PLCMOS

Outputs:
  - csv
  - json

No part3 / part4.
"""

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speechtokenizer import SpeechTokenizer
from speechtokenizer.flow import FlowMatchingModel, PretrainedSpeakerEncoder
from scripts.eval_utils import (
    set_seed,
    nanmean,
    load_audio,
    load_filelist,
    build_spk2files,
    pick_ref_wav,
    calc_visqol_batch,
    calc_plcmos,
    _latent_linear_interp,
    flow_sample,
    encodec_with_plr,
    ffmpeg_codec_with_plr,
)


PLR_LIST = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_esc_model_cache = {}

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

PLOT_STYLE = {
    "ours": {"color": "#d62728", "marker": "o", "label": "Proposed"},
    "AAC": {"color": "#9467bd", "marker": "s", "label": "AAC-LC"},
    "Opus": {"color": "#2ca02c", "marker": "^", "label": "Opus"},
    "EnCodec": {"color": "#17becf", "marker": "D", "label": "EnCodec"},
    "ESC": {"color": "#e377c2", "marker": "P", "label": "ESC"},
}


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


def _slug(x: float) -> str:
    return str(x).replace(".", "p")


def _plot_line(ax, xs, ys, color, marker, label):
    pairs = [(x, y) for x, y in zip(xs, ys) if not np.isnan(float(y))]
    if not pairs:
        return
    px, py = zip(*pairs)
    ax.plot(
        px, py,
        color=color,
        marker=marker,
        linewidth=1.2,
        markersize=4,
        markerfacecolor="none",
        markeredgewidth=1.0,
        label=label,
    )


def _try_load_esc():
    global _esc_model_cache
    if "model" in _esc_model_cache:
        return _esc_model_cache["model"]
    try:
        import yaml
        esc_src = os.path.join(_PROJ_ROOT, "efficient-speech-codec-main")
        if esc_src not in sys.path:
            sys.path.insert(0, esc_src)
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


def _infer_esc_overlap(model) -> int:
    candidates = [
        getattr(model, "overlap", None),
        getattr(model, "frame_overlap", None),
        getattr(model, "hop_length", None),
    ]
    for v in candidates:
        if isinstance(v, int) and v > 1:
            return v

    for mod_name in ["encoder", "codec", "generator"]:
        mod = getattr(model, mod_name, None)
        if mod is None:
            continue
        for attr in ["overlap", "frame_overlap", "hop_length"]:
            v = getattr(mod, attr, None)
            if isinstance(v, int) and v > 1:
                return v
    return 1


@torch.no_grad()
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
        # Start from the 10 ms patch alignment used in eval_final.py, but
        # retry larger alignments because some ESC variants enforce a stricter
        # overlap multiple internally.
        align_candidates = [160, 320, 480, 640, 960, 1280, 1920, 2560, 3200]
        last_err = None
        codes, feat_shape = None, None
        for align in align_candidates:
            wav_try = wav_t
            pad = (-orig_len) % align
            if pad:
                wav_try = torch.nn.functional.pad(
                    wav_t,
                    (0, pad),
                    mode="replicate" if orig_len > 1 else "constant",
                )
            try:
                codes, feat_shape = model.encode(wav_try, num_streams=n_streams)
                wav_t = wav_try
                break
            except Exception as e:
                last_err = e
                codes, feat_shape = None, None
        if codes is None or feat_shape is None:
            raise last_err if last_err is not None else RuntimeError("ESC encode failed")

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

        recon = model.decode(codes, feat_shape).squeeze(0)
        recon = recon[..., :orig_len]
        if esc_sr != sr:
            recon = TAF.resample(recon.unsqueeze(0), esc_sr, sr).squeeze(0)
        return recon.cpu().numpy().astype(np.float64)
    except Exception as e:
        print(f"[ESC] encode/decode failed: {e}", flush=True)
        return None


def build_models(args, device):
    st_model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
    st_model.eval().to(device)

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

    spk_encoder = PretrainedSpeakerEncoder(
        emb_dim=getattr(saved_args, "spk_dim", 256),
        save_dir=os.path.join(os.path.dirname(args.flow_ckpt), "spkrec-ecapa"),
    ).to(device)
    spk_encoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["spk_encoder"].items()}, strict=False)
    spk_encoder.eval()
    return st_model, flow_model, spk_encoder


@torch.no_grad()
def build_candidate_latents(st_model, wav: torch.Tensor, n_layers: int, p_loss: float, device):
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

    decoded_layers = [safe_decode(i) for i in range(n_layers)]
    q1_recv = (torch.rand(t_enc, device=device) >= p_loss)
    latent = decoded_layers[0]
    if not bool(q1_recv.all()):
        latent = _latent_linear_interp(latent, q1_recv)

    for l in range(1, n_layers):
        recv_mask = (torch.rand(t_enc, device=device) >= p_loss).float()
        latent = latent + decoded_layers[l] * recv_mask.unsqueeze(0).unsqueeze(0)
    return latent


@torch.no_grad()
def decode_ours_flow_only(wav, ref_wav, plr, n_layers, st_model, flow_model, spk_encoder, device, n_steps):
    latent_in = build_candidate_latents(st_model, wav, n_layers=n_layers, p_loss=plr, device=device)
    spk_emb = spk_encoder(ref_wav.to(device))
    latent_8 = flow_sample(flow_model, latent_in, spk_emb, n_steps=n_steps, n_layers=n_layers)
    latent_8 = latent_8[..., :latent_in.shape[-1]]
    wav_rec = st_model.decoder(latent_8).squeeze().detach().cpu().numpy().astype(np.float64)
    return wav_rec


def eval_part1_rate(args, st_model, flow_model, spk_encoder, files, spk2f, device):
    out_dir = os.path.join(args.out_dir, "part1_rate")
    os.makedirs(out_dir, exist_ok=True)
    sr = st_model.sample_rate

    selected = files[: args.num_samples]
    rows = []
    details = []

    for n_layers in args.n_layers_list:
        ref_list, deg_list = [], []
        for fpath in selected:
            wav = load_audio(fpath, sr)
            ref_wav = pick_ref_wav(fpath, spk2f, sr)
            deg_np = decode_ours_flow_only(
                wav, ref_wav, args.rate_plr, n_layers, st_model, flow_model, spk_encoder, device, args.n_steps
            )
            ref_np = wav.squeeze(0).numpy().astype(np.float64)
            ref_list.append(ref_np)
            deg_list.append(deg_np)
            details.append({
                "file": fpath,
                "n_layers": int(n_layers),
                "plr": float(args.rate_plr),
                "bitrate_kbps": float(n_layers * 0.5),
            })

        visqol_vals = _calc_visqol_batch_mixed(
            ref_list, deg_list, sr, args.visqol_workers, target_sr=16000, mode="speech"
        )
        rows.append({
            "n_layers": int(n_layers),
            "plr": float(args.rate_plr),
            "bitrate_kbps": float(n_layers * 0.5),
            "visqol": nanmean(visqol_vals),
        })
        print(f"[part1] N={n_layers} rate={n_layers*0.5:.1f} VISQOL={rows[-1]['visqol']:.3f}", flush=True)

    _save_csv(rows, os.path.join(out_dir, "rate_data.csv"))
    _save_csv(details, os.path.join(out_dir, "rate_details.csv"))
    _save_json(rows, os.path.join(out_dir, "rate_data.json"))

    comp_rows = []
    for name, cfg in RATE_COMPETITORS.items():
        for idx, bw in enumerate(cfg["bitrates"]):
            ref_list, deg_list = [], []
            for fpath in selected:
                wav = load_audio(fpath, sr)
                ref_np = wav.squeeze(0).numpy().astype(np.float64)
                if cfg["type"] == "encodec":
                    deg_np = encodec_with_plr(wav.squeeze(0), sr, bw, p_loss=args.rate_plr, device=device)
                elif cfg["type"] == "esc":
                    deg_np = _esc_encode_decode(ref_np, sr, n_streams=cfg["streams"][idx], p_loss=args.rate_plr)
                else:
                    deg_np = ffmpeg_codec_with_plr(wav.squeeze(0), sr, cfg["codec"], bw, p_loss=args.rate_plr)
                if deg_np is None:
                    continue
                ref_list.append(ref_np)
                deg_list.append(deg_np)
            visqol_vals = _calc_visqol_batch_mixed(
                ref_list, deg_list, sr, args.visqol_workers, target_sr=16000, mode="speech"
            )
            comp_rows.append({
                "model": name,
                "plr": float(args.rate_plr),
                "bitrate_kbps": float(bw),
                "visqol": nanmean(visqol_vals),
            })
            print(f"[part1] {name} rate={bw:.1f} VISQOL={comp_rows[-1]['visqol']:.3f}", flush=True)

    _save_csv(comp_rows, os.path.join(out_dir, "rate_competitors.csv"))
    _save_json(comp_rows, os.path.join(out_dir, "rate_competitors.json"))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ours_style = PLOT_STYLE["ours"]
    _plot_line(ax, [r["bitrate_kbps"] for r in rows], [r["visqol"] for r in rows],
               ours_style["color"], ours_style["marker"], ours_style["label"])
    for name in ["AAC", "Opus", "EnCodec", "ESC"]:
        model_rows = [r for r in comp_rows if r["model"] == name]
        style = PLOT_STYLE[name]
        _plot_line(ax, [r["bitrate_kbps"] for r in model_rows], [r["visqol"] for r in model_rows],
                   style["color"], style["marker"], style["label"])
    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel("VISQoL")
    ax.set_title(f"Part 1: VISQoL vs Bitrate (PLR={args.rate_plr:.2f})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_part1_rate_visqol.png"), dpi=200)
    plt.close(fig)


def eval_part2_plr(args, st_model, flow_model, spk_encoder, files, spk2f, device):
    out_dir = os.path.join(args.out_dir, "part2_plr")
    os.makedirs(out_dir, exist_ok=True)
    sr = st_model.sample_rate

    selected = files[: args.num_samples]
    rows = []

    for n_layers in args.plr_n_layers_list:
        for plr in PLR_LIST:
            plc_vals = []
            for fpath in selected:
                wav = load_audio(fpath, sr)
                ref_wav = pick_ref_wav(fpath, spk2f, sr)
                deg_np = decode_ours_flow_only(
                    wav, ref_wav, plr, n_layers, st_model, flow_model, spk_encoder, device, args.n_steps
                )
                ref_np = wav.squeeze(0).numpy().astype(np.float64)
                plc_vals.append(calc_plcmos(deg_np, ref_np, sr))

            rows.append({
                "n_layers": int(n_layers),
                "plr": float(plr),
                "bitrate_kbps": float(n_layers * 0.5),
                "plcmos": nanmean(plc_vals),
            })
            print(f"[part2] N={n_layers} PLR={plr:.2f} PLCMOS={rows[-1]['plcmos']:.3f}", flush=True)

    _save_csv(rows, os.path.join(out_dir, "plr_data.csv"))
    _save_json(rows, os.path.join(out_dir, "plr_data.json"))

    comp_rows = []
    comp_specs = [
        ("AAC", "ffmpeg", {"codec": "aac", "bitrate": args.plr_aac_bitrate}),
        ("Opus", "ffmpeg", {"codec": "libopus", "bitrate": args.plr_opus_bitrate}),
        ("EnCodec", "encodec", {"bitrate": args.plr_encodec_bitrate}),
        ("ESC", "esc", {"streams": args.plr_esc_streams, "bitrate": args.plr_esc_bitrate}),
    ]
    for name, typ, spec in comp_specs:
        for plr in PLR_LIST:
            plc_vals = []
            for fpath in selected:
                wav = load_audio(fpath, sr)
                ref_np = wav.squeeze(0).numpy().astype(np.float64)
                if typ == "encodec":
                    deg_np = encodec_with_plr(wav.squeeze(0), sr, spec["bitrate"], p_loss=plr, device=device)
                elif typ == "esc":
                    deg_np = _esc_encode_decode(ref_np, sr, n_streams=spec["streams"], p_loss=plr)
                else:
                    deg_np = ffmpeg_codec_with_plr(wav.squeeze(0), sr, spec["codec"], spec["bitrate"], p_loss=plr)
                if deg_np is None:
                    continue
                plc_vals.append(calc_plcmos(deg_np, ref_np, sr))
            comp_rows.append({
                "model": name,
                "plr": float(plr),
                "bitrate_kbps": float(spec["bitrate"]),
                "plcmos": nanmean(plc_vals),
            })
            print(f"[part2] {name} PLR={plr:.2f} PLCMOS={comp_rows[-1]['plcmos']:.3f}", flush=True)

    _save_csv(comp_rows, os.path.join(out_dir, "plr_competitors.csv"))
    _save_json(comp_rows, os.path.join(out_dir, "plr_competitors.json"))

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for n_layers in args.plr_n_layers_list:
        model_rows = [r for r in rows if r["n_layers"] == n_layers]
        _plot_line(
            ax,
            [r["plr"] for r in model_rows],
            [r["plcmos"] for r in model_rows],
            color="#d62728",
            marker="o",
            label=f"Proposed ({n_layers * 0.5:.1f} kbps)",
        )
    for name in ["AAC", "Opus", "EnCodec", "ESC"]:
        model_rows = [r for r in comp_rows if r["model"] == name]
        style = PLOT_STYLE[name]
        label = f"{style['label']} ({model_rows[0]['bitrate_kbps']:.1f} kbps)" if model_rows else style["label"]
        _plot_line(ax, [r["plr"] for r in model_rows], [r["plcmos"] for r in model_rows],
                   style["color"], style["marker"], label)
    ax.set_xlabel("Packet Loss Rate")
    ax.set_ylabel("PLCMOS")
    ax.set_title("Part 2: PLCMOS vs PLR")
    ax.grid(True, linestyle="--", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_part2_plr_plcmos.png"), dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Flow-only smoke evaluation")
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path", default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--flow_ckpt", default="output/flow_checkpoints_stage3/best_train.pt")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--out_dir", default="output/eval_flow_smoke")
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--rate_plr", type=float, default=0.05)
    p.add_argument("--n_layers_list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--plr_n_layers_list", type=int, nargs="+", default=[4, 6, 8])
    p.add_argument("--plr_aac_bitrate", type=float, default=8.0)
    p.add_argument("--plr_opus_bitrate", type=float, default=8.0)
    p.add_argument("--plr_encodec_bitrate", type=float, default=6.0)
    p.add_argument("--plr_esc_streams", type=int, default=3)
    p.add_argument("--plr_esc_bitrate", type=float, default=4.5)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--visqol_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--part", choices=["all", "rate", "plr"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st_model, flow_model, spk_encoder = build_models(args, device)

    files = load_filelist(args.data_dir, args.split)
    if not files:
        raise RuntimeError(f"no audio files found under data_dir={args.data_dir} split={args.split}")
    random.shuffle(files)
    spk2f = build_spk2files(files)

    if args.part in ("all", "rate"):
        eval_part1_rate(args, st_model, flow_model, spk_encoder, files, spk2f, device)
    if args.part in ("all", "plr"):
        eval_part2_plr(args, st_model, flow_model, spk_encoder, files, spk2f, device)


if __name__ == "__main__":
    main()
