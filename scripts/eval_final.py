# -*- coding: utf-8 -*-
"""
Part-2-only final evaluation script.

This version is adapted to the new controller checkpoint and only keeps the
PLR robustness experiment:

  - x-axis: packet loss rate
  - y-axis: PLCMOS
  - systems:
      * Proposed controller + flow
      * Opus (fixed bitrate, LBRR-PLC)
      * EnCodec (fixed bitrate, LFR-PLC)
      * ESC (fixed streams, LFR-PLC)

Outputs:
  - fig_part2_plr_plcmos.png
  - part2_plr_data.csv
  - part2_plr_data.json
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
from speechtokenizer.flow import BitrateController, FlowMatchingModel, PretrainedSpeakerEncoder
from scripts.eval_utils import (
    set_seed,
    nanmean,
    load_audio,
    load_filelist,
    build_spk2files,
    pick_ref_wav,
    calc_plcmos,
    calc_visqol_batch,
    _latent_linear_interp,
    flow_sample,
    encodec_with_lfrplc,
    opus_lbrr_with_plr,
)


PLR_LIST = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_esc_model_cache = {}


SYSTEM_STYLES = {
    "ours_rmax_1p5": {"color": "#d62728", "marker": "o", "ls": "--", "lw": 1.1, "label": "Proposed (Rmax=1.5kbps)"},
    "ours_rmax_2p0": {"color": "#d62728", "marker": "o", "ls": "-.", "lw": 1.1, "label": "Proposed (Rmax=2.0kbps)"},
    "ours_rmax_3p0": {"color": "#d62728", "marker": "o", "ls": "-", "lw": 1.2, "label": "Proposed (Rmax=3.0kbps)"},
    "ours_rmax_4p0": {"color": "#d62728", "marker": "o", "ls": ":", "lw": 1.2, "label": "Proposed (Rmax=4.0kbps)"},
    "opus_8k": {"color": "#2ca02c", "marker": "^", "ls": "-", "lw": 1.1, "label": "Opus (8.0kbps, LBRR-PLC)"},
    "encodec_6k": {"color": "#17becf", "marker": "D", "ls": "-", "lw": 1.1, "label": "EnCodec (6.0kbps, LFR-PLC)"},
    "esc_1p5k": {"color": "#e377c2", "marker": "P", "ls": "--", "lw": 1.1, "label": "ESC (1.5kbps, LFR-PLC)"},
    "esc_3p0k": {"color": "#e377c2", "marker": "P", "ls": "-", "lw": 1.1, "label": "ESC (3.0kbps, LFR-PLC)"},
    "esc_4p5k": {"color": "#e377c2", "marker": "P", "ls": "-.", "lw": 1.1, "label": "ESC (4.5kbps, LFR-PLC)"},
}


def _slug(x: float) -> str:
    return str(x).replace(".", "p")


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


def _plot_line(ax, xs, ys, style):
    pairs = [(x, y) for x, y in zip(xs, ys) if not np.isnan(float(y))]
    if not pairs:
        return
    px, py = zip(*pairs)
    ax.plot(
        px, py,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["ls"],
        linewidth=style["lw"],
        markersize=4,
        markerfacecolor="none",
        markeredgewidth=1.0,
        label=style["label"],
    )


def _resample_np_for_visqol(wav_np: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav_np.astype(np.float64)
    import torchaudio.functional as TAF
    wav_t = torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)
    wav_rs = TAF.resample(wav_t, src_sr, dst_sr).squeeze(0).cpu().numpy()
    return wav_rs.astype(np.float64)


def _calc_visqol_batch_mixed(ref_list, deg_list, src_sr: int, target_sr: int, mode: str, n_workers: int) -> list:
    if not ref_list:
        return []
    ref_rs = [_resample_np_for_visqol(r, src_sr, target_sr) for r in ref_list]
    deg_rs = [_resample_np_for_visqol(d, src_sr, target_sr) for d in deg_list]
    return calc_visqol_batch(ref_rs, deg_rs, target_sr, n_workers=n_workers, mode=mode)


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


def load_models(args, device):
    st_model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
    st_model.eval().to(device)

    flow_ckpt = torch.load(args.flow_ckpt, map_location="cpu")
    saved_args = argparse.Namespace(**flow_ckpt.get("args", {}))
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
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt["flow_model"].items()})
    flow_model.eval()

    spk_encoder = PretrainedSpeakerEncoder(
        emb_dim=getattr(saved_args, "spk_dim", 256),
        save_dir=os.path.join(os.path.dirname(args.flow_ckpt), "spkrec-ecapa"),
    ).to(device)
    spk_encoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt["spk_encoder"].items()}, strict=False)
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
        # Some ESC variants require the time dimension to satisfy a stricter
        # internal overlap multiple than the nominal 10 ms frame alignment.
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
        wav_out = model.decode(codes, feat_shape)
        wav_dec = wav_out.squeeze().cpu().numpy()
        wav_dec = wav_dec[:orig_len]
        if esc_sr != sr:
            wav_dec_t = torch.from_numpy(wav_dec.astype(np.float32)).unsqueeze(0)
            wav_dec = TAF.resample(wav_dec_t, esc_sr, sr).squeeze().numpy()
        return wav_dec.astype(np.float64)
    except Exception as e:
        print(f"[ESC] inference failed: {e}", flush=True)
        return None


@torch.no_grad()
def decode_ours_controller(
    wav,
    ref_wav,
    plr,
    rmax,
    st_model,
    flow_model,
    spk_encoder,
    controller,
    candidate_layers,
    device,
    n_steps,
):
    latent_by_n = build_candidate_latents(st_model, wav, candidate_layers, p_loss=plr, device=device)
    anchor_n = candidate_layers[0]
    obs_feat = extract_obs_features(latent_by_n[anchor_n]).unsqueeze(0).to(device)
    spk_emb = spk_encoder(ref_wav.to(device))
    score = controller(
        torch.tensor([plr], device=device),
        r_max=torch.tensor([rmax], device=device),
        obs_feat=obs_feat,
    )
    rate_vec = torch.tensor([n * 0.5 for n in candidate_layers], device=device, dtype=torch.float32)
    feasible = rate_vec <= (rmax + 1e-6)
    masked_score = score.masked_fill(~feasible.unsqueeze(0), -1e9)
    pred_idx = int(masked_score.argmax(dim=1).item())
    n_layers = int(candidate_layers[pred_idx])
    latent_in = latent_by_n[n_layers]
    latent_8 = flow_sample(flow_model, latent_in, spk_emb, n_steps=n_steps, n_layers=n_layers)
    latent_8 = latent_8[..., :latent_in.shape[-1]]
    wav_rec = st_model.decoder(latent_8).squeeze().detach().cpu().numpy().astype(np.float64)
    return wav_rec, n_layers, float(rate_vec[pred_idx].item())


def eval_part2(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st_model, flow_model, spk_encoder, controller, candidate_layers = load_models(args, device)
    sr = st_model.sample_rate

    files = load_filelist(args.data_dir, args.split)
    if not files:
        raise RuntimeError(f"no audio files found under data_dir={args.data_dir} split={args.split}")
    random.shuffle(files)
    files = files[: args.num_samples]
    spk2f = build_spk2files(files)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    systems = {
        f"ours_rmax_{_slug(rmax)}": {"plcmos": [], "visqol": [], "avg_rate_kbps": [], "label": SYSTEM_STYLES[f'ours_rmax_{_slug(rmax)}']["label"]}
        for rmax in args.ours_rmax_values
    }
    systems["opus_8k"] = {"plcmos": [], "visqol": [], "avg_rate_kbps": [], "label": SYSTEM_STYLES["opus_8k"]["label"]}
    systems["encodec_6k"] = {"plcmos": [], "visqol": [], "avg_rate_kbps": [], "label": SYSTEM_STYLES["encodec_6k"]["label"]}
    systems["esc_1p5k"] = {"plcmos": [], "visqol": [], "avg_rate_kbps": [], "label": SYSTEM_STYLES["esc_1p5k"]["label"]}
    systems["esc_3p0k"] = {"plcmos": [], "visqol": [], "avg_rate_kbps": [], "label": SYSTEM_STYLES["esc_3p0k"]["label"]}
    systems["esc_4p5k"] = {"plcmos": [], "visqol": [], "avg_rate_kbps": [], "label": SYSTEM_STYLES["esc_4p5k"]["label"]}

    details = []

    for plr in PLR_LIST:
        print(f"[part2] PLR={plr:.2f}", flush=True)

        for rmax in args.ours_rmax_values:
            key = f"ours_rmax_{_slug(rmax)}"
            plc_list = []
            ref_vis_list = []
            deg_vis_list = []
            rate_list = []
            for fpath in files:
                wav = load_audio(fpath, sr)
                ref_wav = pick_ref_wav(fpath, spk2f, sr)
                deg_np, n_layers, realized_rate = decode_ours_controller(
                    wav,
                    ref_wav,
                    plr,
                    rmax,
                    st_model,
                    flow_model,
                    spk_encoder,
                    controller,
                    candidate_layers,
                    device,
                    args.n_steps,
                )
                ref_np = wav.squeeze(0).numpy().astype(np.float64)
                plc_val = calc_plcmos(deg_np, ref_np, sr)
                plc_list.append(plc_val)
                ref_vis_list.append(ref_np)
                deg_vis_list.append(deg_np)
                rate_list.append(realized_rate)
                details.append({
                    "system": key,
                    "file": fpath,
                    "plr": float(plr),
                    "target_rmax_kbps": float(rmax),
                    "selected_rate_kbps": float(realized_rate),
                    "plcmos": float(plc_val),
                    "visqol": float("nan"),
                    "selected_n_layers": int(n_layers),
                })
            systems[key]["plcmos"].append(nanmean(plc_list))
            visqol_vals = _calc_visqol_batch_mixed(
                ref_vis_list,
                deg_vis_list,
                src_sr=sr,
                target_sr=args.ours_visqol_sr,
                mode=args.ours_visqol_mode,
                n_workers=args.visqol_workers,
            )
            systems[key]["visqol"].append(nanmean(visqol_vals))
            for i in range(len(visqol_vals)):
                details[-len(visqol_vals) + i]["visqol"] = float(visqol_vals[i])
            systems[key]["avg_rate_kbps"].append(nanmean(rate_list))
            print(
                f"  {systems[key]['label']:32s} avg_rate={systems[key]['avg_rate_kbps'][-1]:.3f} "
                f"PLCMOS={systems[key]['plcmos'][-1]:.3f} VISQOL={systems[key]['visqol'][-1]:.3f}",
                flush=True,
            )

        baseline_specs = [
            ("opus_8k", 8.0, lambda w, s: opus_lbrr_with_plr(w, s, 8.0, plr)),
            ("encodec_6k", 6.0, lambda w, s: encodec_with_lfrplc(w, s, 6.0, plr, device=device)),
            ("esc_1p5k", 1.5, lambda w, s: _esc_encode_decode(w.numpy(), s, 1, p_loss=plr)),
            ("esc_3p0k", 3.0, lambda w, s: _esc_encode_decode(w.numpy(), s, 2, p_loss=plr)),
            ("esc_4p5k", 4.5, lambda w, s: _esc_encode_decode(w.numpy(), s, 3, p_loss=plr)),
        ]
        for key, fixed_rate, fn in baseline_specs:
            plc_list = []
            ref_vis_list = []
            deg_vis_list = []
            for fpath in files:
                wav = load_audio(fpath, sr)
                ref_np = wav.squeeze(0).numpy().astype(np.float64)
                deg_np = fn(wav.squeeze(0), sr)
                if deg_np is None:
                    plc_list.append(float("nan"))
                    continue
                plc_list.append(calc_plcmos(deg_np, ref_np, sr))
                ref_vis_list.append(ref_np)
                deg_vis_list.append(deg_np)
                details.append({
                    "system": key,
                    "file": fpath,
                    "plr": float(plr),
                    "target_rmax_kbps": float("nan"),
                    "selected_rate_kbps": float(fixed_rate),
                    "plcmos": float(plc_list[-1]),
                    "visqol": float("nan"),
                    "selected_n_layers": -1,
                })
            systems[key]["plcmos"].append(nanmean(plc_list))
            visqol_vals = _calc_visqol_batch_mixed(
                ref_vis_list,
                deg_vis_list,
                src_sr=sr,
                target_sr=args.base_visqol_sr,
                mode=args.base_visqol_mode,
                n_workers=args.visqol_workers,
            )
            systems[key]["visqol"].append(nanmean(visqol_vals))
            valid_detail_indices = [idx for idx in range(len(details)) if details[idx]["system"] == key and details[idx]["plr"] == float(plr)]
            for idx, vv in zip(valid_detail_indices[-len(visqol_vals):], visqol_vals):
                details[idx]["visqol"] = float(vv)
            systems[key]["avg_rate_kbps"].append(float(fixed_rate))
            print(
                f"  {systems[key]['label']:32s} avg_rate={fixed_rate:.3f} "
                f"PLCMOS={systems[key]['plcmos'][-1]:.3f} VISQOL={systems[key]['visqol'][-1]:.3f}",
                flush=True,
            )

    rows = []
    for key, info in systems.items():
        for i, plr in enumerate(PLR_LIST):
            rows.append({
                "system": key,
                "label": info["label"],
                "plr": float(plr),
                "plcmos": float(info["plcmos"][i]),
                "visqol": float(info["visqol"][i]),
                "avg_rate_kbps": float(info["avg_rate_kbps"][i]),
            })

    _save_csv(rows, os.path.join(out_dir, "part2_plr_data.csv"))
    _save_csv(details, os.path.join(out_dir, "part2_plr_details.csv"))
    _save_json(rows, os.path.join(out_dir, "part2_plr_data.json"))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plot_order = [f"ours_rmax_{_slug(r)}" for r in args.ours_rmax_values] + ["opus_8k", "encodec_6k", "esc_1p5k", "esc_3p0k", "esc_4p5k"]
    for key in plot_order:
        if key not in systems:
            continue
        _plot_line(ax, PLR_LIST, systems[key]["plcmos"], SYSTEM_STYLES[key])
    ax.set_xlabel("Packet Loss Rate")
    ax.set_ylabel("PLCMOS")
    ax.set_title("Part 2: PLCMOS vs PLR")
    ax.set_xticks(PLR_LIST)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_part2_plr_plcmos.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key in plot_order:
        if key not in systems:
            continue
        _plot_line(ax, PLR_LIST, systems[key]["visqol"], SYSTEM_STYLES[key])
    ax.set_xlabel("Packet Loss Rate")
    ax.set_ylabel("VISQoL")
    ax.set_title("Part 2: VISQoL vs PLR")
    ax.set_xticks(PLR_LIST)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_part2_plr_visqol.png"), dpi=220)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="eval_final: part-2-only controller evaluation")
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path", default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--flow_ckpt", default="output/flow_checkpoints_stage3/best_train.pt")
    p.add_argument("--controller_ckpt", default="output/controller_checkpoints/best_controller.pt")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--out_dir", default="output/eval_final_part2")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--ours_rmax_values", type=float, nargs="+", default=[1.5, 2.0, 3.0, 4.0])
    p.add_argument("--n_layers_list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--obs_feat_dim", type=int, default=6)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--visqol_workers", type=int, default=1)
    p.add_argument("--ours_visqol_sr", type=int, default=48000)
    p.add_argument("--ours_visqol_mode", default="audio")
    p.add_argument("--base_visqol_sr", type=int, default=16000)
    p.add_argument("--base_visqol_mode", default="speech")
    return p.parse_args()


if __name__ == "__main__":
    eval_part2(parse_args())
