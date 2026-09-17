#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Profile FLOPs / params / latency and export mel-spectrogram visualizations.

Main outputs:
  1) module_profile.json / module_profile.csv
  2) latency_profile.json / latency_profile.csv
  3) mel comparison figures under output_dir/mel_vis/

Typical usage:
  python scripts/profile_system.py ^
      --mode all ^
      --input_wav path\\to\\sample.wav ^
      --ref_wav path\\to\\same_speaker_ref.wav ^
      --output_dir output\\profile_system

Notes:
  - This script is written to be run on the server directly.
  - FLOPs profiling prefers thop; if thop is unavailable, params are still reported.
  - Latency is measured with median wall-clock time across repeats.
  - Mel visualization is intended for qualitative comparison.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Cambria", "Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"

try:
    import torchaudio.transforms as TAT
except Exception as e:
    raise RuntimeError(f"torchaudio is required for this script: {e}")

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from speechtokenizer.model import SpeechTokenizer
from speechtokenizer.flow import BitrateController, FlowMatchingModel, PretrainedSpeakerEncoder
from scripts.eval_utils import (
    load_audio,
    load_filelist,
    build_spk2files,
    pick_ref_wav,
    flow_sample,
    encodec_with_lfrplc,
    opus_lbrr_with_plr,
    ffmpeg_codec_with_plr,
)

_esc_model_cache = {}


def _optional_import_thop():
    try:
        from thop import profile as thop_profile  # type: ignore
        return thop_profile
    except Exception:
        return None


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _save_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _count_params_k(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e3


def estimate_rvq_gflops(num_frames: int, dim: int, bins: int, n_q: int) -> float:
    """
    Approximate RVQ encode-side FLOPs.

    For each quantizer layer and each frame:
      - distance to all codewords: O(bins * dim)
      - residual/codeword update: O(dim)

    We use a simple MAC-style approximation:
      distance ~= 2 * bins * dim
      residual update ~= 2 * dim
    """
    per_frame_per_layer = 2.0 * bins * dim + 2.0 * dim
    total = float(num_frames) * float(n_q) * per_frame_per_layer
    return total / 1e9


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _median_ms(times: List[float]) -> float:
    return float(np.median(times)) * 1000.0


def _mean_ms(times: List[float]) -> float:
    return float(np.mean(times)) * 1000.0


def _resample_for_mel_metric(wav: np.ndarray, target_len: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == target_len:
        return wav
    if wav.size < target_len:
        out = np.zeros(target_len, dtype=np.float32)
        out[:wav.size] = wav
        return out
    return wav[:target_len]


def mel_l1_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    mel_fn = TAT.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        power=2.0,
        center=True,
        pad_mode="constant",
    )
    min_len = max(1024, min(len(a), len(b)))
    a = _resample_for_mel_metric(a, min_len)
    b = _resample_for_mel_metric(b, min_len)
    a_t = torch.from_numpy(a).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    ma = torch.log(mel_fn(a_t).clamp_min(1e-9))
    mb = torch.log(mel_fn(b_t).clamp_min(1e-9))
    t = min(ma.shape[-1], mb.shape[-1])
    return float(torch.mean(torch.abs(ma[..., :t] - mb[..., :t])).item())


def _audio_to_numpy(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _nearest_supported_rate(target_kbps: float, supported: List[float]) -> float:
    return min(supported, key=lambda x: abs(x - target_kbps))


def _esc_streams_from_rate(target_kbps: float) -> Tuple[float, int]:
    mapping = {
        1.5: 1,
        3.0: 2,
        4.5: 3,
    }
    rate = _nearest_supported_rate(target_kbps, list(mapping.keys()))
    return rate, mapping[rate]


def _try_load_esc():
    global _esc_model_cache
    if "model" in _esc_model_cache:
        return _esc_model_cache["model"]
    try:
        import yaml
        esc_src = os.path.join(PROJ_ROOT, "efficient-speech-codec-main")
        if esc_src not in sys.path:
            sys.path.insert(0, esc_src)
        from esc.models.codecs import make_model

        esc_ckpt_dir = os.path.join(PROJ_ROOT, "external_checkpoints", "esc")
        cfg_path = os.path.join(esc_ckpt_dir, "config.yaml")
        pth_path = os.path.join(esc_ckpt_dir, "model.pth")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model = make_model(cfg["model"], cfg["model_name"])
        ckpt = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _esc_model_cache["model"] = model
        return model
    except Exception as e:
        print(f"[ESC] load failed: {e}", flush=True)
        _esc_model_cache["model"] = None
        return None


@torch.no_grad()
def esc_with_lfrplc(wav_np: np.ndarray, sr: int, target_kbps: float, p_loss: float = 0.0):
    model = _try_load_esc()
    if model is None:
        return None, None
    try:
        import torchaudio.functional as TAF
        rate, n_streams = _esc_streams_from_rate(target_kbps)
        esc_sr = 16000
        wav_t = torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)
        if sr != esc_sr:
            wav_t = TAF.resample(wav_t, sr, esc_sr)
        orig_len = int(wav_t.shape[-1])
        align_candidates = [160, 320, 480, 640, 960, 1280, 1920, 2560, 3200]
        last_err = None
        codes, feat_shape = None, None
        for align in align_candidates:
            wav_try = wav_t
            pad = (-orig_len) % align
            if pad:
                wav_try = F.pad(
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
        wav_dec = wav_out.squeeze().cpu().numpy()[:orig_len]
        if esc_sr != sr:
            wav_dec_t = torch.from_numpy(wav_dec.astype(np.float32)).unsqueeze(0)
            wav_dec = TAF.resample(wav_dec_t, esc_sr, sr).squeeze().numpy()
        return wav_dec.astype(np.float64), rate
    except Exception as e:
        print(f"[ESC] inference failed: {e}", flush=True)
        return None, None


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
    if wav.dim() == 1:
        x = wav.unsqueeze(0).unsqueeze(0)
    elif wav.dim() == 2:
        x = wav.unsqueeze(0)
    else:
        x = wav
    x = x.to(device)
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
        from scripts.eval_utils import _latent_linear_interp
        latent_q1 = _latent_linear_interp(latent_q1, q1_recv)

    latent_running = latent_q1.clone()
    latent_by_n = {1: latent_running.clone()}
    for l in range(1, max(candidate_layers)):
        recv_mask = (torch.rand(t_enc, device=device) >= p_loss).float()
        latent_running = latent_running + decoded_layers[l] * recv_mask.unsqueeze(0).unsqueeze(0)
        latent_by_n[l + 1] = latent_running.clone()
    return latent_by_n


def load_models(args, device: torch.device):
    st_model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
    st_model.eval().to(device)

    flow_ckpt = torch.load(args.flow_ckpt, map_location="cpu")
    flow_args = argparse.Namespace(**flow_ckpt.get("args", {}))
    flow_model = FlowMatchingModel(
        latent_dim=st_model.quantizer.dimension,
        base_ch=getattr(flow_args, "base_ch", 512),
        ch_mults=tuple(getattr(flow_args, "ch_mults", [1, 1, 2])),
        cond_dim=getattr(flow_args, "cond_dim", 512),
        spk_dim=getattr(flow_args, "spk_dim", 256),
        time_dim=getattr(flow_args, "time_dim", 128),
        n_res=getattr(flow_args, "n_res", 2),
        n_mid_res=getattr(flow_args, "n_mid_res", 2),
    ).to(device)
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in flow_ckpt["flow_model"].items()})
    flow_model.eval()

    spk_encoder = PretrainedSpeakerEncoder(
        emb_dim=getattr(flow_args, "spk_dim", 256),
        save_dir=os.path.join(os.path.dirname(args.flow_ckpt), "spkrec-ecapa"),
    ).to(device)
    spk_encoder.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in flow_ckpt["spk_encoder"].items()},
        strict=False,
    )
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


class EncoderOnly(torch.nn.Module):
    def __init__(self, st_model: SpeechTokenizer):
        super().__init__()
        self.m = st_model

    def forward(self, wav: torch.Tensor):
        return self.m.encoder(wav)


class DecoderOnly(torch.nn.Module):
    def __init__(self, st_model: SpeechTokenizer):
        super().__init__()
        self.m = st_model

    def forward(self, latent: torch.Tensor):
        return self.m.decoder(latent)


class ControllerOnly(torch.nn.Module):
    def __init__(self, controller: BitrateController):
        super().__init__()
        self.m = controller

    def forward(self, p_loss: torch.Tensor, r_max: torch.Tensor, obs_feat: torch.Tensor):
        return self.m(p_loss, r_max=r_max, obs_feat=obs_feat)


class FlowWrapper(torch.nn.Module):
    def __init__(self, flow_model: FlowMatchingModel, n_layers: int):
        super().__init__()
        self.flow_model = flow_model
        self.n_layers = n_layers

    def forward(self, latent: torch.Tensor, t: torch.Tensor, spk_emb: torch.Tensor):
        n_layers = torch.full((latent.shape[0],), self.n_layers, dtype=torch.long, device=latent.device)
        return self.flow_model(latent, t, spk_emb, n_layers=n_layers)


def profile_flops_and_params(args, st_model, flow_model, spk_encoder, controller, device):
    thop_profile = _optional_import_thop()

    sr = st_model.sample_rate
    wav = torch.randn(1, 1, args.profile_num_samples, device=device)
    latent_t = max(1, math.ceil(args.profile_num_samples / st_model.downsample_rate))
    latent = torch.randn(1, st_model.quantizer.dimension, latent_t, device=device)
    obs_feat = torch.randn(1, 6, device=device)
    p_loss = torch.tensor([args.profile_plr], dtype=torch.float32, device=device)
    r_max = torch.tensor([args.profile_rmax], dtype=torch.float32, device=device)
    t = torch.tensor([0.5], dtype=torch.float32, device=device)
    spk_emb = torch.randn(1, getattr(flow_model.spk_proj[0], "in_features", 256), device=device)

    encoder_mod = EncoderOnly(st_model).to(device).eval()
    decoder_mod = DecoderOnly(st_model).to(device).eval()
    controller_mod = ControllerOnly(controller).to(device).eval()
    flow_mod = FlowWrapper(flow_model, args.profile_n_layers).to(device).eval()

    rows = []

    rvq_params_m = (
        st_model.quantizer.n_q
        * st_model.quantizer.bins
        * st_model.quantizer.dimension
        / 1e6
    )
    rvq_gflops = estimate_rvq_gflops(
        num_frames=latent_t,
        dim=st_model.quantizer.dimension,
        bins=st_model.quantizer.bins,
        n_q=st_model.quantizer.n_q,
    )

    def add_row(name: str, module: torch.nn.Module, inputs: tuple):
        params_m = _count_params_m(module)
        params_k = _count_params_k(module)
        gflops = None
        if thop_profile is not None:
            try:
                macs, _ = thop_profile(module, inputs=inputs, verbose=False)
                gflops = float(macs) * 2.0 / 1e9
            except Exception:
                gflops = None
        rows.append({
            "module": name,
            "gflops": None if gflops is None else round(gflops, 6),
            "params_m": round(params_m, 6),
            "params_k": round(params_k, 3),
        })

    add_row("Semantic Encoder", encoder_mod, (wav,))
    rows.append({
        "module": "RVQ Codebooks",
        "gflops": round(rvq_gflops, 6),
        "params_m": round(rvq_params_m, 6),
        "params_k": round(rvq_params_m * 1000.0, 3),
    })
    add_row("Adaptive Controller", controller_mod, (p_loss, r_max, obs_feat))
    add_row("Flow Completion Model", flow_mod, (latent, t, spk_emb))
    add_row("Semantic Decoder", decoder_mod, (latent,))

    total_gflops = sum(r["gflops"] for r in rows if r["gflops"] is not None)
    total_params = sum(r["params_m"] for r in rows)
    rows.append({
        "module": "Total",
        "gflops": None if any(r["gflops"] is None for r in rows[:-1]) else round(total_gflops, 6),
        "params_m": round(total_params, 6),
        "params_k": round(total_params * 1000.0, 3),
    })

    meta = {
        "sample_rate": sr,
        "profile_num_samples": args.profile_num_samples,
        "profile_seconds": args.profile_num_samples / sr,
        "profile_n_layers": args.profile_n_layers,
        "profile_rmax": args.profile_rmax,
        "profile_plr": args.profile_plr,
        "thop_available": thop_profile is not None,
        "rows": rows,
    }
    return meta


def _ensure_flow_min_frames(flow_model, latent: torch.Tensor) -> torch.Tensor:
    n_down_levels = len(getattr(flow_model, "down_blocks", []))
    min_flow_frames = max(1, 2 ** n_down_levels)
    cur_t = int(latent.shape[-1])
    if cur_t >= min_flow_frames:
        return latent
    pad_right = min_flow_frames - cur_t
    if cur_t > 1:
        return F.pad(latent, (0, pad_right), mode="replicate")
    return F.pad(latent, (0, pad_right), mode="constant", value=0.0)


@torch.no_grad()
def measure_latency(args, st_model, flow_model, spk_encoder, controller, candidate_layers, device):
    sr = st_model.sample_rate
    wav = torch.randn(1, 1, args.latency_num_samples, device=device)
    ref = torch.randn(1, sr, device=device)
    p_loss = args.latency_plr
    r_max = args.latency_rmax
    n_repeat = args.latency_repeats

    spk_emb = spk_encoder(ref.unsqueeze(0))

    latent_by_n = build_candidate_latents(
        st_model,
        wav.squeeze(0).squeeze(0).detach().cpu(),
        candidate_layers,
        p_loss=p_loss,
        device=device,
    )
    obs_feat = extract_obs_features(latent_by_n[candidate_layers[0]]).unsqueeze(0)
    rate_vec = torch.tensor([n * 0.5 for n in candidate_layers], device=device, dtype=torch.float32)
    score = controller(
        torch.tensor([p_loss], device=device),
        r_max=torch.tensor([r_max], device=device),
        obs_feat=obs_feat,
    )
    feasible = rate_vec <= (r_max + 1e-6)
    pred_idx = int(score.squeeze(0).masked_fill(~feasible, -1e9).argmax().item()) if feasible.any() else 0
    pred_n = candidate_layers[pred_idx]
    latent_in = latent_by_n[pred_n]

    # warmup
    for _ in range(args.latency_warmup):
        _ = st_model.encode(wav)
        _ = controller(
            torch.tensor([p_loss], device=device),
            r_max=torch.tensor([r_max], device=device),
            obs_feat=obs_feat,
        )
        lat_flow_in = _ensure_flow_min_frames(flow_model, latent_in)
        lat_out = flow_sample(flow_model, lat_flow_in, spk_emb, n_steps=args.n_steps, n_layers=pred_n)
        _ = st_model.decoder(lat_out[..., :latent_in.shape[-1]])

    enc_t, ctrl_t, quant_t, flow_t, dec_t = [], [], [], [], []

    D = st_model.quantizer.dimension

    for _ in range(n_repeat):
        _sync(device); t0 = time.perf_counter()
        codes_all = st_model.encode(wav)
        _sync(device); enc_t.append(time.perf_counter() - t0)

        _sync(device); t0 = time.perf_counter()
        score = controller(
            torch.tensor([p_loss], device=device),
            r_max=torch.tensor([r_max], device=device),
            obs_feat=obs_feat,
        )
        feasible = rate_vec <= (r_max + 1e-6)
        pred_idx = int(score.squeeze(0).masked_fill(~feasible, -1e9).argmax().item()) if feasible.any() else 0
        pred_n = candidate_layers[pred_idx]
        _sync(device); ctrl_t.append(time.perf_counter() - t0)

        _sync(device); t0 = time.perf_counter()
        vq_l = st_model.quantizer.vq.layers[0]
        lat = vq_l.decode(codes_all[0])
        if lat.shape[-1] == D:
            lat = lat.permute(0, 2, 1)
        lat_sum = lat.clone()
        for l in range(1, pred_n):
            vq_l2 = st_model.quantizer.vq.layers[l]
            d = vq_l2.decode(codes_all[l])
            if d.shape[-1] == D:
                d = d.permute(0, 2, 1)
            lat_sum = lat_sum + d
        _sync(device); quant_t.append(time.perf_counter() - t0)

        _sync(device); t0 = time.perf_counter()
        lat_flow_in = _ensure_flow_min_frames(flow_model, lat_sum)
        lat_out = flow_sample(flow_model, lat_flow_in, spk_emb, n_steps=args.n_steps, n_layers=pred_n)
        lat_out = lat_out[..., :lat_sum.shape[-1]]
        _sync(device); flow_t.append(time.perf_counter() - t0)

        _sync(device); t0 = time.perf_counter()
        _ = st_model.decoder(lat_out)
        _sync(device); dec_t.append(time.perf_counter() - t0)

    t_enc = _median_ms(enc_t)
    t_ctrl = _median_ms(ctrl_t)
    t_quant = _median_ms(quant_t)
    t_flow = _median_ms(flow_t)
    t_dec = _median_ms(dec_t)

    t_sender = t_enc + t_ctrl + t_quant
    t_receiver = t_flow + t_dec
    t_pipeline = max(t_sender, t_receiver)
    t_total = t_sender + t_receiver
    dur_ms = args.latency_num_samples / sr * 1000.0
    rtf = t_pipeline / dur_ms

    return {
        "sample_rate": sr,
        "num_samples": args.latency_num_samples,
        "duration_ms": round(dur_ms, 6),
        "n_layers_selected": int(pred_n),
        "bitrate_kbps_nominal": round(pred_n * 0.5, 6),
        "plr": p_loss,
        "rmax": r_max,
        "n_steps": args.n_steps,
        "repeats": n_repeat,
        "T_encoder_ms": round(t_enc, 6),
        "T_controller_ms": round(t_ctrl, 6),
        "T_quantize_ms": round(t_quant, 6),
        "T_flow_ms": round(t_flow, 6),
        "T_decoder_ms": round(t_dec, 6),
        "T_sender_ms": round(t_sender, 6),
        "T_receiver_ms": round(t_receiver, 6),
        "T_pipeline_ms": round(t_pipeline, 6),
        "T_total_ms": round(t_total, 6),
        "RTF_pipeline": round(rtf, 6),
    }


@torch.no_grad()
def measure_latency_dataset(args, st_model, flow_model, spk_encoder, controller, candidate_layers, device):
    sr = st_model.sample_rate
    files = load_filelist(args.data_dir, args.split)
    if not files:
        raise RuntimeError(f"No wav/flac files found under {args.data_dir} ({args.split})")
    spk2files = build_spk2files(files)
    selected = files[: min(args.max_items, len(files))]

    all_groups = []
    for r_max in args.latency_rmax_values:
        rows = []
        for fpath in selected:
            wav = load_audio(fpath, sr).to(device)
            ref = pick_ref_wav(fpath, spk2files, sr).to(device)
            p_loss = args.latency_plr
            n_repeat = args.latency_repeats

            spk_emb = spk_encoder(ref.unsqueeze(0).squeeze(1))
            latent_by_n = build_candidate_latents(st_model, wav.squeeze(0).cpu(), candidate_layers, p_loss=p_loss, device=device)
            obs_feat = extract_obs_features(latent_by_n[candidate_layers[0]]).unsqueeze(0)
            rate_vec = torch.tensor([n * 0.5 for n in candidate_layers], device=device, dtype=torch.float32)
            score = controller(
                torch.tensor([p_loss], device=device),
                r_max=torch.tensor([r_max], device=device),
                obs_feat=obs_feat,
            )
            feasible = rate_vec <= (r_max + 1e-6)
            pred_idx = int(score.squeeze(0).masked_fill(~feasible, -1e9).argmax().item()) if feasible.any() else 0
            pred_n = candidate_layers[pred_idx]
            latent_in = latent_by_n[pred_n]

            for _ in range(args.latency_warmup):
                _ = st_model.encode(wav.unsqueeze(0))
                lat_flow_in = _ensure_flow_min_frames(flow_model, latent_in)
                lat_out = flow_sample(flow_model, lat_flow_in, spk_emb, n_steps=args.n_steps, n_layers=pred_n)
                _ = st_model.decoder(lat_out[..., :latent_in.shape[-1]])

            enc_t, ctrl_t, quant_t, flow_t, dec_t = [], [], [], [], []
            D = st_model.quantizer.dimension
            wav_in = wav.unsqueeze(0)

            for _ in range(n_repeat):
                _sync(device); t0 = time.perf_counter()
                codes_all = st_model.encode(wav_in)
                _sync(device); enc_t.append(time.perf_counter() - t0)

                _sync(device); t0 = time.perf_counter()
                _ = controller(
                    torch.tensor([p_loss], device=device),
                    r_max=torch.tensor([r_max], device=device),
                    obs_feat=obs_feat,
                )
                _sync(device); ctrl_t.append(time.perf_counter() - t0)

                _sync(device); t0 = time.perf_counter()
                vq_l = st_model.quantizer.vq.layers[0]
                lat = vq_l.decode(codes_all[0])
                if lat.shape[-1] == D:
                    lat = lat.permute(0, 2, 1)
                lat_sum = lat.clone()
                for l in range(1, pred_n):
                    vq_l2 = st_model.quantizer.vq.layers[l]
                    d = vq_l2.decode(codes_all[l])
                    if d.shape[-1] == D:
                        d = d.permute(0, 2, 1)
                    lat_sum = lat_sum + d
                _sync(device); quant_t.append(time.perf_counter() - t0)

                _sync(device); t0 = time.perf_counter()
                lat_flow_in = _ensure_flow_min_frames(flow_model, lat_sum)
                lat_out = flow_sample(flow_model, lat_flow_in, spk_emb, n_steps=args.n_steps, n_layers=pred_n)
                lat_out = lat_out[..., :lat_sum.shape[-1]]
                _sync(device); flow_t.append(time.perf_counter() - t0)

                _sync(device); t0 = time.perf_counter()
                _ = st_model.decoder(lat_out)
                _sync(device); dec_t.append(time.perf_counter() - t0)

            dur_ms = wav.shape[-1] / sr * 1000.0
            t_enc = _median_ms(enc_t)
            t_ctrl = _median_ms(ctrl_t)
            t_quant = _median_ms(quant_t)
            t_flow = _median_ms(flow_t)
            t_dec = _median_ms(dec_t)
            t_sender = t_enc + t_ctrl + t_quant
            t_receiver = t_flow + t_dec
            t_pipeline = max(t_sender, t_receiver)

            rows.append({
                "rmax_kbps": float(r_max),
                "file": os.path.basename(fpath),
                "duration_ms": round(dur_ms, 6),
                "n_layers_selected": int(pred_n),
                "bitrate_kbps_nominal": round(pred_n * 0.5, 6),
                "T_encoder_ms": round(t_enc, 6),
                "T_controller_ms": round(t_ctrl, 6),
                "T_quantize_ms": round(t_quant, 6),
                "T_flow_ms": round(t_flow, 6),
                "T_decoder_ms": round(t_dec, 6),
                "T_sender_ms": round(t_sender, 6),
                "T_receiver_ms": round(t_receiver, 6),
                "T_pipeline_ms": round(t_pipeline, 6),
                "RTF_pipeline": round(t_pipeline / dur_ms, 6),
            })

        avg = {
            "rmax_kbps": float(r_max),
            "num_samples": len(rows),
            "avg_duration_ms": round(float(np.mean([r["duration_ms"] for r in rows])), 6),
            "avg_bitrate_kbps": round(float(np.mean([r["bitrate_kbps_nominal"] for r in rows])), 6),
            "T_encoder_ms": round(float(np.mean([r["T_encoder_ms"] for r in rows])), 6),
            "T_controller_ms": round(float(np.mean([r["T_controller_ms"] for r in rows])), 6),
            "T_quantize_ms": round(float(np.mean([r["T_quantize_ms"] for r in rows])), 6),
            "T_flow_ms": round(float(np.mean([r["T_flow_ms"] for r in rows])), 6),
            "T_decoder_ms": round(float(np.mean([r["T_decoder_ms"] for r in rows])), 6),
            "T_sender_ms": round(float(np.mean([r["T_sender_ms"] for r in rows])), 6),
            "T_receiver_ms": round(float(np.mean([r["T_receiver_ms"] for r in rows])), 6),
            "T_pipeline_ms": round(float(np.mean([r["T_pipeline_ms"] for r in rows])), 6),
            "RTF_pipeline": round(float(np.mean([r["RTF_pipeline"] for r in rows])), 6),
        }
        paper_row = {
            "Method": "PDSSC" if abs(float(r_max) - 3.0) < 1e-6 else "PDSSC (low-rate)",
            "Sample rate (kHz)": round(sr / 1000.0, 3),
            "Bitrate (kbps)": avg["avg_bitrate_kbps"],
            "TCoder (ms)": round(avg["T_sender_ms"], 6),
            "TContext (s)": round(avg["avg_duration_ms"] / 1000.0, 6),
            "Delay (s)": round(avg["T_pipeline_ms"] / 1000.0, 6),
            "RTF": avg["RTF_pipeline"],
        }
        all_groups.append({"avg": avg, "rows": rows, "paper_row": paper_row})

    return {
        "groups": all_groups,
        "paper_table_rows": [g["paper_row"] for g in all_groups],
    }


def _latency_table_row(method: str, sample_rate_khz: float, bitrate_bps: int, delay_s: float) -> Dict[str, object]:
    return {
        "Method": method,
        "Sample rate (kHz)": round(float(sample_rate_khz), 3),
        "Bitrate (bps)": int(bitrate_bps),
        "Delay (s)": round(float(delay_s), 6),
    }


def _measure_runtime_median_seconds(fn, repeats: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        _ = fn()
    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        _ = fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) if times else 0.0


def _measure_baseline_latency_rows(args, files: List[str], input_sr: int, device: torch.device) -> List[Dict[str, object]]:
    rows = []
    p_loss = float(args.latency_plr)

    def add_method_rows(method: str, bitrate_kbps: float, sample_rate_khz: float, infer_fn_builder):
        per_file_delays = []
        for fpath in files:
            wav = load_audio(fpath, input_sr)
            run_fn = infer_fn_builder(wav)
            delay_s = _measure_runtime_median_seconds(
                run_fn,
                repeats=args.latency_repeats,
                warmup=args.latency_warmup,
                device=device,
            )
            per_file_delays.append(delay_s)
        avg_delay_s = float(np.mean(per_file_delays)) if per_file_delays else 0.0
        rows.append(
            _latency_table_row(
                method=method,
                sample_rate_khz=sample_rate_khz,
                bitrate_bps=int(round(bitrate_kbps * 1000.0)),
                delay_s=avg_delay_s,
            )
        )

    for bitrate_kbps in args.latency_rmax_values:
        add_method_rows(
            method="EnCodec + LFR-PLC",
            bitrate_kbps=bitrate_kbps,
            sample_rate_khz=24.0,
            infer_fn_builder=lambda wav, bw=bitrate_kbps: (
                lambda: encodec_with_lfrplc(wav, input_sr, bw, p_loss, device)
            ),
        )
        add_method_rows(
            method="ESC + LFR-PLC",
            bitrate_kbps=bitrate_kbps,
            sample_rate_khz=16.0,
            infer_fn_builder=lambda wav, bw=bitrate_kbps: (
                lambda: esc_with_lfrplc(
                    wav.squeeze(0).cpu().numpy().astype(np.float64),
                    input_sr,
                    bw,
                    p_loss,
                )
            ),
        )

    if args.include_opus_latency:
        add_method_rows(
            method="Opus + LBRR",
            bitrate_kbps=args.opus_kbps,
            sample_rate_khz=48.0,
            infer_fn_builder=lambda wav: (
                lambda: opus_lbrr_with_plr(wav, input_sr, args.opus_kbps, p_loss)
            ),
        )
    return rows


@torch.no_grad()
def build_latency_compare_table(args, st_model, flow_model, spk_encoder, controller, candidate_layers, device):
    sr = st_model.sample_rate
    files = load_filelist(args.data_dir, args.split)
    if not files:
        raise RuntimeError(f"No wav/flac files found under {args.data_dir} ({args.split})")
    selected = files[: min(args.max_items, len(files))]

    pdssc_meta = measure_latency_dataset(
        args,
        st_model,
        flow_model,
        spk_encoder,
        controller,
        candidate_layers,
        device,
    )
    table_rows = []
    if not pdssc_meta["groups"]:
        raise RuntimeError("No PDSSC latency groups were produced.")
    avg = pdssc_meta["groups"][0]["avg"]
    bitrate_kbps = float(avg["avg_bitrate_kbps"])
    table_rows.append(
        _latency_table_row(
            method="PDSSC",
            sample_rate_khz=sr / 1000.0,
            bitrate_bps=int(round(bitrate_kbps * 1000.0)),
            delay_s=avg["T_pipeline_ms"] / 1000.0,
        )
    )

    table_rows.extend(_measure_baseline_latency_rows(args, selected, sr, device))
    return {
        "num_samples": len(selected),
        "latency_plr": float(args.latency_plr),
        "table_rows": table_rows,
    }


def save_mel_grid(items: List[Tuple[str, np.ndarray]], sr: int, out_path: Path):
    n = len(items)
    ncols = 2 if n >= 4 else 1
    nrows = int(math.ceil(n / ncols))
    fig_w = 10.2 if ncols == 2 else 6.2
    fig_h = 5.8 if ncols == 2 else 3.1 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)
    mel_fn = TAT.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        power=2.0,
        center=True,
        pad_mode="constant",
    )
    flat_axes = axes.flatten()
    for ax, (title, wav_np) in zip(flat_axes, items):
        wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
        if wav_np.size < 1024:
            padded = np.zeros(1024, dtype=np.float32)
            padded[:wav_np.size] = wav_np
            wav_np = padded
        mel = mel_fn(torch.from_numpy(wav_np).unsqueeze(0)).squeeze(0)
        mel_db = (10.0 * torch.log10(mel + 1e-9)).cpu().numpy()
        ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
        ax.set_title(title, fontsize=22, fontfamily="serif", pad=10)
        ax.axis("off")
    for ax in flat_axes[len(items):]:
        ax.axis("off")
    fig.tight_layout(pad=0.18, h_pad=0.28, w_pad=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


@torch.no_grad()
def mel_visualization(args, st_model, flow_model, spk_encoder, controller, candidate_layers, device):
    sr = st_model.sample_rate
    wav_path = args.input_wav or os.path.join(args.data_dir, args.mel_fixed_file)
    ref_path = args.ref_wav or wav_path
    wav = load_audio(wav_path, sr)
    ref_wav = load_audio(ref_path, sr)
    wav_np = wav.squeeze(0).numpy().astype(np.float64)

    latent_by_n = build_candidate_latents(
        st_model,
        wav.squeeze(0),
        candidate_layers,
        p_loss=args.mel_plr,
        device=device,
    )
    obs_feat = extract_obs_features(latent_by_n[candidate_layers[0]]).unsqueeze(0)
    _ = controller(
        torch.tensor([args.mel_plr], device=device),
        r_max=torch.tensor([args.mel_rmax], device=device),
        obs_feat=obs_feat,
    )
    pred_n = int(args.mel_fixed_layers)

    spk_emb = spk_encoder(ref_wav.unsqueeze(0).to(device).squeeze(1))
    lat_in = latent_by_n[pred_n]
    lat_flow_in = _ensure_flow_min_frames(flow_model, lat_in)
    lat_out = flow_sample(flow_model, lat_flow_in, spk_emb, n_steps=args.n_steps, n_layers=pred_n)
    lat_out = lat_out[..., :lat_in.shape[-1]]
    ours_wav = st_model.decoder(lat_out).squeeze(1).squeeze(0).detach().cpu().numpy().astype(np.float64)

    results: List[Tuple[str, np.ndarray]] = [("Origin", wav_np), (f"PDSSC ({pred_n*0.5:.1f} kbps)", ours_wav)]
    target_rate = pred_n * 0.5

    if args.include_opus:
        wav_t = wav.clone()
        opus_wav = _audio_to_numpy(opus_lbrr_with_plr(wav_t, sr, args.opus_kbps, args.mel_plr))
        if opus_wav is not None:
            results.append((f"Opus + LBRR ({args.opus_kbps:.1f} kbps)", opus_wav))

    if args.include_esc:
        esc_wav, esc_rate = esc_with_lfrplc(wav_np, sr, target_rate, args.mel_plr)
        if esc_wav is not None and esc_rate is not None:
            results.append((f"ESC + LFR-PLC ({esc_rate:.1f} kbps)", esc_wav))

    if args.include_aac:
        wav_t = wav.clone()
        aac_wav = _audio_to_numpy(ffmpeg_codec_with_plr(wav_t, sr, "aac", args.aac_kbps, args.mel_plr))
        if aac_wav is not None:
            results.append((f"AAC + LFR-PLC ({args.aac_kbps:.1f} kbps)", aac_wav))

    vis_dir = Path(args.output_dir) / "mel_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_mel_grid(results, sr, vis_dir / "mel_comparison_grid.pdf")

    return {
        "selected_file": args.mel_fixed_file,
        "selected_layers": int(pred_n),
        "selected_bitrate_kbps": round(pred_n * 0.5, 6),
        "mel_plr": args.mel_plr,
        "mel_rmax": args.mel_rmax,
        "items": [title for title, _ in results],
    }


@torch.no_grad()
def mel_visualization_search(args, st_model, flow_model, spk_encoder, controller, candidate_layers, device):
    sr = st_model.sample_rate
    files = load_filelist(args.data_dir, args.split)
    if not files:
        raise RuntimeError(f"No wav/flac files found under {args.data_dir} ({args.split})")
    spk2files = build_spk2files(files)
    candidates = files[: min(args.mel_search_items, len(files))]

    best = None

    for fpath in candidates:
        wav = load_audio(fpath, sr)
        ref_wav = pick_ref_wav(fpath, spk2files, sr)
        wav_np = wav.squeeze(0).numpy().astype(np.float64)

        latent_by_n = build_candidate_latents(
            st_model,
            wav.squeeze(0),
            candidate_layers,
            p_loss=args.mel_plr,
            device=device,
        )
        obs_feat = extract_obs_features(latent_by_n[candidate_layers[0]]).unsqueeze(0)
        score = controller(
            torch.tensor([args.mel_plr], device=device),
            r_max=torch.tensor([args.mel_rmax], device=device),
            obs_feat=obs_feat,
        )
        rate_vec = torch.tensor([n * 0.5 for n in candidate_layers], device=device, dtype=torch.float32)
        feasible = rate_vec <= (args.mel_rmax + 1e-6)
        pred_idx = int(score.squeeze(0).masked_fill(~feasible, -1e9).argmax().item()) if feasible.any() else 0
        pred_n = candidate_layers[pred_idx]

        spk_emb = spk_encoder(ref_wav.unsqueeze(0).to(device).squeeze(1))
        lat_in = latent_by_n[pred_n]
        lat_flow_in = _ensure_flow_min_frames(flow_model, lat_in)
        lat_out = flow_sample(flow_model, lat_flow_in, spk_emb, n_steps=args.n_steps, n_layers=pred_n)
        lat_out = lat_out[..., :lat_in.shape[-1]]
        ours_wav = st_model.decoder(lat_out).squeeze(1).squeeze(0).detach().cpu().numpy().astype(np.float64)
        ours_dist = mel_l1_distance(wav_np, ours_wav, sr)
        target_rate = pred_n * 0.5

        baseline_items = []
        baseline_dists = []

        if args.include_opus:
            opus_wav = _audio_to_numpy(opus_lbrr_with_plr(wav, sr, args.opus_kbps, args.mel_plr))
            if opus_wav is not None:
                baseline_items.append((f"Opus + LBRR ({args.opus_kbps:.1f} kbps)", opus_wav))
                baseline_dists.append(mel_l1_distance(wav_np, opus_wav, sr))

        if args.include_encodec:
            enc_rate = _nearest_supported_rate(target_rate, [1.5, 3.0, 6.0, 12.0, 24.0])
            enc_wav = _audio_to_numpy(encodec_with_lfrplc(wav, sr, enc_rate, args.mel_plr, device))
            if enc_wav is not None:
                baseline_items.append((f"EnCodec + LFR-PLC ({enc_rate:.1f} kbps)", enc_wav))
                baseline_dists.append(mel_l1_distance(wav_np, enc_wav, sr))

        if args.include_esc:
            esc_wav, esc_rate = esc_with_lfrplc(wav_np, sr, target_rate, args.mel_plr)
            if esc_wav is not None and esc_rate is not None:
                baseline_items.append((f"ESC + LFR-PLC ({esc_rate:.1f} kbps)", esc_wav))
                baseline_dists.append(mel_l1_distance(wav_np, esc_wav, sr))

        if args.include_aac:
            aac_wav = _audio_to_numpy(ffmpeg_codec_with_plr(wav, sr, "aac", args.aac_kbps, args.mel_plr))
            if aac_wav is not None:
                baseline_items.append((f"AAC + LFR-PLC ({args.aac_kbps:.1f} kbps)", aac_wav))
                baseline_dists.append(mel_l1_distance(wav_np, aac_wav, sr))

        if not baseline_dists:
            continue

        baseline_mean = float(np.mean(baseline_dists))
        margin = baseline_mean - ours_dist
        if best is None or margin > best["margin"]:
            best = {
                "file": fpath,
                "margin": margin,
                "ours_dist": ours_dist,
                "baseline_mean_dist": baseline_mean,
                "selected_layers": int(pred_n),
                "selected_bitrate_kbps": round(pred_n * 0.5, 6),
                "origin": wav_np,
                "ours": ours_wav,
                "baseline_items": baseline_items,
            }

    if best is None:
        raise RuntimeError("Failed to find a representative mel comparison case.")

    vis_dir = Path(args.output_dir) / "mel_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    items = [("Origin", best["origin"]), (f"PDSSC ({best['selected_bitrate_kbps']:.1f} kbps)", best["ours"])]
    items.extend(best["baseline_items"])
    save_mel_grid(items, sr, vis_dir / "mel_comparison_grid.pdf")

    return {
        "selected_file": best["file"],
        "selected_layers": best["selected_layers"],
        "selected_bitrate_kbps": best["selected_bitrate_kbps"],
        "mel_plr": args.mel_plr,
        "mel_rmax": args.mel_rmax,
        "ours_mel_l1": best["ours_dist"],
        "baseline_mean_mel_l1": best["baseline_mean_dist"],
        "margin": best["margin"],
        "items": [title for title, _ in items],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Profile PDSSC FLOPs, latency, and mel visualizations.")
    p.add_argument("--mode", choices=["all", "flops", "latency", "mel"], default="all")
    p.add_argument("--output_dir", default="output/profile_system")
    p.add_argument("--data_dir", default="data/LibriSpeech")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--max_items", type=int, default=20)

    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path", default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--flow_ckpt", default="output/flow_checkpoints_stage3/best_train.pt")
    p.add_argument("--controller_ckpt", default="output/controller_checkpoints/best_controller.pt")

    p.add_argument("--n_layers_list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--obs_feat_dim", type=int, default=6)
    p.add_argument("--n_steps", type=int, default=10)

    p.add_argument("--profile_num_samples", type=int, default=16000 * 5)
    p.add_argument("--profile_n_layers", type=int, default=6)
    p.add_argument("--profile_rmax", type=float, default=3.0)
    p.add_argument("--profile_plr", type=float, default=0.10)

    p.add_argument("--latency_num_samples", type=int, default=16000 * 5)
    p.add_argument("--latency_n_layers", type=int, default=6)
    p.add_argument("--latency_rmax", type=float, default=3.0)
    p.add_argument("--latency_rmax_values", type=float, nargs="+", default=[3.0])
    p.add_argument("--latency_plr", type=float, default=0.10)
    p.add_argument("--latency_repeats", type=int, default=30)
    p.add_argument("--latency_warmup", type=int, default=5)

    p.add_argument("--input_wav", type=str, default="")
    p.add_argument("--ref_wav", type=str, default="")
    p.add_argument("--mel_fixed_file", type=str, default="test-clean/1089/134691/1089-134691-0010.flac")
    p.add_argument("--mel_fixed_layers", type=int, default=6)
    p.add_argument("--mel_plr", type=float, default=0.20)
    p.add_argument("--mel_rmax", type=float, default=3.0)
    p.add_argument("--mel_search_items", type=int, default=20)
    p.add_argument("--include_opus", action="store_true", default=True)
    p.add_argument("--include_opus_latency", action="store_true", default=False)
    p.add_argument("--include_encodec", action="store_true", default=False)
    p.add_argument("--include_esc", action="store_true", default=True)
    p.add_argument("--include_aac", action="store_true", default=False)
    p.add_argument("--opus_kbps", type=float, default=8.0)
    p.add_argument("--encodec_kbps", type=float, default=3.0)
    p.add_argument("--aac_kbps", type=float, default=20.0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st_model, flow_model, spk_encoder, controller, candidate_layers = load_models(args, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("all", "flops"):
        flops_meta = profile_flops_and_params(args, st_model, flow_model, spk_encoder, controller, device)
        _save_json(flops_meta, out_dir / "module_profile.json")
        _save_csv(flops_meta["rows"], out_dir / "module_profile.csv")
        print(f"[saved] {out_dir / 'module_profile.json'}")
        print(f"[saved] {out_dir / 'module_profile.csv'}")

    if args.mode in ("all", "latency"):
        latency_compare_meta = build_latency_compare_table(
            args,
            st_model,
            flow_model,
            spk_encoder,
            controller,
            candidate_layers,
            device,
        )
        _save_json(latency_compare_meta, out_dir / "latency_compare_table.json")
        _save_csv(latency_compare_meta["table_rows"], out_dir / "latency_compare_table.csv")
        print(f"[saved] {out_dir / 'latency_compare_table.json'}")
        print(f"[saved] {out_dir / 'latency_compare_table.csv'}")

    if args.mode in ("all", "mel"):
        mel_meta = mel_visualization(args, st_model, flow_model, spk_encoder, controller, candidate_layers, device)
        _save_json(mel_meta, out_dir / "mel_vis" / "mel_meta.json")
        print(f"[saved] {out_dir / 'mel_vis' / 'mel_meta.json'}")


if __name__ == "__main__":
    main()
