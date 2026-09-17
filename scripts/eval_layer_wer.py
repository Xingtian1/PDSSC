# -*- coding: utf-8 -*-
"""
Evaluate WER for different transmitted layer counts N.

This script measures content preservation by:
1. reconstructing waveform from the first N layers directly (without Flow), and
2. reconstructing waveform after TimbreFlow completion (with Flow),
then transcribing the reconstructed waveform with Whisper and computing WER
against the reference transcript.

Typical usage:
  python scripts/eval_layer_wer.py ^
      --flow_ckpt output/flow_checkpoints_stage2/best.pt ^
      --num_samples 50 ^
      --n_layers_list 1 2 3 4 5 6 7 8

Notes:
  - Requires openai-whisper, jiwer, and soundfile for WER evaluation.
  - Uses LibriSpeech transcripts resolved from the wav/flac path.
"""

import os
import json
import time
import random
import argparse
from typing import Dict, List

import numpy as np
import torch

from speechtokenizer import SpeechTokenizer
from speechtokenizer.flow import FlowMatchingModel, PretrainedSpeakerEncoder
from scripts.eval_utils import (
    set_seed,
    load_audio,
    load_filelist,
    build_spk2files,
    pick_ref_wav,
    channel_simulate,
    flow_sample,
    load_transcript,
    calc_wer,
)


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

    del ckpt
    return st_model, flow_model, spk_encoder


def decode_without_flow(st_model, wav: torch.Tensor, n_layers: int, device: str) -> np.ndarray:
    """
    Directly decode from the first n_layers RVQ codes without TimbreFlow.
    """
    x = wav.unsqueeze(0).to(device)  # (1, 1, T)
    with torch.no_grad():
        codes = st_model.encode(x)[:n_layers]
        wav_dec = st_model.decode(codes).squeeze(0).squeeze(0).cpu().numpy()
    return wav_dec.astype(np.float64)


def decode_with_flow(
    fpath: str,
    wav: torch.Tensor,
    n_layers: int,
    st_model,
    flow_model,
    spk_encoder,
    spk2f: Dict[str, List[str]],
    device: str,
    sr: int,
    n_steps: int,
) -> np.ndarray:
    """
    Simulate partial-layer reception with no packet loss, then complete with TimbreFlow.
    """
    sim = channel_simulate(st_model, wav, n_layers, p_loss=0.0, device=device)
    with torch.no_grad():
        ref_wav = pick_ref_wav(fpath, spk2f, sr).to(device)
        spk_emb = spk_encoder(ref_wav)
        lat_flow = flow_sample(flow_model, sim["latent_ch"], spk_emb, n_steps=n_steps, n_layers=n_layers)
        wav_flow = st_model.decoder(lat_flow).squeeze(0).squeeze(0).cpu().numpy()
    return wav_flow.astype(np.float64)


def safe_crop_pair(ref_np: np.ndarray, deg_np: np.ndarray):
    n = min(len(ref_np), len(deg_np))
    return ref_np[:n].astype(np.float64), deg_np[:n].astype(np.float64)


def summarize(vals: List[float]) -> Dict[str, float]:
    arr = np.array([v for v in vals if not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "count": int(arr.size),
    }


def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    st_model, flow_model, spk_encoder = load_models(args, device)
    sr = st_model.sample_rate

    files = load_filelist(args.data_dir, args.split)
    if not files:
        raise FileNotFoundError(f"No wav/flac files found for split={args.split} under {args.data_dir}")
    spk2f = build_spk2files(files)

    random.shuffle(files)
    selected = files[: min(args.num_samples, len(files))]

    print("=" * 80)
    print(f"[Layer-WER] device={device}  sr={sr}  samples={len(selected)}  layers={args.n_layers_list}")
    print(f"[Layer-WER] split={args.split}  whisper_model={args.whisper_model}  n_steps={args.n_steps}")
    print("=" * 80)

    per_layer = {
        int(N): {
            "without_flow": [],
            "with_flow": [],
            "details": [],
        }
        for N in args.n_layers_list
    }

    t0 = time.time()
    for idx, fpath in enumerate(selected, 1):
        wav = load_audio(fpath, sr)  # (1, T)
        wav = wav[:, : int(args.max_sec * sr)]
        ref_np = wav.squeeze(0).numpy().astype(np.float64)
        ref_text = load_transcript(fpath)

        print(f"[{idx:03d}/{len(selected):03d}] {os.path.basename(fpath)}")

        for N in args.n_layers_list:
            row = {"file": fpath, "n_layers": int(N)}

            if args.eval_without_flow:
                deg_np = decode_without_flow(st_model, wav, N, device)
                ref_clip, deg_clip = safe_crop_pair(ref_np, deg_np)
                wer_wo = calc_wer(deg_clip, ref_text, sr, model_size=args.whisper_model)
                per_layer[int(N)]["without_flow"].append(wer_wo)
                row["wer_without_flow"] = wer_wo
            else:
                row["wer_without_flow"] = None

            if args.eval_with_flow:
                deg_np = decode_with_flow(
                    fpath, wav, N, st_model, flow_model, spk_encoder, spk2f, device, sr, args.n_steps
                )
                ref_clip, deg_clip = safe_crop_pair(ref_np, deg_np)
                wer_wf = calc_wer(deg_clip, ref_text, sr, model_size=args.whisper_model)
                per_layer[int(N)]["with_flow"].append(wer_wf)
                row["wer_with_flow"] = wer_wf
            else:
                row["wer_with_flow"] = None

            per_layer[int(N)]["details"].append(row)

            msg_wo = "NA" if row["wer_without_flow"] is None or np.isnan(row["wer_without_flow"]) else f"{row['wer_without_flow']:.4f}"
            msg_wf = "NA" if row["wer_with_flow"] is None or np.isnan(row["wer_with_flow"]) else f"{row['wer_with_flow']:.4f}"
            print(f"  N={N}: WER(no-flow)={msg_wo}  WER(flow)={msg_wf}")

    summary = {
        "meta": {
            "split": args.split,
            "num_samples": len(selected),
            "n_layers_list": args.n_layers_list,
            "whisper_model": args.whisper_model,
            "n_steps": args.n_steps,
            "device": device,
            "elapsed_sec": time.time() - t0,
        },
        "layers": {},
    }

    print("\n" + "=" * 80)
    print("Layer-wise WER summary")
    print("=" * 80)
    print(f"{'N':>3}  {'WER(no-flow)':>18}  {'WER(flow)':>18}  {'count':>8}")
    print("-" * 80)

    for N in args.n_layers_list:
        s_wo = summarize(per_layer[int(N)]["without_flow"])
        s_wf = summarize(per_layer[int(N)]["with_flow"])
        summary["layers"][str(N)] = {
            "without_flow": s_wo,
            "with_flow": s_wf,
            "details": per_layer[int(N)]["details"],
        }
        msg_wo = "nan" if np.isnan(s_wo["mean"]) else f"{s_wo['mean']:.4f} ± {s_wo['std']:.4f}"
        msg_wf = "nan" if np.isnan(s_wf["mean"]) else f"{s_wf['mean']:.4f} ± {s_wf['std']:.4f}"
        count = max(s_wo["count"], s_wf["count"])
        print(f"{N:>3}  {msg_wo:>18}  {msg_wf:>18}  {count:>8}")

    out_json = os.path.join(args.output_dir, "layer_wer_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSaved:", out_json)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path", default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--flow_ckpt", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--max_sec", type=float, default=10.0)
    p.add_argument("--n_layers_list", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--whisper_model", default="base", help="Whisper model size for WER, e.g. base / small / medium")
    p.add_argument("--eval_without_flow", action="store_true", default=True)
    p.add_argument("--eval_with_flow", action="store_true", default=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="output/layer_wer")
    args = p.parse_args()
    main(args)
