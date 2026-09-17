# -*- coding: utf-8 -*-
"""
eval_combined_v3.py  ─  综合评估脚本 v3

Part 1 (rate)    : 码率-质量曲线 (PLR=0%)  指标: VISQoL / UTMOS / PESQ / STOI / WER
Part 2 (plr)     : PLR-质量曲线            指标: VISQoL / PLCMOS / UTMOS / PESQ / STOI / WER
Part 3 (ablation): 消融实验 (有/无 Flow)   指标: VISQoL / PLCMOS / UTMOS / PESQ / STOI / WER
Part 4 (vis)     : 可视化 (3×3 频谱 + 编码器输入特征)

PLR 竞品:
  - Opus ~8kbps (inband FEC / LBRR)
  - AMR-NB 12.2kbps (内置 PLC)
  - AAC 24kbps (最高档 + 线性插值 PLC)
  - EnCodec 1.5k / 3.0k (LFR-PLC)

用法:
  python scripts/eval_combined_v3.py \\
      --flow_ckpt output/flow_checkpoints_stage2/best.pt \\
      --num_samples 200 --part all
"""

import os
import json
import argparse
import random
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speechtokenizer import SpeechTokenizer
from speechtokenizer.flow import FlowMatchingModel, PretrainedSpeakerEncoder
from scripts.eval_utils import (
    set_seed, load_audio, load_filelist, build_spk2files, pick_ref_wav,
    channel_simulate, flow_sample,
    calc_visqol, calc_visqol_batch, calc_utmos, calc_plcmos, calc_mcd,
    nanmean,
    HAS_ENCODEC,
    encodec_with_plr, encodec_with_lfrplc,
    ffmpeg_codec_with_interp, ffmpeg_codec_with_plr, ffmpeg_codec_with_builtin_plc, opus_lbrr_with_plr,
)


# =============================================================================
# 竞品 / 系统配置
# =============================================================================

RATE_COMPETITORS = {
    "AAC": {
        "type"    : "ffmpeg",
        "codec"   : "aac",
        "bitrates": [4, 5, 6, 7, 8, 10, 12, 16, 20, 24],
        "color"   : "#17becf", "marker": "s",
    },
    "Opus": {
        "type"    : "ffmpeg",
        "codec"   : "libopus",
        "bitrates": [6.0, 8.0, 12.0, 16.0],
        "color"   : "#2ca02c", "marker": "^",
    },
    "AMR-NB": {
        "type"    : "ffmpeg",
        "codec"   : "libopencore_amrnb",
        "bitrates": [4.75, 5.9, 7.95, 12.2],
        "color"   : "#7f7f7f", "marker": "v",
    },
    "EnCodec": {
        "type"    : "encodec",
        "bitrates": [1.5, 3.0, 6.0, 12.0],
        "color"   : "#9467bd", "marker": "D",
    },
}

# PLR 测试系统
PLR_SYSTEMS = {
    "ours_1.5k"   : {"label": "Ours (1.5kbps, Flow-PLC)",                  "type": "ours",    "n_layers": 3},
    "ours_3.0k"   : {"label": "Ours (3.0kbps, Flow-PLC)",                  "type": "ours",    "n_layers": 6},
    "encodec_1.5k": {"label": "EnCodec (1.5kbps, LFR-PLC)",                "type": "encodec", "bw": 1.5},
    "encodec_3.0k": {"label": "EnCodec (3.0kbps, LFR-PLC)",                "type": "encodec", "bw": 3.0},
    "opus_8k"     : {"label": "Opus (~8kbps, LBRR)",                        "type": "opus",    "bw": 8.0},
    "amrnb_12k"   : {"label": "AMR-NB (12.2kbps, EC)",                     "type": "amrnb",   "bw": 12.2},
    "aac_20k"     : {"label": "AAC (20kbps, Linear-Interp PLC)",           "type": "aac",     "bw": 20.0},
}

PLR_LIST = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

RATE_METRICS     = ["visqol", "utmos", "mcd"]
PLR_METRICS      = ["visqol", "plcmos", "utmos", "mcd"]
ABLATION_METRICS = ["visqol", "plcmos", "utmos", "mcd"]

METRIC_LABELS = {
    "visqol": "VISQoL MOS-LQO",
    "utmos" : "UTMOS",
    "plcmos": "PLCMOS",
    "mcd"   : "MCD (dB) ↓",
}

_PLR_STYLE = {
    "ours_1.5k"   : {"color": "#d62728", "marker": "o", "ls": "--", "lw": 1.3},
    "ours_3.0k"   : {"color": "#d62728", "marker": "o", "ls": "-",  "lw": 1.6},
    "encodec_1.5k": {"color": "#9467bd", "marker": "D", "ls": "--", "lw": 1.1},
    "encodec_3.0k": {"color": "#9467bd", "marker": "D", "ls": "-",  "lw": 1.1},
    "opus_8k"     : {"color": "#2ca02c", "marker": "^", "ls": "-",  "lw": 1.1},
    "amrnb_12k"   : {"color": "#7f7f7f", "marker": "v", "ls": "-",  "lw": 1.1},
    "aac_20k"     : {"color": "#17becf", "marker": "s", "ls": "-",  "lw": 1.1},
}


# =============================================================================
# 模型加载
# =============================================================================

def _strip_compile_prefix(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def load_models(args, device):
    st_model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
    st_model.eval().to(device)
    for p in st_model.parameters():
        p.requires_grad_(False)

    ckpt       = torch.load(args.flow_ckpt, map_location="cpu")
    saved_args = argparse.Namespace(**ckpt.get("args", {}))
    flow_model = FlowMatchingModel(
        latent_dim = st_model.quantizer.dimension,
        base_ch    = getattr(saved_args, "base_ch",   512),
        ch_mults   = tuple(getattr(saved_args, "ch_mults", [1, 1, 2])),
        cond_dim   = getattr(saved_args, "cond_dim",  512),
        spk_dim    = getattr(saved_args, "spk_dim",   256),
        time_dim   = getattr(saved_args, "time_dim",  128),
        n_res      = getattr(saved_args, "n_res",     2),
        n_mid_res  = getattr(saved_args, "n_mid_res", 2),
    ).to(device)
    flow_model.load_state_dict(_strip_compile_prefix(ckpt["flow_model"]))
    flow_model.eval()

    spk_encoder = PretrainedSpeakerEncoder(
        emb_dim  = getattr(saved_args, "spk_dim", 256),
        save_dir = os.path.join(os.path.dirname(args.flow_ckpt), "spkrec-ecapa"),
    ).to(device)
    spk_encoder.load_state_dict(_strip_compile_prefix(ckpt["spk_encoder"]), strict=False)
    spk_encoder.eval()
    del ckpt
    return st_model, flow_model, spk_encoder


# =============================================================================
# 通用绘图辅助
# =============================================================================

def _plot_line(ax, xs, ys, color, marker, label, lw=1.1, ls="-", zorder=3):
    pairs = [(x, y) for x, y in zip(xs, ys) if not np.isnan(float(y))]
    if not pairs:
        return
    px, py = zip(*pairs)
    ax.plot(px, py, color=color, marker=marker, linewidth=lw,
            linestyle=ls, markersize=3.5, markerfacecolor="none",
            markeredgewidth=0.9, label=label, zorder=zorder)


def _make_grid_fig(n_metrics, ncols=3, cell_w=4.5, cell_h=4.5):
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, cell_h * nrows))
    if nrows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    for mi in range(n_metrics, len(axes_flat)):
        axes_flat[mi].set_visible(False)
    return fig, axes_flat


# =============================================================================
# Part 1: 码率测试
# =============================================================================

def eval_rate_ours(args, st_model, flow_model, spk_encoder, selected, spk2f, device, out_dir):
    sr   = st_model.sample_rate
    rows = []
    for N in args.n_layers_list:
        bitrate = N * 0.5
        print(f"\n[Ours] N={N} ({bitrate:.1f}kbps) ...", flush=True)
        acc      = {m: [] for m in RATE_METRICS}
        ref_list, deg_list = [], []

        for i, fpath in enumerate(selected):
            print(f"  [{i+1}/{len(selected)}]", end="\r", flush=True)
            wav     = load_audio(fpath, sr)[:, :int(args.max_sec * sr)]
            ref_np  = wav.squeeze().numpy().astype(np.float64)
            ref_wav = pick_ref_wav(fpath, spk2f, sr).to(device)

            result = channel_simulate(st_model, wav, N, p_loss=0.0, device=device)
            with torch.no_grad():
                spk_emb  = spk_encoder(ref_wav)
                lat_flow = flow_sample(flow_model, result["latent_ch"],
                                       spk_emb, n_steps=args.n_steps, n_layers=N)
                wav_flow = st_model.decoder(lat_flow).squeeze(0).cpu()
            deg = wav_flow.squeeze().numpy().astype(np.float64)

            ref_list.append(ref_np)
            deg_list.append(deg)
            acc["utmos"].append(calc_utmos(deg, sr))
            acc["mcd"].append(calc_mcd(deg, ref_np, sr))

        v_list = calc_visqol_batch(ref_list, deg_list, sr, args.visqol_workers) \
                 if not args.no_visqol else [float("nan")] * len(ref_list)
        acc["visqol"] = v_list

        row = {"bitrate_kbps": bitrate}
        row.update({m: nanmean(acc[m]) for m in RATE_METRICS})
        rows.append(row)
        print(f"\n  N={N}: " + "  ".join(f"{m.upper()}={row[m]:.3f}" for m in RATE_METRICS))
        _save_rate_json(rows, {}, out_dir)   # 每轮保存一次
    return rows


def _eval_rate_codec_fn(codec_fn, selected, spk2f, sr, max_sec,
                         no_visqol, visqol_workers):
    ref_list, deg_list = [], []
    acc = {m: [] for m in RATE_METRICS}
    for fpath in selected:
        wav    = load_audio(fpath, sr)[:, :int(max_sec * sr)]
        ref_np = wav.squeeze().numpy().astype(np.float64)
        deg = codec_fn(wav.squeeze(0), sr)
        if deg is None:
            continue
        ref_list.append(ref_np)
        deg_list.append(deg)
        acc["utmos"].append(calc_utmos(deg, sr))
        acc["mcd"].append(calc_mcd(deg, ref_np, sr))

    v_list = calc_visqol_batch(ref_list, deg_list, sr, visqol_workers) \
             if (not no_visqol and ref_list) else [float("nan")] * len(ref_list)
    acc["visqol"] = v_list
    return {m: nanmean(acc[m]) for m in RATE_METRICS}


def eval_rate_competitors(args, selected, spk2f, sr, device):
    results = {}
    for name, cfg in RATE_COMPETITORS.items():
        if cfg["type"] == "encodec" and not HAS_ENCODEC:
            print(f"  [skip] {name}: encodec not installed")
            continue
        rows = []
        for bw in cfg["bitrates"]:
            print(f"\n[{name}] {bw}kbps ...", flush=True)
            if cfg["type"] == "encodec":
                fn = lambda w, s, bw=bw: encodec_with_plr(w, s, bw, p_loss=0.0, device=device)
            else:
                fn = lambda w, s, bw=bw, cd=cfg["codec"]: \
                         ffmpeg_codec_with_plr(w, s, cd, bw, p_loss=0.0)
            m = _eval_rate_codec_fn(fn, selected, spk2f, sr, args.max_sec,
                                    args.no_visqol, args.visqol_workers)
            m["bitrate_kbps"] = bw
            rows.append(m)
            print("  " + "  ".join(
                f"{k.upper()}={v:.3f}" for k, v in m.items() if k != "bitrate_kbps"
            ))
        results[name] = rows
    return results


def _save_rate_json(ours_rows, comp_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = {
        "ours":        ours_rows,
        "competitors": comp_results,
    }
    path = os.path.join(out_dir, "rate_data.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def plot_rate(ours_rows, comp_results, out_dir, n_samples):
    fig, axes_flat = _make_grid_fig(len(RATE_METRICS))
    for mi, metric in enumerate(RATE_METRICS):
        ax = axes_flat[mi]
        for name, cfg in RATE_COMPETITORS.items():
            if name not in comp_results:
                continue
            rows = comp_results[name]
            _plot_line(ax, [r["bitrate_kbps"] for r in rows],
                       [r.get(metric, float("nan")) for r in rows],
                       cfg["color"], cfg["marker"], name)
        _plot_line(ax, [r["bitrate_kbps"] for r in ours_rows],
                   [r.get(metric, float("nan")) for r in ours_rows],
                   "#d62728", "o", "TimbreFlow (Ours)", lw=1.6, zorder=6)
        ax.set_xlabel("Bitrate (kbps)", fontsize=10)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"({chr(97 + mi)})", fontsize=10, pad=4)

    fig.suptitle(f"Quality vs Bitrate  (PLR=0%, n={n_samples})", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_rate_v3.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")


# =============================================================================
# Part 2: PLR 测试
# =============================================================================

def _eval_plr_one_file(fpath, plr, st_model, flow_model, spk_encoder,
                        spk2f, device, args):
    sr     = st_model.sample_rate
    wav    = load_audio(fpath, sr)[:, :int(args.max_sec * sr)]
    wav_np = wav.squeeze().numpy().astype(np.float64)

    deg_map = {}

    # Ours
    for n_layers, key in [(3, "ours_1.5k"), (6, "ours_3.0k")]:
        try:
            sim = channel_simulate(st_model, wav, n_layers, p_loss=plr, device=device)
            with torch.no_grad():
                ref_wav  = pick_ref_wav(fpath, spk2f, sr).to(device)
                spk_emb  = spk_encoder(ref_wav)
                lat_flow = flow_sample(flow_model, sim["latent_ch"],
                                       spk_emb, n_steps=args.n_steps, n_layers=n_layers)
                wav_flow = st_model.decoder(lat_flow).squeeze(0).cpu()
            deg_map[key] = wav_flow.squeeze().numpy().astype(np.float64)
        except Exception as e:
            print(f"  [{key}] {e}")
            deg_map[key] = None

    # EnCodec LFR-PLC
    for bw, key in [(1.5, "encodec_1.5k"), (3.0, "encodec_3.0k")]:
        deg_map[key] = encodec_with_lfrplc(wav.squeeze(0), sr, bw, plr, device) \
                       if HAS_ENCODEC else None

    # Opus ~8kbps inband FEC
    try:
        deg_map["opus_8k"] = opus_lbrr_with_plr(wav.squeeze(0), sr, 8.0, plr)
    except Exception:
        deg_map["opus_8k"] = None

    # AMR-NB 12.2kbps + built-in-like PLC (frame-hold approximation)
    try:
        deg_map["amrnb_12k"] = ffmpeg_codec_with_builtin_plc(
            wav.squeeze(0), sr, "libopencore_amrnb", 12.2, plr)
    except Exception:
        deg_map["amrnb_12k"] = None

    # AAC 20kbps
    try:
        deg_map["aac_20k"] = ffmpeg_codec_with_interp(
            wav.squeeze(0), sr, "aac", 20.0, plr)
    except Exception:
        deg_map["aac_20k"] = None

    # 批量 VISQoL
    valid_keys = [k for k in PLR_SYSTEMS if deg_map.get(k) is not None]
    v_scores   = calc_visqol_batch(
        [wav_np] * len(valid_keys),
        [deg_map[k] for k in valid_keys],
        sr, n_workers=max(1, len(valid_keys))
    ) if (not args.no_visqol and valid_keys) else [float("nan")] * len(valid_keys)
    visqol_map = dict(zip(valid_keys, v_scores))

    results = {}
    for key in PLR_SYSTEMS:
        deg_np = deg_map.get(key)
        if deg_np is None:
            results[key] = {m: float("nan") for m in PLR_METRICS}
        else:
            results[key] = {
                "visqol" : visqol_map.get(key, float("nan")),
                "plcmos" : calc_plcmos(deg_np, wav_np, sr),
                "utmos"  : calc_utmos(deg_np, sr),
            }
    return results


def _save_plr_json(final, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = {
        "plr_list": PLR_LIST,
        "systems":  final,
    }
    path = os.path.join(out_dir, "plr_data.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def plot_plr(final, out_dir, n_samples):
    fig, axes_flat = _make_grid_fig(len(PLR_METRICS))
    plr_pct = [p * 100 for p in PLR_LIST]

    for mi, metric in enumerate(PLR_METRICS):
        ax = axes_flat[mi]
        for key, sys_cfg in PLR_SYSTEMS.items():
            style = _PLR_STYLE[key]
            vals  = final[key][metric]
            _plot_line(ax, plr_pct, vals,
                       style["color"], style["marker"], sys_cfg["label"],
                       lw=style["lw"], ls=style["ls"])
        ax.set_xlabel("Packet Loss Probability (%)", fontsize=10)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_xlim(-1, 32)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"({chr(97 + mi)})", fontsize=10, pad=4)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Quality vs Packet Loss Rate  (n={n_samples})", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    path = os.path.join(out_dir, "fig_plr_v3.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")


# =============================================================================
# Part 3: 消融实验 (有/无 Flow) — x 轴为 PLR，6 条线 (3 码率 × 2 条件)
# =============================================================================

_ABLATION_COLORS = {2: "#d62728", 4: "#1f77b4", 6: "#2ca02c"}


def eval_ablation(args, st_model, flow_model, spk_encoder,
                  selected, spk2f, device, out_dir):
    """
    对每个 N in ablation_layers，在 PLR_LIST 各 PLR 下评测有/无 Flow 的质量。
    返回 result[N]["with"|"without"][metric] = [value_per_plr]
    共 6 条线 × 7 个 PLR 点。
    """
    sr     = st_model.sample_rate
    result = {N: {"with":    {m: [] for m in ABLATION_METRICS},
                  "without": {m: [] for m in ABLATION_METRICS}}
              for N in args.ablation_layers}

    total   = len(PLR_LIST) * len(args.ablation_layers)
    done    = 0
    t_start = time.time()

    for pi, plr in enumerate(PLR_LIST):
        print(f"\n[Ablation] PLR={plr*100:.0f}%  ({pi+1}/{len(PLR_LIST)})", flush=True)
        for N in args.ablation_layers:
            bitrate = N * 0.5
            per_w  = {m: [] for m in ABLATION_METRICS}
            per_wo = {m: [] for m in ABLATION_METRICS}
            ref_w, deg_w   = [], []
            ref_wo, deg_wo = [], []

            for fpath in selected:
                wav     = load_audio(fpath, sr)[:, :int(args.max_sec * sr)]
                ref_np  = wav.squeeze().numpy().astype(np.float64)
                ref_wav = pick_ref_wav(fpath, spk2f, sr).to(device)

                sim = channel_simulate(st_model, wav, N, p_loss=plr, device=device)

                # With flow
                with torch.no_grad():
                    spk_emb  = spk_encoder(ref_wav)
                    lat_flow = flow_sample(flow_model, sim["latent_ch"],
                                           spk_emb, n_steps=args.n_steps, n_layers=N)
                    wav_with = st_model.decoder(lat_flow).squeeze(0).cpu()
                d_with = wav_with.squeeze().numpy().astype(np.float64)

                # Without flow: 直接解码 N 层 latent
                with torch.no_grad():
                    wav_without = st_model.decoder(sim["latent_ch"]).squeeze(0).cpu()
                d_without = wav_without.squeeze().numpy().astype(np.float64)

                ref_w.append(ref_np);  deg_w.append(d_with)
                ref_wo.append(ref_np); deg_wo.append(d_without)
                per_w["utmos"].append(calc_utmos(d_with, sr))
                per_w["plcmos"].append(calc_plcmos(d_with, ref_np, sr))
                per_wo["utmos"].append(calc_utmos(d_without, sr))
                per_wo["plcmos"].append(calc_plcmos(d_without, ref_np, sr))

            v_with   = calc_visqol_batch(ref_w,  deg_w,  sr, args.visqol_workers) \
                       if not args.no_visqol else [float("nan")] * len(ref_w)
            v_without = calc_visqol_batch(ref_wo, deg_wo, sr, args.visqol_workers) \
                        if not args.no_visqol else [float("nan")] * len(ref_wo)
            per_w["visqol"]  = v_with
            per_wo["visqol"] = v_without

            for m in ABLATION_METRICS:
                result[N]["with"][m].append(nanmean(per_w[m]))
                result[N]["without"][m].append(nanmean(per_wo[m]))

            done += 1
            eta = (time.time() - t_start) / done * (total - done)
            print(f"  N={N} ({bitrate:.1f}k)  "
                  f"VISQoL w/={result[N]['with']['visqol'][-1]:.3f}  "
                  f"wo/={result[N]['without']['visqol'][-1]:.3f}  "
                  f"ETA {eta/60:.0f}min", flush=True)

    _save_ablation_json(result, out_dir)
    return result


def _save_ablation_json(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # JSON key 必须是字符串
    data = {
        "plr_list": PLR_LIST,
        "result":   {str(N): v for N, v in result.items()},
    }
    path = os.path.join(out_dir, "ablation_data.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def plot_ablation(result, out_dir, n_samples):
    fig, axes_flat = _make_grid_fig(len(ABLATION_METRICS))
    plr_pct = [p * 100 for p in PLR_LIST]

    for mi, metric in enumerate(ABLATION_METRICS):
        ax = axes_flat[mi]
        for N in sorted(result.keys()):
            bitrate = N * 0.5
            color   = _ABLATION_COLORS.get(N, "#7f7f7f")
            ys_w    = result[N]["with"][metric]
            ys_wo   = result[N]["without"][metric]
            _plot_line(ax, plr_pct, ys_w,  color, "o",
                       f"{bitrate:.1f}kbps w/ Flow",  lw=2.2, ls="-",  zorder=5)
            _plot_line(ax, plr_pct, ys_wo, color, "s",
                       f"{bitrate:.1f}kbps w/o Flow", lw=1.8, ls="--", zorder=4)
        ax.set_xlabel("Packet Loss Probability (%)", fontsize=10)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_xlim(-1, 32)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=1)
        ax.set_title(f"({chr(97 + mi)})", fontsize=10, pad=4)

    fig.suptitle(f"Ablation: w/ vs w/o Flow Model  (n={n_samples})", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_ablation_v3.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")


# =============================================================================
# Part 4: 可视化
#   4-A: PLR 系统频谱对比 (8 条件 3×3 + Encoder pre-RVQ latent)
#   4-B: 特征分离 t-SNE (来自 viz_disentangle.py)
# =============================================================================

def _log_mel_spectrogram(wav_np: np.ndarray, sr: int,
                          n_fft: int = 1024, hop: int = 256, n_mels: int = 80):
    """返回 (n_mels, T) log-mel 矩阵 (dB)，用 torchaudio 实现。"""
    import torchaudio.transforms as T
    wav_t  = torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)
    mel_fn = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop,
                               n_mels=n_mels, power=2.0)
    spec   = mel_fn(wav_t).squeeze(0)
    return (10.0 * torch.log10(spec + 1e-9)).numpy()


def _show_spec(ax, wav_np, sr, title):
    try:
        spec = _log_mel_spectrogram(wav_np, sr)
        ax.imshow(spec, aspect="auto", origin="lower", interpolation="none",
                  cmap="magma", vmin=-80, vmax=0)
        ax.set_title(title, fontsize=8, pad=3)
        ax.axis("off")
    except Exception as e:
        ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_title(title, fontsize=8, pad=3)
        ax.axis("off")


# ── 4-A: PLR 系统频谱对比 ─────────────────────────────────────────────────────

def eval_vis(args, st_model, flow_model, spk_encoder, selected, spk2f, device, out_dir):
    """
    取第 1 条文件，生成 8 种条件（Original + 7 PLR 系统，PLR=0%）的频谱图。

    子图布局 (3 行 × 3 列):
      9 格，放 8 个频谱，第 9 格（row2 col2）留空
    """
    sr     = st_model.sample_rate
    fpath  = selected[0]
    wav    = load_audio(fpath, sr)[:, :int(args.max_sec * sr)]
    wav_np = wav.squeeze().numpy().astype(np.float64)
    print(f"\n[Vis-A] File: {os.path.basename(fpath)}", flush=True)

    ordered = []   # list of (title, wav_np)
    ordered.append(("Original", wav_np))

    # Ours 1.5kbps (N=3, w/ Flow)
    res = channel_simulate(st_model, wav, 3, p_loss=0.0, device=device)
    with torch.no_grad():
        ref_wav = pick_ref_wav(fpath, spk2f, sr).to(device)
        spk_emb = spk_encoder(ref_wav)
        lat     = flow_sample(flow_model, res["latent_ch"], spk_emb, n_steps=args.n_steps, n_layers=3)
        w_f     = st_model.decoder(lat).squeeze(0).cpu()
    ordered.append(("Ours 1.5kbps (Semantic Interp)", w_f.squeeze().numpy().astype(np.float64)))

    # Ours 3.0kbps (N=6, w/ Flow)
    res = channel_simulate(st_model, wav, 6, p_loss=0.0, device=device)
    with torch.no_grad():
        lat = flow_sample(flow_model, res["latent_ch"], spk_emb, n_steps=args.n_steps, n_layers=6)
        w_f = st_model.decoder(lat).squeeze(0).cpu()
    ordered.append(("Ours 3.0kbps (Semantic Interp)", w_f.squeeze().numpy().astype(np.float64)))

    # EnCodec 1.5kbps
    d = encodec_with_lfrplc(wav.squeeze(0), sr, 1.5, 0.0, device) if HAS_ENCODEC else None
    ordered.append(("EnCodec 1.5kbps (LFR-PLC)", d if d is not None else wav_np))

    # EnCodec 3.0kbps
    d = encodec_with_lfrplc(wav.squeeze(0), sr, 3.0, 0.0, device) if HAS_ENCODEC else None
    ordered.append(("EnCodec 3.0kbps (LFR-PLC)", d if d is not None else wav_np))

    # Opus ~8kbps LBRR
    d = opus_lbrr_with_plr(wav.squeeze(0), sr, 8.0, 0.0)
    ordered.append(("Opus ~8kbps (LBRR)", d if d is not None else wav_np))

    # AMR-NB 12.2kbps + built-in-like PLC (frame-hold approximation)
    d = ffmpeg_codec_with_builtin_plc(wav.squeeze(0), sr, "libopencore_amrnb", 12.2, 0.0)
    ordered.append(("AMR-NB 12.2kbps (EC)", d if d is not None else wav_np))

    # AAC 20kbps
    d = ffmpeg_codec_with_interp(wav.squeeze(0), sr, "aac", 20.0, 0.0)
    ordered.append(("AAC 20kbps (Linear-Interp PLC)", d if d is not None else wav_np))

    # ── 绘图：3×3，前 8 格频谱 + 第 9 格空白 ──────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes_flat = axes.flatten()   # 9 格

    # 格 0-7：8 个频谱（Original + 7 PLR 系统）
    for i, (title, w_np) in enumerate(ordered):
        _show_spec(axes_flat[i], w_np.astype(np.float32), sr, title)

    # 格 8：空白
    axes_flat[8].set_visible(False)

    fig.suptitle(f"PLR Systems Audio Comparison (PLR=0%) | {os.path.basename(fpath)}",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_vis_v3.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")


# ── 4-B: 特征分离 t-SNE（来自 viz_disentangle.py）────────────────────────────

@torch.no_grad()
def _extract_rvq_features(st_model, fpath, sr, max_sec, device):
    """提取各 RVQ 层的时间平均 latent 向量。"""
    wav   = load_audio(fpath, sr)[:, :int(max_sec * sr)].unsqueeze(0).to(device)
    codes = st_model.encode(wav)    # (8, 1, T_enc)
    D     = st_model.quantizer.dimension

    def decode_layer(idx):
        vq_l = st_model.quantizer.vq.layers[idx]
        d    = vq_l.decode(codes[idx])
        if d.shape[-1] == D:
            d = d.permute(0, 2, 1)
        return d.contiguous()   # (1, D, T_enc)

    to_vec = lambda x: x.squeeze(0).mean(dim=-1).cpu().numpy()   # (D,)
    layers = [decode_layer(i) for i in range(8)]
    result = {f"rvq{i+1}": to_vec(lyr) for i, lyr in enumerate(layers)}
    result["rvq28"] = to_vec(sum(layers[1:]))
    return result


def eval_disentangle(args, st_model, spk2f, device, out_dir):
    """
    t-SNE 可视化各 RVQ 层特征（语义 vs 音色分离）。
    输出:
      fig_disentangle_tsne.png  — 3×3 t-SNE 子图（RVQ1..8 + RVQ2-8 sum）
      fig_disentangle_sim.png   — 说话人内/间余弦相似度条形图
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[Vis-B] sklearn 未安装，跳过 t-SNE。pip install scikit-learn")
        return

    sr          = st_model.sample_rate
    spk_ids     = sorted(spk2f.keys())
    random.shuffle(spk_ids)
    sel_spks    = spk_ids[:args.disentangle_n_speakers]
    feat_keys   = [f"rvq{i}" for i in range(1, 9)] + ["rvq28"]
    accum       = {k: [] for k in feat_keys}
    spk_labels  = []

    print(f"\n[Vis-B] t-SNE: {args.disentangle_n_speakers} speakers × "
          f"{args.disentangle_n_per_speaker} utts ...", flush=True)
    for spk_idx, spk in enumerate(sel_spks):
        chosen = random.sample(spk2f[spk], min(args.disentangle_n_per_speaker, len(spk2f[spk])))
        for fpath in chosen:
            try:
                feats = _extract_rvq_features(st_model, fpath, sr, args.max_sec, device)
                for k in feat_keys:
                    accum[k].append(feats[k])
                spk_labels.append(spk_idx)
            except Exception as e:
                print(f"  [skip] {os.path.basename(fpath)}: {e}")

    if len(spk_labels) < 5:
        print("[Vis-B] 样本不足，跳过 t-SNE。")
        return

    feat_dict = {k: np.stack(accum[k]) for k in feat_keys}
    print(f"  Total samples: {len(spk_labels)}")

    # ── 保存特征数据供复现 ─────────────────────────────────────────────────────
    npz_path = os.path.join(out_dir, "disentangle_data.npz")
    np.savez(npz_path, spk_labels=np.array(spk_labels), **feat_dict)
    print(f"Saved: {npz_path}")

    # ── t-SNE 3×3 图 ──────────────────────────────────────────────────────────
    unique_spks = sorted(set(spk_labels))
    cmap        = plt.get_cmap("tab10")
    colors      = {s: cmap(i % 10) for i, s in enumerate(unique_spks)}
    panel_titles = {f"rvq{i}": f"RVQ Layer {i}" for i in range(1, 9)}
    panel_titles["rvq1"]  += "\n(semantic)"
    panel_titles["rvq28"]  = "RVQ 2-8 Sum\n(timbre)"

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    axes_flat  = axes.flatten()
    for pi, key in enumerate(feat_keys):
        ax   = axes_flat[pi]
        feats = feat_dict[key]
        print(f"  t-SNE: {key} ...", flush=True)
        emb = TSNE(n_components=2, perplexity=min(30, len(feats) - 1),
                   random_state=42, max_iter=1000).fit_transform(feats)
        for spk in unique_spks:
            idx = [i for i, s in enumerate(spk_labels) if s == spk]
            ax.scatter(emb[idx, 0], emb[idx, 1], color=colors[spk],
                       label=f"Spk {spk}", s=30, alpha=0.8, edgecolors="none")
        ax.set_title(panel_titles[key], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, alpha=0.2)

    handles, labels_leg = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center",
               ncol=min(len(unique_spks), 10),
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("RVQ Layer Feature t-SNE (colored by speaker identity)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_disentangle_tsne.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")


# =============================================================================
# 主流程
# =============================================================================

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  part={args.part}")
    st_model, flow_model, spk_encoder = load_models(args, device)
    sr = st_model.sample_rate

    files    = load_filelist(args.data_dir, args.split)
    spk2f    = build_spk2files(files)
    selected = random.sample(files, min(args.num_samples, len(files)))
    print(f"Samples: {len(selected)}  |  split={args.split}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Part 1: 码率测试 ─────────────────────────────────────────────────────
    if args.part in ("rate", "all"):
        print("\n" + "=" * 60)
        print("Part 1: Quality vs Bitrate (PLR=0%)")
        print("=" * 60)
        ours_rows    = eval_rate_ours(args, st_model, flow_model, spk_encoder,
                                      selected, spk2f, device, args.out_dir)
        comp_results = eval_rate_competitors(args, selected, spk2f, sr, device)
        _save_rate_json(ours_rows, comp_results, args.out_dir)
        plot_rate(ours_rows, comp_results, args.out_dir, len(selected))

    # ── Part 2: PLR 测试 ─────────────────────────────────────────────────────
    if args.part in ("plr", "all"):
        print("\n" + "=" * 60)
        print("Part 2: Quality vs PLR")
        print("=" * 60)
        final   = {key: {m: [] for m in PLR_METRICS} for key in PLR_SYSTEMS}
        total   = len(PLR_LIST) * len(selected)
        done    = 0
        t_start = time.time()
        for pi, plr in enumerate(PLR_LIST):
            print(f"\nPLR={plr*100:.0f}%  ({pi+1}/{len(PLR_LIST)})", flush=True)
            per_file = {key: {m: [] for m in PLR_METRICS} for key in PLR_SYSTEMS}
            for fi, fpath in enumerate(selected):
                t0  = time.time()
                res = _eval_plr_one_file(fpath, plr, st_model, flow_model,
                                          spk_encoder, spk2f, device, args)
                done  += 1
                eta    = (time.time() - t_start) / done * (total - done)
                print(f"  [{fi+1}/{len(selected)} PLR {pi+1}/{len(PLR_LIST)}] "
                      f"{time.time()-t0:.1f}s  ETA {eta/60:.0f}min  "
                      f"{os.path.basename(fpath)}", flush=True)
                for key in PLR_SYSTEMS:
                    for m in PLR_METRICS:
                        per_file[key][m].append(res.get(key, {}).get(m, float("nan")))
            print()
            for key in PLR_SYSTEMS:
                for m in PLR_METRICS:
                    final[key][m].append(nanmean(per_file[key][m]))
                v  = final[key]["visqol"][-1]
                ut = final[key]["utmos"][-1]
                pc = final[key]["plcmos"][-1]
                print(f"  {PLR_SYSTEMS[key]['label']:45s}  "
                      f"VISQoL={v:.3f}  UTMOS={ut:.3f}  PLCMOS={pc:.3f}")
            _save_plr_json(final, args.out_dir)
        plot_plr(final, args.out_dir, len(selected))

    # ── Part 3: 消融实验 ─────────────────────────────────────────────────────
    if args.part in ("ablation", "all"):
        print("\n" + "=" * 60)
        print("Part 3: Ablation Study (w/ vs w/o Flow, PLR sweep)")
        print("=" * 60)
        ablation_result = eval_ablation(
            args, st_model, flow_model, spk_encoder,
            selected, spk2f, device, args.out_dir)
        plot_ablation(ablation_result, args.out_dir, len(selected))

    # ── Part 4: 可视化 ───────────────────────────────────────────────────────
    if args.part in ("vis", "all"):
        print("\n" + "=" * 60)
        print("Part 4-A: PLR Systems Spectrogram Visualization")
        print("=" * 60)
        eval_vis(args, st_model, flow_model, spk_encoder,
                 selected, spk2f, device, args.out_dir)
        print("\n" + "=" * 60)
        print("Part 4-B: RVQ Feature Disentanglement (t-SNE)")
        print("=" * 60)
        eval_disentangle(args, st_model, spk2f, device, args.out_dir)

    print(f"\n结果已保存至: {args.out_dir}")


# =============================================================================
# 参数
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="eval_combined_v3: 综合评估 v3")
    p.add_argument("--config_path",      default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path",        default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--flow_ckpt",        default="output/flow_checkpoints_stage2/best.pt")
    p.add_argument("--data_dir",         default="/root/SpeechTokenizer-main/LibriSpeech")
    p.add_argument("--split",            default="test-clean")
    p.add_argument("--num_samples",      type=int,   default=200)
    p.add_argument("--num_samples_comp", type=int,   default=100)
    p.add_argument("--max_sec",          type=float, default=10.0)
    p.add_argument("--n_steps",          type=int,   default=10)
    p.add_argument("--n_layers_list",    type=int,   nargs="+", default=list(range(2, 9)))
    p.add_argument("--ablation_layers",  type=int,   nargs="+", default=[2, 4, 6],
                   help="消融实验码率点 N 值，默认 [2,4,6] 对应 1/2/3 kbps")
    p.add_argument("--out_dir",          default="output/eval_results_v3")
    p.add_argument("--no_visqol",        action="store_true")
    p.add_argument("--visqol_workers",   type=int,   default=8)
    p.add_argument("--disentangle_n_speakers",    type=int, default=20,
                   help="t-SNE 可视化使用的说话人数（默认 20）")
    p.add_argument("--disentangle_n_per_speaker", type=int, default=20,
                   help="每位说话人取的句子数（默认 20）")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--part",             default="all",
                   choices=["all", "rate", "plr", "ablation", "vis"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
