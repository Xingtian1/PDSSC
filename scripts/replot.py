# -*- coding: utf-8 -*-
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATE_COMPETITORS = {
    "AAC": {"color": "#17becf", "marker": "s"},
    "Opus": {"color": "#2ca02c", "marker": "^"},
    "AMR-NB": {"color": "#7f7f7f", "marker": "v"},
    "EnCodec": {"color": "#9467bd", "marker": "D"},
}

PLR_SYSTEMS = {
    "ours_1.5k": {"label": "Ours (1.5kbps, Flow-PLC)"},
    "ours_3.0k": {"label": "Ours (3.0kbps, Flow-PLC)"},
    "encodec_1.5k": {"label": "EnCodec (1.5kbps, LFR-PLC)"},
    "encodec_3.0k": {"label": "EnCodec (3.0kbps, LFR-PLC)"},
    "opus_8k": {"label": "Opus (~8kbps, LBRR)"},
    "amrnb_12k": {"label": "AMR-NB (12.2kbps, EC)"},
}

_PLR_STYLE = {
    "ours_1.5k": {"color": "#d62728", "marker": "o", "ls": "--", "lw": 1.3},
    "ours_3.0k": {"color": "#d62728", "marker": "o", "ls": "-", "lw": 1.6},
    "encodec_1.5k": {"color": "#9467bd", "marker": "D", "ls": "--", "lw": 1.1},
    "encodec_3.0k": {"color": "#9467bd", "marker": "D", "ls": "-", "lw": 1.1},
    "opus_8k": {"color": "#2ca02c", "marker": "^", "ls": "-", "lw": 1.1},
    "amrnb_12k": {"color": "#7f7f7f", "marker": "v", "ls": "-", "lw": 1.1},
}

_ABLATION_COLORS = {3: "#d62728", 4: "#1f77b4", 5: "#ff7f0e", 6: "#2ca02c"}

METRIC_LABELS = {
    "visqol": "VISQoL MOS-LQO",
    "utmos": "UTMOS",
    "plcmos": "PLCMOS",
}

RATE_METRICS = ["visqol", "utmos"]
PLR_METRICS = ["visqol", "plcmos"]
ABLATION_METRICS = ["visqol", "plcmos", "utmos"]


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


def plot_rate(data, out_dir):
    ours_rows = data["ours"]
    comp_results = data.get("competitors", {})
    n_samples = data.get("n_samples", "?")

    fig, axes_flat = _make_grid_fig(len(RATE_METRICS))
    for mi, metric in enumerate(RATE_METRICS):
        ax = axes_flat[mi]
        for name, cfg in RATE_COMPETITORS.items():
            if name not in comp_results or not comp_results[name]:
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


def plot_plr(data, out_dir):
    final = data["systems"]
    plr_list = data["plr_list"]
    n_samples = data.get("n_samples", "?")
    plr_pct = [p * 100 for p in plr_list]

    fig, axes_flat = _make_grid_fig(len(PLR_METRICS))
    for mi, metric in enumerate(PLR_METRICS):
        ax = axes_flat[mi]
        for key, sys_cfg in PLR_SYSTEMS.items():
            if key not in final:
                continue
            style = _PLR_STYLE[key]
            vals = final[key].get(metric, [])
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


def plot_ablation(data, out_dir):
    result = {int(k): v for k, v in data["result"].items()}
    plr_list = data["plr_list"]
    n_samples = data.get("n_samples", "?")
    plr_pct = [p * 100 for p in plr_list]

    fig, axes_flat = _make_grid_fig(len(ABLATION_METRICS))
    for mi, metric in enumerate(ABLATION_METRICS):
        ax = axes_flat[mi]
        for N in sorted(result.keys()):
            bitrate = N * 0.5
            color = _ABLATION_COLORS.get(N, "#7f7f7f")
            ys_w = result[N]["with"].get(metric, [])
            ys_wo = result[N]["without"].get(metric, [])
            _plot_line(ax, plr_pct, ys_w, color, "o",
                       f"{bitrate:.1f}kbps w/ Flow", lw=2.2, ls="-", zorder=5)
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


def main():
    p = argparse.ArgumentParser(description="replot: regenerate figures from JSON/NPZ")
    p.add_argument("--data_dir", required=True,
                   help="Output directory generated by eval_combined_v3.py (contains JSON/NPZ).")
    p.add_argument("--out_dir", default=None,
                   help="PNG output directory (default: data_dir).")
    p.add_argument("--part", nargs="+",
                   default=["rate", "plr", "ablation", "disentangle"],
                   choices=["rate", "plr", "ablation", "disentangle"],
                   help="Parts to replot (default: all).")
    args = p.parse_args()

    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    def _load_json(name):
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            print(f"[skip] missing: {path}")
            return None
        with open(path) as f:
            return json.load(f)

    if "rate" in args.part:
        data = _load_json("rate_data.json")
        if data:
            plot_rate(data, out_dir)
    if "plr" in args.part:
        data = _load_json("plr_data.json")
        if data:
            plot_plr(data, out_dir)
    if "ablation" in args.part:
        data = _load_json("ablation_data.json")
        if data:
            plot_ablation(data, out_dir)


if __name__ == "__main__":
    main()
