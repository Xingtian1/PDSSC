# -*- coding: utf-8 -*-
"""Single-panel figures with legend placed OUTSIDE the axes, at the bottom.

Based on replot_single_panel.py.  Everything is identical except that the
legend for every panel is anchored below the axes (full width, centred)
instead of inside the plot area.
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import replot_final as R  # reuse style constants / helpers (same scripts dir)

FIG_W = 3.5
FIG_H_RATE = 3.0
FIG_H_PLR = 3.0
LABEL_FS = 9.5
TICK_FS = 8.0
LEGEND_FS = 7.5
INSET_TICK_FS = 6.5
SPINE_LW = 0.9
MARKER_EDGE_W = 0.9
GRID_MAJOR = "#BFBFBF"
GRID_MINOR = "#E3E3E3"

RATE_STYLE = {
    "AAC":     {"color": "#9A6A5A", "marker": "s", "lw": 1.3, "ms": 4.2, "ls": "-", "z": 2},
    "Opus":    {"color": "#148A1A", "marker": "o", "lw": 1.3, "ms": 4.2, "ls": "-", "z": 3},
    "EnCodec": {"color": "#1F39FF", "marker": "^", "lw": 1.4, "ms": 4.4, "ls": "-", "z": 4},
    "ESC":     {"color": "#FF68B3", "marker": "v", "lw": 1.5, "ms": 4.4, "ls": "-", "z": 5},
    "ours":    {"color": "#FF1F1F", "marker": "D", "lw": 1.8, "ms": 4.8, "ls": "-", "z": 6},
}

RATE_LEGEND_ORDER = ["ours", "Opus", "ESC", "AAC", "EnCodec"]

_PLR_SINGLE_MS = {
    "ours_1.5k": 4.6, "ours_3.0k": 4.8,
    "encodec_1.5k": 4.4, "encodec_3.0k": 4.4,
    "esc_1.5k": 4.4, "esc_3.0k": 4.4,
    "opus_8k": 4.4,
}
PLR_LEGEND_ORDER = [
    "ours_3.0k", "ours_1.5k",
    "encodec_3.0k", "encodec_1.5k",
    "esc_3.0k", "esc_1.5k",
    "opus_8k",
]


def _plot_line(ax, xs, ys, color, marker, ls, lw, ms, label, z):
    pairs = [(x, y) for x, y in zip(xs, ys) if R._is_finite_scalar(y)]
    if not pairs:
        return None
    px, py = zip(*pairs)
    line, = ax.plot(
        px, py, color=color, marker=marker, linestyle=ls, linewidth=lw,
        markersize=ms, markerfacecolor="none", markeredgecolor=color,
        markeredgewidth=MARKER_EDGE_W, label=label, zorder=z,
    )
    return line


def _apply_axis_style(ax):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(SPINE_LW)
    ax.tick_params(labelsize=TICK_FS, length=3.2, width=0.9)
    plt.setp(ax.get_xticklabels(), fontweight="normal")
    plt.setp(ax.get_yticklabels(), fontweight="normal")
    ax.minorticks_on()
    ax.grid(which="major", color=GRID_MAJOR, linewidth=0.5, alpha=0.85)
    ax.grid(which="minor", color=GRID_MINOR, linewidth=0.3, alpha=0.95)
    ax.set_axisbelow(True)


def _legend_below(ax, handles, labels, ncol, y_offset=-0.28):
    """Place legend *outside* the axes, full width, centred at the bottom."""
    kw = dict(
        fontsize=LEGEND_FS, frameon=True, fancybox=False,
        framealpha=0.12, edgecolor="none", facecolor="white",
        handlelength=1.6, columnspacing=0.8, borderpad=0.2,
        labelspacing=0.25, borderaxespad=0.0,
        ncol=ncol, mode="expand",
        bbox_to_anchor=(0.0, y_offset, 1.0, 0.102),
        loc="upper left",
    )
    leg = ax.legend(handles, labels, **kw)
    for txt in leg.get_texts():
        txt.set_fontweight("normal")
    return leg


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_rate_single(data, out_dir, metric, fname):
    ours_rows = data["ours"]
    comp = data.get("competitors", {})
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_RATE))

    handle_map = {}
    for name in ["AAC", "Opus", "EnCodec", "ESC"]:
        rows = comp.get(name, [])
        if not R._has_metric(rows, metric):
            continue
        st = RATE_STYLE[name]
        xs = [r["bitrate_kbps"] for r in rows]
        ys = [r.get(metric, np.nan) for r in rows]
        line = _plot_line(ax, xs, ys, st["color"], st["marker"], st["ls"],
                         st["lw"], st["ms"], R.RATE_LEGEND_LABELS.get(name, name), st["z"])
        if line is not None:
            handle_map[name] = line

    st = RATE_STYLE["ours"]
    xs = [r["bitrate_kbps"] for r in ours_rows]
    ys = [r.get(metric, np.nan) for r in ours_rows]
    line = _plot_line(ax, xs, ys, st["color"], st["marker"], st["ls"],
                     st["lw"], st["ms"], "PDSSC", st["z"])
    if line is not None:
        handle_map["ours"] = line

    ax.set_xlim(0.8, 20.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xticklabels(["1", "5", "10", "15", "20"])
    if metric == "visqol":
        ax.set_ylim(1.8, 4.46)
        ax.set_yticks([2.0, 2.5, 3.0, 3.5, 4.0])
    else:
        ax.set_ylim(1.2, 4.05)
        ax.set_yticks([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ax.set_xlabel("Rate (kbps)", fontsize=LABEL_FS, fontweight="normal")
    ax.set_ylabel(R.METRIC_LABELS[metric], fontsize=LABEL_FS, fontweight="normal")
    _apply_axis_style(ax)

    if metric == "visqol":
        esc_rows = comp.get("ESC", [])
        inset = ax.inset_axes([0.40, 0.66, 0.33, 0.30])
        inset.patch.set_facecolor((1.0, 1.0, 1.0, 0.55))
        if R._has_metric(esc_rows, "visqol"):
            _plot_line(inset, [r["bitrate_kbps"] for r in esc_rows],
                       [r.get("visqol", np.nan) for r in esc_rows],
                       RATE_STYLE["ESC"]["color"], RATE_STYLE["ESC"]["marker"],
                       "-", RATE_STYLE["ESC"]["lw"], 3.6, "ESC", 5)
        _plot_line(inset, xs, ys, RATE_STYLE["ours"]["color"],
                   RATE_STYLE["ours"]["marker"], "-", RATE_STYLE["ours"]["lw"],
                   3.6, "PDSSC", 6)
        inset.set_xlim(0.9, 4.65)
        inset.set_ylim(3.80, 4.45)
        inset.set_xticks([1, 2, 3, 4])
        inset.set_yticks([3.8, 4.0, 4.2, 4.3])
        inset.grid(False)
        inset.tick_params(labelsize=INSET_TICK_FS, length=2.2, width=0.7)
        for side in ("top", "right", "bottom", "left"):
            inset.spines[side].set_color("#8F8F8F")
            inset.spines[side].set_linewidth(0.5)

    handles = [handle_map.get(k) for k in RATE_LEGEND_ORDER]
    handles = [h for h in handles if h is not None]
    labels = [h.get_label() for h in handles]
    _legend_below(ax, handles, labels, ncol=5)

    fig.subplots_adjust(left=0.135, right=0.975, top=0.95, bottom=0.26)
    _save(fig, out_dir, fname)


def plot_plr_single(data, out_dir, metric, fname):
    final = data["systems"]
    plr_pct = [p * 100 for p in data["plr_list"]]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_PLR))

    handle_map = {}
    for key in PLR_LEGEND_ORDER:
        if key not in final:
            continue
        st = R._PLR_STYLE[key]
        line = _plot_line(ax, plr_pct, final[key].get(metric, []),
                         st["color"], st["marker"], st["ls"],
                         st["lw"] * 0.7, _PLR_SINGLE_MS[key],
                         R.PLR_SYSTEMS[key]["label"], 3)
        if line is not None:
            handle_map[key] = line

    ax.set_xlim(-0.5, 30.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    if metric == "visqol":
        ax.set_ylim(3.0, 4.35)
        ax.set_yticks([3.0, 3.3, 3.6, 3.9, 4.2])
    else:
        ax.set_ylim(2.4, 4.1)
        ax.set_yticks([2.4, 2.8, 3.2, 3.6, 4.0])
    ax.set_xlabel("Packet loss rate (%)", fontsize=LABEL_FS, fontweight="normal")
    ax.set_ylabel(R.METRIC_LABELS[metric], fontsize=LABEL_FS, fontweight="normal")
    _apply_axis_style(ax)

    handles = [handle_map.get(k) for k in PLR_LEGEND_ORDER]
    handles = [h for h in handles if h is not None]
    labels = [h.get_label() for h in handles]
    _legend_below(ax, handles, labels, ncol=4, y_offset=-0.30)

    fig.subplots_adjust(left=0.135, right=0.975, top=0.95, bottom=0.27)
    _save(fig, out_dir, fname)


def plot_ablation_single(data, out_dir, metric, fname):
    result = {int(k): v for k, v in data["result"].items()}
    plr_pct = [p * 100 for p in data.get("plr_list", [data.get("fixed_plr", 0.05)])]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_PLR))

    handle_map = {}
    for N in [3, 6]:
        if N not in result:
            continue
        bitrate = N * 0.5
        st = R._ABLATION_STYLE.get(N, {"color": "#7f7f7f", "marker": "o"})
        label_with = f"PDSSC (avg {bitrate:.1f} kbps)"
        label_without = f"w/o flow (avg {bitrate:.1f} kbps)"
        line = _plot_line(ax, plr_pct, result[N]["with"].get(metric, []),
                         st["color"], st["marker"], "-", 1.6, 4.6, label_with, 4)
        _plot_line(ax, plr_pct, result[N]["without"].get(metric, []),
                  st["color"], st["marker"], "--", 1.3, 4.6, label_without, 3)
        if line is not None:
            handle_map[(N, "with")] = line
        handle_map[(N, "without")] = Line2D(
            [], [], color=st["color"], marker=st["marker"], linestyle="--",
            linewidth=1.3, markersize=4.6, markerfacecolor="none",
            markeredgecolor=st["color"], markeredgewidth=MARKER_EDGE_W,
            label=label_without,
        )

    ax.set_xlim(-0.5, 30.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    if metric == "visqol":
        ax.set_ylim(3.7, 4.32)
        ax.set_yticks([3.7, 3.9, 4.1, 4.3])
    else:
        ax.set_ylim(2.6, 4.05)
        ax.set_yticks([2.6, 3.0, 3.4, 3.8, 4.0])
    ax.set_xlabel("Packet loss rate (%)", fontsize=LABEL_FS, fontweight="normal")
    ax.set_ylabel(R.METRIC_LABELS[metric], fontsize=LABEL_FS, fontweight="normal")
    _apply_axis_style(ax)

    order = [(6, "with"), (6, "without"), (3, "with"), (3, "without")]
    handles = [handle_map.get(k) for k in order]
    handles = [h for h in handles if h is not None]
    labels = [h.get_label() for h in handles]
    _legend_below(ax, handles, labels, ncol=2, y_offset=-0.30)

    fig.subplots_adjust(left=0.135, right=0.975, top=0.95, bottom=0.27)
    _save(fig, out_dir, fname)


def main():
    p = argparse.ArgumentParser(
        description="Single-panel figures with legend outside axes (bottom).")
    p.add_argument(
        "--eval_dir",
        default=os.path.join("root", "SpeechTokenizer-main", "output", "eval_final_full"),
        help="Directory containing part1_rate/ part2_plr/ part3_ablation/.")
    p.add_argument(
        "--out_dir",
        default=os.path.join("IEEE-Transactions-LaTeX2e-templates-and-instructions", "paper_figs"),
        help="Where to write the single-panel PDFs.")
    p.add_argument("--figs", nargs="+", default=["fig4", "fig6", "fig8"],
                  choices=["fig4", "fig6", "fig8"])
    args = p.parse_args()

    def _load_json(rel):
        path = os.path.join(args.eval_dir, rel)
        if not os.path.exists(path):
            print(f"[skip] missing: {path}")
            return None
        with open(path) as f:
            return json.load(f)

    if "fig4" in args.figs:
        data = _load_json(os.path.join("part1_rate", "rate_data.json"))
        if data:
            plot_rate_single(data, args.out_dir, "visqol", "fig4_visqol_lo")
            plot_rate_single(data, args.out_dir, "utmos", "fig4_utmos_lo")
    if "fig6" in args.figs:
        data = _load_json(os.path.join("part2_plr", "plr_data.json"))
        if data:
            plot_plr_single(data, args.out_dir, "visqol", "fig6_visqol_lo")
            plot_plr_single(data, args.out_dir, "plcmos", "fig6_plcmos_lo")
    if "fig8" in args.figs:
        data = _load_json(os.path.join("part3_ablation", "ablation_data.json"))
        if data:
            plot_ablation_single(data, args.out_dir, "visqol", "fig8_visqol_lo")
            plot_ablation_single(data, args.out_dir, "plcmos", "fig8_plcmos_lo")


if __name__ == "__main__":
    main()
