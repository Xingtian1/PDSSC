# -*- coding: utf-8 -*-
"""Double-column two-panel IEEE Transactions figures for fig4 / fig6 / fig8.

Each figure is ONE double-column figure (figure*) with two side-by-side panels
(one metric per panel) and a SHARED legend placed below the panels, outside the
axes area. Visual style is reused from replot_single_panel (which itself mirrors
replot_final.py): colors, markers, line styles, grid, four spines, serif font.

Data sources (same as replot_final.py / replot_single_panel.py):
  part1_rate/rate_data.json          -> fig4 (a: ViSQOL, b: UTMOS)
  part2_plr/plr_data.json            -> fig6 (a: ViSQOL, b: PLCMOS)
  part3_ablation/ablation_data.json  -> fig8 (a: ViSQOL, b: PLCMOS)
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import replot_final as R
import replot_single_panel as SP

# Override to Times New Roman for double-panel IEEE figures.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["pdf.fonttype"] = 42

# IEEE Transactions double-column (figure*) text width ~7.16 in.
FIG_W_DOUBLE = 7.5
FIG_H = 2.8
PANEL_LABEL_FS = 9.5
LEGEND_FS_DOUBLE = 9.0
TICK_FS_DOUBLE = 9.0


# Subplot margins are asymmetric (left=0.085, right=0.985); the axes centre is
# at 0.535, not 0.5. With bbox_inches="tight" the y-axis labels extend the
# content left edge, so a legend anchored at figure x=0.5 reads as left-aligned.
# Centre the legend on the plot area instead.
SUBPLOT_LEFT = 0.085
SUBPLOT_RIGHT = 0.985
LEGEND_CX = (SUBPLOT_LEFT + SUBPLOT_RIGHT) / 2  # 0.535


def _fig_legend(fig, handles, labels, ncol, markerfirst=False):
    kw = dict(
        fontsize=LEGEND_FS_DOUBLE, frameon=True, fancybox=False,
        framealpha=0.12, edgecolor="none", facecolor="white",
        handlelength=1.7, columnspacing=1.0, borderpad=0.3,
        labelspacing=0.30, borderaxespad=0.0,
        markerfirst=markerfirst, ncol=ncol,
    )
    leg = fig.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(LEGEND_CX, -0.06), **kw)
    for txt in leg.get_texts():
        txt.set_fontweight("normal")
    return leg


def _panel_label(ax, text):
    ax.text(0.5, -0.22, text, transform=ax.transAxes,
            fontsize=PANEL_LABEL_FS, fontweight="bold", va="center", ha="center")


def _finalize(fig, out_dir, fname, handles, labels, ncol, markerfirst):
    fig.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, top=0.95,
                        bottom=0.15, wspace=0.22)
    _fig_legend(fig, handles, labels, ncol, markerfirst=markerfirst)
    SP._save(fig, out_dir, fname)


# --------------------------------------------------------------------------- #
# fig4 -- rate (ViSQOL, UTMOS)
# --------------------------------------------------------------------------- #
def _plot_rate_panel(ax, data, metric):
    ours_rows = data["ours"]
    comp = data.get("competitors", {})
    handle_map = {}
    for name in ["AAC", "Opus", "EnCodec", "ESC"]:
        rows = comp.get(name, [])
        if not R._has_metric(rows, metric):
            continue
        st = SP.RATE_STYLE[name]
        xs = [r["bitrate_kbps"] for r in rows]
        ys = [r.get(metric, np.nan) for r in rows]
        line = SP._plot_line(ax, xs, ys, st["color"], st["marker"], st["ls"],
                             st["lw"], st["ms"],
                             R.RATE_LEGEND_LABELS.get(name, name), st["z"])
        if line is not None:
            handle_map[name] = line

    st = SP.RATE_STYLE["ours"]
    xs = [r["bitrate_kbps"] for r in ours_rows]
    ys = [r.get(metric, np.nan) for r in ours_rows]
    line = SP._plot_line(ax, xs, ys, st["color"], st["marker"], st["ls"],
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
    ax.set_xlabel("Rate (kbps)", fontsize=SP.LABEL_FS, fontweight="normal")
    ax.set_ylabel(R.METRIC_LABELS[metric], fontsize=SP.LABEL_FS,
                  fontweight="normal")
    SP._apply_axis_style(ax)
    ax.tick_params(labelsize=TICK_FS_DOUBLE)

    # zoom-in inset only on the ViSQOL panel (low-rate regime, ESC vs PDSSC)
    if metric == "visqol":
        esc_rows = comp.get("ESC", [])
        inset = ax.inset_axes([0.40, 0.66, 0.33, 0.30])
        inset.patch.set_facecolor((1.0, 1.0, 1.0, 0.55))
        if R._has_metric(esc_rows, "visqol"):
            SP._plot_line(inset, [r["bitrate_kbps"] for r in esc_rows],
                          [r.get("visqol", np.nan) for r in esc_rows],
                          SP.RATE_STYLE["ESC"]["color"],
                          SP.RATE_STYLE["ESC"]["marker"], "-",
                          SP.RATE_STYLE["ESC"]["lw"], 3.6, "ESC", 5)
        SP._plot_line(inset, xs, ys, SP.RATE_STYLE["ours"]["color"],
                      SP.RATE_STYLE["ours"]["marker"], "-",
                      SP.RATE_STYLE["ours"]["lw"], 3.6, "PDSSC", 6)
        inset.set_xlim(0.9, 4.65)
        inset.set_ylim(3.80, 4.45)
        inset.set_xticks([1, 2, 3, 4])
        inset.set_yticks([3.8, 4.0, 4.2, 4.3])
        inset.grid(False)
        inset.tick_params(labelsize=SP.INSET_TICK_FS, length=2.2, width=0.7)
        for side in ("top", "right", "bottom", "left"):
            inset.spines[side].set_color("#8F8F8F")
            inset.spines[side].set_linewidth(0.5)
    return handle_map


def plot_rate_double(data, out_dir, fname):
    metrics = ["visqol", "utmos"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    handle_map = {}
    for ax, metric, lab in zip(axes, metrics, ["(a)", "(b)"]):
        handle_map.update(_plot_rate_panel(ax, data, metric))
        _panel_label(ax, lab)
    handles = [handle_map.get(k) for k in SP.RATE_LEGEND_ORDER]
    handles = [h for h in handles if h is not None]
    labels = [h.get_label() for h in handles]
    _finalize(fig, out_dir, fname, handles, labels, len(handles),
              markerfirst=True)


# --------------------------------------------------------------------------- #
# fig6 -- packet-loss robustness (ViSQOL, PLCMOS)
# --------------------------------------------------------------------------- #
def _plot_plr_panel(ax, data, metric):
    final = data["systems"]
    plr_pct = [p * 100 for p in data["plr_list"]]
    handle_map = {}
    for key in SP.PLR_LEGEND_ORDER:
        if key not in final:
            continue
        st = R._PLR_STYLE[key]
        line = SP._plot_line(ax, plr_pct, final[key].get(metric, []),
                             st["color"], st["marker"], st["ls"],
                             st["lw"] * 0.7, SP._PLR_SINGLE_MS[key],
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
    ax.set_xlabel("Packet loss rate (%)", fontsize=SP.LABEL_FS,
                  fontweight="normal")
    ax.set_ylabel(R.METRIC_LABELS[metric], fontsize=SP.LABEL_FS,
                  fontweight="normal")
    SP._apply_axis_style(ax)
    ax.tick_params(labelsize=TICK_FS_DOUBLE)
    return handle_map


def plot_plr_double(data, out_dir, fname):
    metrics = ["visqol", "plcmos"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    handle_map = {}
    for ax, metric, lab in zip(axes, metrics, ["(a)", "(b)"]):
        handle_map.update(_plot_plr_panel(ax, data, metric))
        _panel_label(ax, lab)
    handles = [handle_map.get(k) for k in SP.PLR_LEGEND_ORDER]
    handles = [h for h in handles if h is not None]
    labels = [h.get_label() for h in handles]
    _finalize(fig, out_dir, fname, handles, labels, 4, markerfirst=True)


# --------------------------------------------------------------------------- #
# fig8 -- ablation of flow completion (ViSQOL, PLCMOS)
# --------------------------------------------------------------------------- #
def _plot_ablation_panel(ax, data, metric):
    result = {int(k): v for k, v in data["result"].items()}
    plr_pct = [p * 100 for p in
               data.get("plr_list", [data.get("fixed_plr", 0.05)])]
    handle_map = {}
    for N in [3, 6]:
        if N not in result:
            continue
        bitrate = N * 0.5
        st = R._ABLATION_STYLE.get(N, {"color": "#7f7f7f", "marker": "o"})
        label_with = f"PDSSC (avg {bitrate:.1f} kbps)"
        label_without = f"w/o flow (avg {bitrate:.1f} kbps)"
        line = SP._plot_line(ax, plr_pct, result[N]["with"].get(metric, []),
                             st["color"], st["marker"], "-", 1.6, 4.6,
                             label_with, 4)
        SP._plot_line(ax, plr_pct, result[N]["without"].get(metric, []),
                      st["color"], st["marker"], "--", 1.3, 4.6,
                      label_without, 3)
        if line is not None:
            handle_map[(N, "with")] = line
        handle_map[(N, "without")] = Line2D(
            [], [], color=st["color"], marker=st["marker"], linestyle="--",
            linewidth=1.3, markersize=4.6, markerfacecolor="none",
            markeredgecolor=st["color"], markeredgewidth=SP.MARKER_EDGE_W,
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
    ax.set_xlabel("Packet loss rate (%)", fontsize=SP.LABEL_FS,
                  fontweight="normal")
    ax.set_ylabel(R.METRIC_LABELS[metric], fontsize=SP.LABEL_FS,
                  fontweight="normal")
    SP._apply_axis_style(ax)
    ax.tick_params(labelsize=TICK_FS_DOUBLE)
    return handle_map


def plot_ablation_double(data, out_dir, fname):
    metrics = ["visqol", "plcmos"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    handle_map = {}
    for ax, metric, lab in zip(axes, metrics, ["(a)", "(b)"]):
        handle_map.update(_plot_ablation_panel(ax, data, metric))
        _panel_label(ax, lab)
    order = [(6, "with"), (6, "without"), (3, "with"), (3, "without")]
    handles = [handle_map.get(k) for k in order]
    handles = [h for h in handles if h is not None]
    labels = [h.get_label() for h in handles]
    _finalize(fig, out_dir, fname, handles, labels, 4, markerfirst=True)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Double-column two-panel IEEE Transactions figures for "
                    "fig4 / fig6 / fig8.")
    p.add_argument(
        "--eval_dir",
        default=os.path.join("root", "SpeechTokenizer-main", "output",
                             "eval_final_full"),
        help="Directory containing part1_rate/ part2_plr/ part3_ablation/.")
    p.add_argument(
        "--out_dir",
        default=os.path.join("IEEE-conference-template-062824", "writing",
                             "paper_figs"),
        help="Where to write the double-panel PDFs.")
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
            plot_rate_double(data, args.out_dir, "fig4")
    if "fig6" in args.figs:
        data = _load_json(os.path.join("part2_plr", "plr_data.json"))
        if data:
            plot_plr_double(data, args.out_dir, "fig6")
    if "fig8" in args.figs:
        data = _load_json(os.path.join("part3_ablation", "ablation_data.json"))
        if data:
            plot_ablation_double(data, args.out_dir, "fig8")


if __name__ == "__main__":
    main()
