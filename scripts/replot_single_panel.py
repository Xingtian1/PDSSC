# -*- coding: utf-8 -*-
"""Single-panel IEEE Transactions figures for fig4 / fig6 / fig8.

Each original (a)/(b) double-panel figure is split into two independent
single-column figures (one metric per figure). Visual style is preserved
from replot_final.py (colors, markers, line styles, grid, four spines,
low-alpha legend frame, serif font); only the canvas is rescaled to the
IEEE Transactions single-column width (~3.5 in) with proportionally
smaller fonts / markers, and the (a)/(b) panel labels + suptitle are
dropped because each panel is now a standalone figure.

Data sources (same as replot_final.py):
  part1_rate/rate_data.json          -> fig4 (ViSQOL, UTMOS)
  part2_plr/plr_data.json            -> fig6 (ViSQOL, PLCMOS)
  part3_ablation/ablation_data.json  -> fig8 (ViSQOL, PLCMOS)
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase

import replot_final as R  # reuse style constants / helpers (same scripts dir)

# ---- IEEE Transactions single-column sizing (~3.5 in) ----
# Narrower canvas so downscaling in the 2x3 figure* grid is minimal;
# fonts bumped so text stays legible after LaTeX shrinks the panel.
FIG_W = 3.2
FIG_H_RATE = 2.95
FIG_H_PLR = 2.95
LABEL_FS = 9.5
TICK_FS = 7.0
LEGEND_FS = 8.0
INSET_TICK_FS = 5.5
SPINE_LW = 1.0
MARKER_EDGE_W = 1.0
GRID_MAJOR = "#BFBFBF"
GRID_MINOR = "#E3E3E3"

# Rate series style, matching the visqol-only block in replot_final.plot_rate.
RATE_STYLE = {
    "AAC":     {"color": "#9A6A5A", "marker": "s", "lw": 1.6, "ms": 5.0, "ls": "-", "z": 2},
    "Opus":    {"color": "#148A1A", "marker": "o", "lw": 1.6, "ms": 5.0, "ls": "-", "z": 3},
    "EnCodec": {"color": "#1F39FF", "marker": "^", "lw": 1.7, "ms": 5.5, "ls": "-", "z": 4},
    "ESC":     {"color": "#FF68B3", "marker": "v", "lw": 1.8, "ms": 5.5, "ls": "-", "z": 5},
    "ours":    {"color": "#FF1F1F", "marker": "D", "lw": 2.2, "ms": 5.5, "ls": "-", "z": 6},
}

RATE_LEGEND_ORDER = ["ours", "Opus", "ESC", "AAC", "EnCodec"]

# Single-column marker sizes for the PLR series (original used ms=11).
_PLR_SINGLE_MS = {
    "ours_1.5k": 5.0, "ours_3.0k": 5.5,
    "encodec_1.5k": 4.8, "encodec_3.0k": 4.8,
    "esc_1.5k": 4.8, "esc_3.0k": 4.8,
    "opus_8k": 4.8,
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
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(labelsize=TICK_FS, length=4.0, width=1.1)
    plt.setp(ax.get_xticklabels(), fontweight="normal")
    plt.setp(ax.get_yticklabels(), fontweight="normal")
    ax.minorticks_on()
    ax.grid(which="major", color=GRID_MAJOR, linewidth=0.5, alpha=0.85)
    ax.grid(which="minor", color=GRID_MINOR, linewidth=0.3, alpha=0.95)
    ax.set_axisbelow(True)


class _MergedRow:
    """Legend-handle descriptor for a merged rate pair: a dashed member
    (1.5 kbps) and a solid member (3.0 kbps), drawn by _MergedHandler."""

    def __init__(self, dash, solid):
        self.dash = dash
        self.solid = solid


class _MergedHandler(HandlerBase):
    """Draws [dashed handle] / [solid handle] inside one handlebox.

    Each member is a full-size handle (1.6 units, same as the fig4 rows --
    pass handlelength=3.4 so the box covers 1.6 + slash gap + 1.6), with the
    line spanning the full member width and the hollow marker centered, and
    a '/' glyph between the two members.

    NB: matplotlib 3.7's HandlerTuple is unusable here -- it lays tuple
    members out right-to-left (xds_cycle = xdescent - (width+pad)*i), so the
    second member is drawn outside the handlebox and the handles come out
    reversed. Drawing the row manually gives full control over the layout.
    """

    def __init__(self):
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        dash, solid = orig_handle.dash, orig_handle.solid
        yc = (height - ydescent) / 2
        # members each ~1.5 units of the 3.4-unit box; '/' centered between
        # with a wider gap on both sides (0.44 / 0.12 / 0.44)
        wd = width * 0.44
        x_slash = width * 0.50
        x_solid0 = width * 0.56

        def _member(x0, x1, ms, mstyle, marker, dashes):
            # full-width line + centered hollow marker (two artists so the
            # dash pattern stays visible under the marker)
            line = Line2D([x0, x1], [yc, yc], color=mstyle["color"],
                          lw=mstyle["lw"], markerfacecolor="none",
                          markeredgecolor=mstyle["color"],
                          markeredgewidth=MARKER_EDGE_W)
            if dashes:
                line.set_dashes(dashes)
            mk = Line2D([(x0 + x1) / 2], [yc], marker=marker,
                        markersize=ms, markerfacecolor="none",
                        markeredgecolor=mstyle["color"],
                        markeredgewidth=MARKER_EDGE_W)
            return line, mk

        artists = []
        ln, mk = _member(0.0, wd, dash["ms"], dash, dash["marker"],
                         dash["dashes"])
        artists += [ln, mk]
        artists.append(Text(x_slash, yc, "/", ha="center", va="center",
                            fontsize=fontsize, fontweight="bold",
                            color="0.35"))
        ln, mk = _member(x_solid0, width, solid["ms"], solid,
                         solid["marker"], None)
        artists += [ln, mk]
        for a in artists:
            a.set_transform(trans)
        return artists


def _soft_legend(ax, handles, labels, loc, markerfirst=False, ncol=1,
                 bbox_to_anchor=None, handlelength=1.6, handler_map=None):
    kw = dict(
        fontsize=LEGEND_FS, frameon=True, fancybox=False,
        framealpha=0.12, edgecolor="none", facecolor="white",
        handlelength=handlelength, columnspacing=0.8, borderpad=0.2,
        labelspacing=0.25, borderaxespad=0.0,
        markerfirst=markerfirst, ncol=ncol,
    )
    if bbox_to_anchor is not None:
        kw["bbox_to_anchor"] = bbox_to_anchor
    if handler_map is not None:
        kw["handler_map"] = handler_map
    leg = ax.legend(handles, labels, loc=loc, **kw)
    for txt in leg.get_texts():
        txt.set_fontweight("normal")
    return leg


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _rate_legend(ax, handle_map, loc="lower left"):
    handles = [handle_map.get(k) for k in RATE_LEGEND_ORDER]
    handles = [h for h in handles if h is not None]
    labels = [h.get_label() for h in handles]
    return _soft_legend(ax, handles, labels, loc=loc, markerfirst=True)


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
                   3.0, "PDSSC", 6)
        inset.set_xlim(0.9, 4.65)
        inset.set_ylim(3.80, 4.45)
        inset.set_xticks([1, 2, 3, 4])
        inset.set_yticks([3.8, 4.0, 4.2, 4.3])
        inset.grid(False)
        inset.tick_params(labelsize=INSET_TICK_FS, length=2.2, width=0.7)
        for side in ("top", "right", "bottom", "left"):
            inset.spines[side].set_color("#8F8F8F")
            inset.spines[side].set_linewidth(0.5)
    _rate_legend(ax, handle_map, loc="lower right")

    fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.17)
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
        # Lower bound from 2.7 (user request); data max is 4.284.
        ax.set_ylim(2.7, 4.35)
        ax.set_yticks([2.7, 3.0, 3.3, 3.6, 3.9, 4.2])
    else:
        # Lower bound from 2.2 (user request); data max is 4.023.
        ax.set_ylim(2.2, 4.1)
        ax.set_yticks([2.2, 2.6, 3.0, 3.4, 3.8])
    ax.set_xlabel("Packet loss rate (%)", fontsize=LABEL_FS, fontweight="normal")
    ax.set_ylabel(R.METRIC_LABELS[metric], fontsize=LABEL_FS, fontweight="normal")
    _apply_axis_style(ax)

    # Merge the two rate variants of each system into one legend row: the
    # dashed handle (1.5 kbps) and the solid handle (3.0 kbps) are drawn side
    # by side inside one handlebox, with a '/' between them (matching the
    # "(1.5/3.0 kbps)" label). Keeps the legend at 4 rows so the y-axis stays
    # tight and the curves keep their visible slope. Row width stays
    # handlelength=1.6 (same as fig4's rows); marker shapes and sizes match
    # the plot lines. Opus keeps the default single-handle style.
    #
    # The row is drawn manually by _MergedHandler -- matplotlib 3.7's
    # HandlerTuple lays tuple members out right-to-left (xds_cycle =
    # xdescent - (width+pad)*i), so the second member escapes the handlebox
    # and the two handles come out reversed.
    merged = [
        (["ours_1.5k", "ours_3.0k"], "PDSSC (avg 1.5/3.0 kbps)"),
        (["encodec_1.5k", "encodec_3.0k"], "EnCodec + LFR-PLC (1.5/3.0 kbps)"),
        (["esc_1.5k", "esc_3.0k"], "ESC + LFR-PLC (1.5/3.0 kbps)"),
        (["opus_8k"], "Opus + LBRR (8.0 kbps)"),
    ]
    handles, labels = [], []
    for keys, label in merged:
        if len(keys) == 1:
            # single-rate system: keep the default full-width handle
            if keys[0] in handle_map:
                handles.append(handle_map[keys[0]])
                labels.append(label)
            continue
        st = R._PLR_STYLE[keys[0]]
        ln = handle_map[keys[0]]
        handles.append(_MergedRow(
            dict(color=st["color"], lw=ln.get_linewidth(),
                 marker=st["marker"], ms=ln.get_markersize(),
                 dashes=(1.9, 0.9)),
            dict(color=st["color"], lw=ln.get_linewidth(),
                 marker=st["marker"], ms=ln.get_markersize()),
        ))
        labels.append(label)
    _soft_legend(ax, handles, labels, loc="lower left", markerfirst=True,
                 handlelength=3.4, handler_map={_MergedRow: _MergedHandler()})

    fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.17)
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
                         st["color"], st["marker"], "-", 1.9, 5.0, label_with, 4)
        _plot_line(ax, plr_pct, result[N]["without"].get(metric, []),
                   st["color"], st["marker"], "--", 1.5, 5.0, label_without, 3)
        if line is not None:
            handle_map[(N, "with")] = line
        handle_map[(N, "without")] = Line2D(
            [], [], color=st["color"], marker=st["marker"], linestyle="--",
            linewidth=1.5, markersize=5.0, markerfacecolor="none",
            markeredgecolor=st["color"], markeredgewidth=MARKER_EDGE_W,
            label=label_without,
        )

    ax.set_xlim(-0.5, 30.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    if metric == "visqol":
        # Y axis starts at 3.5 (lowest curve value is 3.739), leaving room
        # below so the legend never crowds the bottom curves.
        ax.set_ylim(3.5, 4.32)
        ax.set_yticks([3.5, 3.7, 3.9, 4.1, 4.3])
        loc = "lower left"
    else:
        ax.set_ylim(2.6, 4.05)
        ax.set_yticks([2.6, 3.0, 3.4, 3.8, 4.0])
        loc = "lower left"
    ax.set_xlabel("Packet loss rate (%)", fontsize=LABEL_FS, fontweight="normal")
    ax.set_ylabel(R.METRIC_LABELS[metric], fontsize=LABEL_FS, fontweight="normal")
    _apply_axis_style(ax)

    order = [(6, "with"), (6, "without"), (3, "with"), (3, "without")]
    handles = [handle_map.get(k) for k in order]
    handles = [h for h in handles if h is not None]
    labels = [h.get_label() for h in handles]
    _soft_legend(ax, handles, labels, loc=loc, markerfirst=True)

    fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.17)
    _save(fig, out_dir, fname)


def main():
    p = argparse.ArgumentParser(
        description="Split fig4/fig6/fig8 into single-panel IEEE Transactions figures.")
    p.add_argument(
        "--eval_dir", default=os.path.join("root", "SpeechTokenizer-main", "output", "eval_final_full"),
        help="Directory containing part1_rate/ part2_plr/ part3_ablation/ (from eval_final_controller.py).")
    p.add_argument(
        "--out_dir", default=os.path.join("IEEE-Transactions-LaTeX2e-templates-and-instructions", "paper_figs"),
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
            plot_rate_single(data, args.out_dir, "visqol", "fig4_visqol")
            plot_rate_single(data, args.out_dir, "utmos", "fig4_utmos")
    if "fig6" in args.figs:
        data = _load_json(os.path.join("part2_plr", "plr_data.json"))
        if data:
            plot_plr_single(data, args.out_dir, "visqol", "fig6_visqol")
            plot_plr_single(data, args.out_dir, "plcmos", "fig6_plcmos")
    if "fig8" in args.figs:
        data = _load_json(os.path.join("part3_ablation", "ablation_data.json"))
        if data:
            plot_ablation_single(data, args.out_dir, "visqol", "fig8_visqol")
            plot_ablation_single(data, args.out_dir, "plcmos", "fig8_plcmos")


if __name__ == "__main__":
    main()
