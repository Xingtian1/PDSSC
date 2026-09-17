# -*- coding: utf-8 -*-
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Cambria", "Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"

PALETTE = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_base": "#E4CCD8",
    "ours_edge": "#B64342",
    "neutral_mid": "#7A7A7A",
    "neutral_dark": "#4D4D4D",
    "grid": "#D9D9D9",
}

RATE_COMPETITORS = {
    "AAC": {"color": PALETTE["baseline_soft"], "marker": "s"},
    "Opus": {"color": PALETTE["neutral_mid"], "marker": "^"},
    "EnCodec": {"color": PALETTE["baseline_mid"], "marker": "D"},
    "ESC": {"color": PALETTE["baseline_dark"], "marker": "o"},
}

RATE_LEGEND_LABELS = {
    "AAC": "AAC + LFR-PLC",
    "Opus": "Opus + LBRR",
    "EnCodec": "EnCodec + LFR-PLC",
    "ESC": "ESC + LFR-PLC",
    "ours": "PDSSC",
}

PLR_SYSTEMS = {
    "ours_1.5k": {"label": "PDSSC (avg 1.5 kbps)"},
    "ours_3.0k": {"label": "PDSSC (avg 3.0 kbps)"},
    "encodec_1.5k": {"label": "EnCodec + LFR-PLC (1.5 kbps)"},
    "encodec_3.0k": {"label": "EnCodec + LFR-PLC (3.0 kbps)"},
    "esc_1.5k": {"label": "ESC + LFR-PLC (1.5 kbps)"},
    "esc_3.0k": {"label": "ESC + LFR-PLC (3.0 kbps)"},
    "opus_8k": {"label": "Opus + LBRR (8.0 kbps)"},
}

_PLR_STYLE = {
    "ours_1.5k": {"color": "#FF1F1F", "marker": "D", "ls": "--", "lw": 2.4, "ms": 11.0},
    "ours_3.0k": {"color": "#FF1F1F", "marker": "D", "ls": "-", "lw": 2.8, "ms": 11.0},
    "encodec_1.5k": {"color": "#1F39FF", "marker": "^", "ls": "--", "lw": 2.0, "ms": 11.0},
    "encodec_3.0k": {"color": "#1F39FF", "marker": "^", "ls": "-", "lw": 2.2, "ms": 11.0},
    "esc_1.5k": {"color": "#FF68B3", "marker": "v", "ls": "--", "lw": 2.0, "ms": 11.0},
    "esc_3.0k": {"color": "#FF68B3", "marker": "v", "ls": "-", "lw": 2.3, "ms": 11.0},
    "opus_8k": {"color": "#148A1A", "marker": "o", "ls": "-", "lw": 2.1, "ms": 11.0},
}

_ABLATION_STYLE = {
    3: {"color": "#355CFF", "marker": "^"},
    6: {"color": "#FF1F1F", "marker": "D"},
}

METRIC_LABELS = {
    "visqol": "ViSQOL",
    "utmos": "UTMOS",
    "plcmos": "PLCMOS",
    "wer": "WER",
}

RATE_METRICS = ["visqol", "utmos", "wer"]
PLR_METRICS = ["visqol", "plcmos"]
ABLATION_METRICS = ["visqol", "plcmos", "utmos"]
ABLATION_PLOT_METRICS = ["visqol", "plcmos"]


def apply_publication_style(font_size=28, axes_linewidth=1.2):
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = axes_linewidth
    plt.rcParams["ytick.major.width"] = axes_linewidth
    plt.rcParams["xtick.minor.width"] = axes_linewidth * 0.8
    plt.rcParams["ytick.minor.width"] = axes_linewidth * 0.8


def save_pub(fig, filename_base, dpi=600):
    fig.savefig(f"{filename_base}.svg", bbox_inches="tight")
    fig.savefig(f"{filename_base}.pdf", bbox_inches="tight")


def add_panel_label(ax, label):
    ax.text(
        -0.10, 1.03, label,
        transform=ax.transAxes,
        fontsize=28,
        fontweight="bold",
        color=PALETTE["neutral_dark"],
        ha="left",
        va="bottom",
    )


def _is_finite_scalar(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _has_metric(rows, metric):
    return any(_is_finite_scalar(row.get(metric, np.nan)) for row in rows)


def _collect_rate_metrics(ours_rows, comp_results):
    metrics = []
    all_groups = [ours_rows] + list(comp_results.values())
    for metric in RATE_METRICS:
        if any(_has_metric(rows, metric) for rows in all_groups if rows):
            metrics.append(metric)
    return metrics


def _plot_line(ax, xs, ys, color, marker, label, lw=1.1, ls="-", zorder=3,
               markersize=4.6, markeredgewidth=1.0, markevery=None):
    pairs = [(x, y) for x, y in zip(xs, ys) if not np.isnan(float(y))]
    if not pairs:
        return
    px, py = zip(*pairs)
    ax.plot(px, py, color=color, marker=marker, linewidth=lw,
            linestyle=ls, markersize=markersize, markerfacecolor="none",
            markeredgewidth=markeredgewidth, label=label, zorder=zorder,
            markevery=markevery)


def _make_grid_fig(n_metrics, ncols=3, cell_w=4.5, cell_h=4.5):
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, cell_h * nrows))
    if nrows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    for mi in range(n_metrics, len(axes_flat)):
        axes_flat[mi].set_visible(False)
    return fig, axes_flat


def _make_single_fig(width=4.8, height=3.6):
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    return fig, ax


def _load_npz_dict(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _to_feature_matrix(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, None].astype(np.float32)
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1).astype(np.float32)
    return arr.astype(np.float32)


def _extract_labels(npz_dict):
    for key in ("spk_labels", "speaker_labels", "labels"):
        if key in npz_dict:
            return np.asarray(npz_dict[key]).reshape(-1)
    raise KeyError("Missing speaker labels in NPZ file.")


def _resolve_disentangle_panels(npz_dict):
    labels = _extract_labels(npz_dict)
    layer_feats = {}
    prefix = None
    for candidate in ("rvq", "cb"):
        candidate_feats = {}
        for key, value in npz_dict.items():
            if key.startswith(candidate) and key[len(candidate):].isdigit():
                candidate_feats[int(key[len(candidate):])] = _to_feature_matrix(value)
        if candidate_feats:
            prefix = candidate
            layer_feats = candidate_feats
            break

    if prefix is None or 1 not in layer_feats or 2 not in layer_feats:
        raise KeyError("Expected layer-1 and layer-2 features in disentangle NPZ.")

    later = None
    later_label = "Layers 2-8 Sum"
    if all(idx in layer_feats for idx in range(3, 9)):
        later_stack = [layer_feats[idx] for idx in range(3, 9)]
        base_dim = later_stack[0].shape[1]
        if all(feat.shape[1] == base_dim for feat in later_stack):
            later = np.sum(later_stack, axis=0)
        else:
            later = np.concatenate(later_stack, axis=1)
    elif f"{prefix}28" in npz_dict:
        later = _to_feature_matrix(npz_dict[f"{prefix}28"])
        if 2 in layer_feats and later.shape == layer_feats[2].shape:
            later = later - layer_feats[2]
    else:
        remaining = [feat for idx, feat in sorted(layer_feats.items()) if idx >= 3]
        if remaining:
            base_dim = remaining[0].shape[1]
            if all(feat.shape[1] == base_dim for feat in remaining):
                later = np.sum(remaining, axis=0)
            else:
                later = np.concatenate(remaining, axis=1)

    if later is None:
        raise KeyError("Could not resolve the aggregated later-layer feature.")

    panels = [
        ("Layer 1", layer_feats[1]),
        ("Layer 2", layer_feats[2]),
        (later_label, later),
    ]
    for _, feat in panels:
        if feat.shape[0] != len(labels):
            raise ValueError("Feature/sample count does not match speaker labels.")
    return labels, panels


def _make_speaker_colors(all_labels):
    unique_labels = list(dict.fromkeys(np.asarray(all_labels).tolist()))
    n_labels = len(unique_labels)
    palette = [
        "#FF1F1F", "#1F39FF", "#FF68B3", "#148A1A", "#9A6A5A",
        "#6A5ACD", "#E67E22", "#00A6A6", "#C0392B", "#2E86C1",
        "#D81B60", "#2ECC71", "#8E44AD", "#16A085", "#F39C12",
        "#34495E", "#E74C3C", "#3498DB", "#27AE60", "#7F8C8D",
    ]
    return {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}


def _tsne_embed(feats):
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("scikit-learn is required for the disentangle visualization.") from exc

    feats = np.asarray(feats, dtype=np.float32)
    feats = feats - feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    feats = feats / np.where(std < 1e-6, 1.0, std)
    n_samples = feats.shape[0]
    if n_samples < 4:
        raise ValueError("Need at least 4 samples for t-SNE visualization.")
    perplexity = min(30, max(5, n_samples // 4), n_samples - 1)
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(feats)


def _soft_legend(ax, loc):
    leg = ax.legend(
        fontsize=26,
        loc=loc,
        framealpha=0.0,
        handlelength=2.0,
        labelspacing=0.35,
        borderpad=0.25,
    )
    for txt in leg.get_texts():
        txt.set_fontweight("normal")
    return leg


def _bold_ticklabels(ax):
    plt.setp(ax.get_xticklabels(), fontweight="normal")
    plt.setp(ax.get_yticklabels(), fontweight="normal")


def _apply_curve_style(ax, xlabel, ylabel, legend_loc):
    ax.set_xlabel(xlabel, fontsize=26, fontweight="normal")
    ax.set_ylabel(ylabel, fontsize=26, fontweight="normal")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax.tick_params(labelsize=24, length=4.6, width=1.1)
    _bold_ticklabels(ax)
    ax.set_axisbelow(True)
    _soft_legend(ax, legend_loc)


def _infer_rate_ylims(metric, series_groups):
    values = []
    for rows in series_groups:
        for row in rows:
            value = row.get(metric, np.nan)
            if _is_finite_scalar(value):
                values.append(float(value))
    if not values:
        return None
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 0.08)
    pad = span * 0.10
    if metric == "wer":
        return max(0.0, lo - pad), min(1.0, hi + pad)
    return lo - pad, hi + pad


def _rate_title(rate_plr):
    if rate_plr is None:
        return "Quality vs average bitrate"
    return f"Quality vs average bitrate (PLR={int(round(float(rate_plr) * 100))}%)"


def plot_rate(data, out_dir):
    apply_publication_style(font_size=28, axes_linewidth=1.2)
    ours_rows = data["ours"]
    comp_results = data.get("competitors", {})
    rate_plr = float(data.get("rate_plr", 0.0))
    metrics = _collect_rate_metrics(ours_rows, comp_results)
    if not metrics:
        return

    ncols = min(len(metrics), 3)
    fig_w = 5.3 * ncols
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 5.9), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    series_groups = [ours_rows] + [rows for rows in comp_results.values() if rows]
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for name, cfg in RATE_COMPETITORS.items():
            rows = comp_results.get(name, [])
            if not _has_metric(rows, metric):
                continue
            comp_x = [r["bitrate_kbps"] for r in rows]
            comp_y = [r.get(metric, float("nan")) for r in rows]
            _plot_line(
                ax, comp_x, comp_y,
                cfg["color"], cfg["marker"], RATE_LEGEND_LABELS.get(name, name),
                lw=1.1, markersize=3.7, markeredgewidth=0.9, zorder=3,
            )
        _plot_line(
            ax,
            [r["bitrate_kbps"] for r in ours_rows],
            [r.get(metric, float("nan")) for r in ours_rows],
            PALETTE["ours_edge"], "o", RATE_LEGEND_LABELS["ours"],
            lw=1.8, zorder=6, markersize=4.0, markeredgewidth=0.9,
        )
        ax.set_xscale("log")
        ax.set_xlim(0.9, 21.0)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ylims = _infer_rate_ylims(metric, series_groups)
        if ylims is not None:
            ax.set_ylim(*ylims)
        legend_loc = "lower right" if metric != "wer" else "upper right"
        _apply_curve_style(ax, "Average bitrate (kbps)", METRIC_LABELS[metric], legend_loc)
        add_panel_label(ax, f"({chr(97 + idx)})")

    fig.suptitle(_rate_title(rate_plr), fontsize=28, y=1.05, fontweight="normal")
    plt.close(fig)

    visqol_only = _has_metric(ours_rows, "visqol") or any(
        _has_metric(rows, "visqol") for rows in comp_results.values()
    )
    if visqol_only:
        fig, axes = plt.subplots(
            1, 2, figsize=(13.6, 8.2),
            gridspec_kw={"width_ratios": [1.02, 1.0]}
        )
        ax_visqol, ax_utmos = axes

        style_map = {
            "AAC": {"color": "#9A6A5A", "marker": "s", "lw": 2.0, "ms": 11.0, "ls": "-", "z": 2},
            "Opus": {"color": "#148A1A", "marker": "o", "lw": 2.0, "ms": 11.0, "ls": "-", "z": 3},
            "EnCodec": {"color": "#1F39FF", "marker": "^", "lw": 2.2, "ms": 11.0, "ls": "-", "z": 4},
            "ESC": {"color": "#FF68B3", "marker": "v", "lw": 2.4, "ms": 11.0, "ls": "-", "z": 5},
            "ours": {"color": "#FF1F1F", "marker": "D", "lw": 2.8, "ms": 11.0, "ls": "-", "z": 6},
        }

        def _plot_filled(ax_obj, xs, ys, color, marker, label, lw, ms, ls, z):
            pairs = [(x, y) for x, y in zip(xs, ys) if not np.isnan(float(y))]
            if not pairs:
                return None
            px, py = zip(*pairs)
            line, = ax_obj.plot(
                px, py,
                color=color,
                marker=marker,
                linewidth=lw,
                linestyle=ls,
                markersize=ms,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.0,
                label=label,
                zorder=z,
            )
            return line

        handles = []
        handle_map = {}
        plot_order = ["AAC", "Opus", "EnCodec", "ESC"]
        for name in plot_order:
            rows = comp_results.get(name, [])
            if not _has_metric(rows, "visqol"):
                continue
            st = style_map[name]
            line = _plot_filled(
                ax_visqol,
                [r["bitrate_kbps"] for r in rows],
                [r.get("visqol", float("nan")) for r in rows],
                st["color"], st["marker"], RATE_LEGEND_LABELS.get(name, name),
                st["lw"], st["ms"], st["ls"], st["z"],
            )
            if line is not None:
                handles.append(line)
                handle_map[name] = line

        ours_x = [r["bitrate_kbps"] for r in ours_rows]
        ours_y = [r.get("visqol", float("nan")) for r in ours_rows]
        line = _plot_filled(
            ax_visqol, ours_x, ours_y,
            style_map["ours"]["color"], style_map["ours"]["marker"], "PDSSC",
            style_map["ours"]["lw"], style_map["ours"]["ms"],
            style_map["ours"]["ls"], style_map["ours"]["z"],
        )
        if line is not None:
            handles.append(line)
            handle_map["ours"] = line

        ax_visqol.set_xlim(0.8, 20.5)
        ax_visqol.set_ylim(1.8, 4.46)
        ax_visqol.set_xticks([1, 5, 10, 15, 20])
        ax_visqol.set_yticks([2.0, 2.5, 3.0, 3.5, 4.0])
        ax_visqol.set_xlabel("Rate (kbps)", fontsize=28, fontweight="normal")
        ax_visqol.set_ylabel("ViSQOL", fontsize=28, fontweight="normal")
        ax_visqol.minorticks_on()
        ax_visqol.grid(which="major", color="#BFBFBF", linewidth=0.55, alpha=0.85)
        ax_visqol.grid(which="minor", color="#E3E3E3", linewidth=0.35, alpha=0.95)
        ax_visqol.tick_params(labelsize=24, length=4.4, width=1.1)
        _bold_ticklabels(ax_visqol)
        ax_visqol.text(0.50, 1.005, "(a)", transform=ax_visqol.transAxes, fontsize=28, fontweight="normal", ha="center", va="bottom")

        esc_rows = comp_results.get("ESC", [])
        esc_x = [r["bitrate_kbps"] for r in esc_rows]
        esc_visqol = [r.get("visqol", float("nan")) for r in esc_rows]
        visqol_inset = ax_visqol.inset_axes([0.60, 0.12, 0.37, 0.34])
        visqol_inset.patch.set_facecolor((1.0, 1.0, 1.0, 0.72))
        if _has_metric(esc_rows, "visqol"):
            st = style_map["ESC"]
            _plot_filled(
                visqol_inset,
                esc_x,
                esc_visqol,
                st["color"], st["marker"], "ESC",
                st["lw"], 11.0, st["ls"], st["z"],
            )
        _plot_filled(
            visqol_inset,
            ours_x, ours_y,
            style_map["ours"]["color"], style_map["ours"]["marker"], "PDSSC",
            style_map["ours"]["lw"], 11.0, style_map["ours"]["ls"], style_map["ours"]["z"],
        )
        visqol_inset.set_xlim(0.9, 4.65)
        visqol_inset.set_ylim(3.80, 4.45)
        visqol_inset.set_xticks([1, 2, 3, 4])
        visqol_inset.set_yticks([3.8, 4.0, 4.2, 4.3])
        visqol_inset.grid(False)
        visqol_inset.tick_params(labelsize=16, length=2.7, width=0.9)
        _bold_ticklabels(visqol_inset)
        for side in ["top", "right", "bottom", "left"]:
            visqol_inset.spines[side].set_color("#8F8F8F")
            visqol_inset.spines[side].set_linewidth(0.65)

        for name in plot_order:
            rows = comp_results.get(name, [])
            if not _has_metric(rows, "utmos"):
                continue
            st = style_map[name]
            _plot_filled(
                ax_utmos,
                [r["bitrate_kbps"] for r in rows],
                [r.get("utmos", float("nan")) for r in rows],
                st["color"], st["marker"], RATE_LEGEND_LABELS.get(name, name),
                st["lw"], st["ms"], st["ls"], st["z"],
            )
        _plot_filled(
            ax_utmos,
            [r["bitrate_kbps"] for r in ours_rows],
            [r.get("utmos", float("nan")) for r in ours_rows],
            style_map["ours"]["color"], style_map["ours"]["marker"], "PDSSC",
            style_map["ours"]["lw"], style_map["ours"]["ms"],
            style_map["ours"]["ls"], style_map["ours"]["z"],
        )
        ax_utmos.set_xlim(0.8, 20.5)
        ax_utmos.set_ylim(1.2, 4.05)
        ax_utmos.set_xticks([1, 5, 10, 15, 20])
        ax_utmos.set_yticks([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        ax_utmos.set_xlabel("Rate (kbps)", fontsize=28, fontweight="normal")
        ax_utmos.set_ylabel("UTMOS", fontsize=28, fontweight="normal")
        ax_utmos.minorticks_on()
        ax_utmos.grid(which="major", color="#BFBFBF", linewidth=0.55, alpha=0.85)
        ax_utmos.grid(which="minor", color="#E3E3E3", linewidth=0.35, alpha=0.95)
        ax_utmos.tick_params(labelsize=24, length=4.4, width=1.1)
        _bold_ticklabels(ax_utmos)
        ax_utmos.text(0.50, 1.005, "(b)", transform=ax_utmos.transAxes, fontsize=28, fontweight="normal", ha="center", va="bottom")

        esc_utmos = [r.get("utmos", float("nan")) for r in esc_rows]
        ours_utmos = [r.get("utmos", float("nan")) for r in ours_rows]
        for ax_obj in axes:
            ax_obj.spines["top"].set_visible(True)
            ax_obj.spines["right"].set_visible(True)
            ax_obj.spines["top"].set_linewidth(0.8)
            ax_obj.spines["right"].set_linewidth(0.8)

        fig.subplots_adjust(left=0.038, right=0.992, top=0.950, bottom=0.15, wspace=0.024)
        pos_a = ax_visqol.get_position()
        pos_b = ax_utmos.get_position()
        ax_utmos.set_position([pos_b.x0 + 0.060, pos_b.y0, pos_a.width, pos_a.height])
        ax_utmos.yaxis.set_label_coords(-0.08, 0.5)
        legend_handles = [
            handle_map.get("ours"),
            handle_map.get("Opus"),
            handle_map.get("ESC"),
            handle_map.get("AAC"),
            handle_map.get("EnCodec"),
        ]
        legend_handles = [h for h in legend_handles if h is not None]
        legend = ax_utmos.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="lower right",
            bbox_to_anchor=(1.0, 0.0),
            ncol=1,
            fontsize=24,
            frameon=True,
            fancybox=False,
            framealpha=0.12,
            edgecolor="none",
            facecolor="white",
            markerfirst=False,
            alignment="right",
            handlelength=1.8,
            columnspacing=0.9,
            borderpad=0.05,
            borderaxespad=0.0,
            labelspacing=0.22,
        )
        for txt in legend.get_texts():
            txt.set_fontweight("normal")
            txt.set_ha("right")
        try:
            legend._legend_box.align = "right"
        except Exception:
            pass

        save_pub(fig, os.path.join(out_dir, "fig_rate_final"))
        plt.close(fig)

def plot_plr(data, out_dir):
    final = data["systems"]
    plr_list = data["plr_list"]
    plr_pct = [p * 100 for p in plr_list]

    fig, axes = plt.subplots(
        1, 2, figsize=(13.8, 8.2),
        gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    ax_visqol, ax_plcmos = axes

    def _plot_plr_series(ax_obj, xs, ys, style, label):
        pairs = [(x, y) for x, y in zip(xs, ys) if _is_finite_scalar(y)]
        if not pairs:
            return None
        px, py = zip(*pairs)
        line, = ax_obj.plot(
            px, py,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            markersize=style["ms"],
            markerfacecolor="none",
            markeredgecolor=style["color"],
            markeredgewidth=1.0,
            label=label,
        )
        return line

    handles = []
    handle_map = {}
    order = [
        "ours_1.5k", "ours_3.0k",
        "encodec_1.5k", "encodec_3.0k",
        "esc_1.5k", "esc_3.0k",
        "opus_8k",
    ]

    for key in order:
        if key not in final:
            continue
        style = _PLR_STYLE[key]
        label = PLR_SYSTEMS[key]["label"]
        line = _plot_plr_series(ax_visqol, plr_pct, final[key].get("visqol", []), style, label)
        _plot_plr_series(ax_plcmos, plr_pct, final[key].get("plcmos", []), style, label)
        if line is not None:
            handles.append(line)
            handle_map[key] = line

    for ax_obj, ylabel, panel in [
        (ax_visqol, "ViSQOL", "(a)"),
        (ax_plcmos, "PLCMOS", "(b)"),
    ]:
        ax_obj.set_xlim(-0.5, 30.5)
        ax_obj.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax_obj.set_xlabel("Packet loss rate (%)", fontsize=28, fontweight="normal")
        ax_obj.set_ylabel(ylabel, fontsize=28, fontweight="normal")
        ax_obj.minorticks_on()
        ax_obj.grid(which="major", color="#BFBFBF", linewidth=0.55, alpha=0.85)
        ax_obj.grid(which="minor", color="#E3E3E3", linewidth=0.35, alpha=0.95)
        ax_obj.tick_params(labelsize=24, length=4.6, width=1.15)
        _bold_ticklabels(ax_obj)
        ax_obj.text(0.50, 1.005, panel, transform=ax_obj.transAxes, fontsize=28, fontweight="normal", ha="center", va="bottom")

    ax_visqol.set_ylim(3.0, 4.35)
    ax_visqol.set_yticks([3.0, 3.3, 3.6, 3.9, 4.2])
    ax_plcmos.set_ylim(2.4, 4.1)
    ax_plcmos.set_yticks([2.4, 2.8, 3.2, 3.6, 4.0])

    for ax_obj in axes:
        ax_obj.spines["top"].set_visible(True)
        ax_obj.spines["right"].set_visible(True)
        ax_obj.spines["top"].set_linewidth(0.8)
        ax_obj.spines["right"].set_linewidth(0.8)

    legend_handles = [
        handle_map.get("ours_3.0k"),
        handle_map.get("ours_1.5k"),
        handle_map.get("encodec_3.0k"),
        handle_map.get("encodec_1.5k"),
        handle_map.get("esc_3.0k"),
        handle_map.get("esc_1.5k"),
        handle_map.get("opus_8k"),
    ]
    legend_handles = [h for h in legend_handles if h is not None]
    fig.subplots_adjust(left=0.038, right=0.992, top=0.950, bottom=0.15, wspace=0.024)
    pos_a = ax_visqol.get_position()
    pos_b = ax_plcmos.get_position()
    ax_plcmos.set_position([pos_b.x0 + 0.060, pos_b.y0, pos_a.width, pos_a.height])
    ax_plcmos.yaxis.set_label_coords(-0.08, 0.5)
    legend = ax_plcmos.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        ncol=1,
        fontsize=24,
        frameon=True,
        fancybox=False,
        framealpha=0.12,
        edgecolor="none",
        facecolor="white",
        handlelength=1.8,
        columnspacing=0.8,
        borderpad=0.05,
        borderaxespad=0.0,
        labelspacing=0.22,
    )
    for txt in legend.get_texts():
        txt.set_fontweight("normal")

    save_pub(fig, os.path.join(out_dir, "fig_plr_final"))
    plt.close(fig)


def plot_ablation(data, out_dir):
    result = {int(k): v for k, v in data["result"].items()}
    plr_list = data.get("plr_list", [data.get("fixed_plr", 0.05)])
    plr_pct = [p * 100 for p in plr_list]
    selected = [3, 6]
    fig, axes = plt.subplots(
        1, 2, figsize=(13.8, 8.2),
        gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    ax_visqol, ax_plcmos = axes

    def _plot_ablation_series(ax_obj, xs, ys, color, marker, label, ls, lw):
        pairs = [(x, y) for x, y in zip(xs, ys) if _is_finite_scalar(y)]
        if not pairs:
            return None
        px, py = zip(*pairs)
        line, = ax_obj.plot(
            px, py,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=lw,
            markersize=11.0,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.1,
            label=label,
        )
        return line

    handle_map = {}
    for N in selected:
        if N not in result:
            continue
        bitrate = N * 0.5
        style = _ABLATION_STYLE.get(N, {"color": "#7f7f7f", "marker": "o"})
        label_with = f"PDSSC (avg {bitrate:.1f} kbps)"
        label_without = f"w/o flow (avg {bitrate:.1f} kbps)"

        line = _plot_ablation_series(
            ax_visqol, plr_pct, result[N]["with"].get("visqol", []),
            style["color"], style["marker"], label_with, "-", 2.4
        )
        _plot_ablation_series(
            ax_visqol, plr_pct, result[N]["without"].get("visqol", []),
            style["color"], style["marker"], label_without, "--", 2.1
        )
        _plot_ablation_series(
            ax_plcmos, plr_pct, result[N]["with"].get("plcmos", []),
            style["color"], style["marker"], label_with, "-", 2.4
        )
        _plot_ablation_series(
            ax_plcmos, plr_pct, result[N]["without"].get("plcmos", []),
            style["color"], style["marker"], label_without, "--", 2.1
        )
        if line is not None:
            handle_map[(N, "with")] = line
        handle_map[(N, "without")] = Line2D(
            [], [],
            color=style["color"],
            marker=style["marker"],
            linestyle="--",
            linewidth=2.1,
            markersize=11.0,
            markerfacecolor="none",
            markeredgecolor=style["color"],
            markeredgewidth=1.1,
            label=label_without,
        )

    for ax_obj, ylabel, panel in [
        (ax_visqol, "ViSQOL", "(a)"),
        (ax_plcmos, "PLCMOS", "(b)"),
    ]:
        ax_obj.set_xlim(-0.5, 30.5)
        ax_obj.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax_obj.set_xlabel("Packet loss rate (%)", fontsize=28, fontweight="normal")
        ax_obj.set_ylabel(ylabel, fontsize=28, fontweight="normal")
        ax_obj.minorticks_on()
        ax_obj.grid(which="major", color="#BFBFBF", linewidth=0.55, alpha=0.85)
        ax_obj.grid(which="minor", color="#E3E3E3", linewidth=0.35, alpha=0.95)
        ax_obj.tick_params(labelsize=24, length=4.4, width=1.1)
        _bold_ticklabels(ax_obj)
        ax_obj.text(0.50, 1.005, panel, transform=ax_obj.transAxes, fontsize=28, fontweight="normal", ha="center", va="bottom")

    ax_visqol.set_ylim(3.7, 4.32)
    ax_visqol.set_yticks([3.7, 3.9, 4.1, 4.3])
    ax_plcmos.set_ylim(2.6, 4.05)
    ax_plcmos.set_yticks([2.6, 3.0, 3.4, 3.8, 4.0])

    for ax_obj in axes:
        ax_obj.spines["top"].set_visible(True)
        ax_obj.spines["right"].set_visible(True)
        ax_obj.spines["top"].set_linewidth(0.8)
        ax_obj.spines["right"].set_linewidth(0.8)

    legend_handles = []
    for N in [6, 3]:
        legend_handles.append(handle_map.get((N, "with")))
        legend_handles.append(handle_map.get((N, "without")))
    legend_handles = [h for h in legend_handles if h is not None]
    fig.subplots_adjust(left=0.038, right=0.992, top=0.950, bottom=0.15, wspace=0.024)
    pos_a = ax_visqol.get_position()
    pos_b = ax_plcmos.get_position()
    ax_plcmos.set_position([pos_b.x0 + 0.060, pos_b.y0, pos_a.width, pos_a.height])
    ax_plcmos.yaxis.set_label_coords(-0.08, 0.5)
    legend = ax_plcmos.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        ncol=1,
        fontsize=24,
        frameon=True,
        fancybox=False,
        framealpha=0.12,
        edgecolor="none",
        facecolor="white",
        handlelength=1.8,
        columnspacing=0.9,
        borderpad=0.05,
        borderaxespad=0.0,
        labelspacing=0.30,
    )
    for txt in legend.get_texts():
        txt.set_fontweight("normal")

    save_pub(fig, os.path.join(out_dir, "fig_ablation_final"))
    plt.close(fig)


def plot_disentangle(ours_npz_path, encodec_npz_path, out_dir):
    apply_publication_style(font_size=28, axes_linewidth=1.2)
    ours = _load_npz_dict(ours_npz_path)
    encodec = _load_npz_dict(encodec_npz_path)

    ours_labels, ours_panels = _resolve_disentangle_panels(ours)
    enc_labels, enc_panels = _resolve_disentangle_panels(encodec)
    color_map = _make_speaker_colors(np.concatenate([ours_labels, enc_labels], axis=0))

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 10.8))
    row_defs = [
        ("PDSSC", ours_labels, ours_panels),
        ("EnCodec", enc_labels, enc_panels),
    ]

    for row_idx, (row_name, row_labels, row_panels) in enumerate(row_defs):
        for col_idx, (title, feats) in enumerate(row_panels):
            ax = axes[row_idx, col_idx]
            emb = _tsne_embed(feats)
            for label in sorted(set(row_labels.tolist())):
                mask = row_labels == label
                ax.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    s=44,
                    c=[color_map[label]],
                    alpha=1.0,
                    edgecolors="#111111",
                    linewidths=0.65,
                    rasterized=True,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.75)
                spine.set_color("#C8C8C8")
            if row_idx == 0:
                ax.set_title(title, fontsize=28, pad=40, fontweight="normal")
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=28, labelpad=16, fontweight="normal")
                ax.yaxis.set_label_coords(-0.06, 0.5)
            else:
                ax.set_ylabel("")

    panel_tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for idx, ax in enumerate(axes.flatten()):
        ax.text(
            0.50, 1.02, panel_tags[idx],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=28,
            fontweight="bold",
            color=PALETTE["neutral_dark"],
        )

    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.05, wspace=0.10, hspace=0.10)
    save_pub(fig, os.path.join(out_dir, "fig_disentangle_final"))
    plt.close(fig)


def plot_latency(data, out_dir):
    # Keep latency JSON only; do not regenerate fig_latency.png.
    _ = data
    _ = out_dir
    return


def main():
    p = argparse.ArgumentParser(description="replot_final: regenerate figures from JSON/NPZ")
    p.add_argument("--data_dir", required=True,
                   help="Output directory generated by eval_final.py (contains JSON/NPZ).")
    p.add_argument("--out_dir", default=None,
                   help="PNG output directory (default: data_dir).")
    p.add_argument("--part", nargs="+",
                   default=["rate", "plr", "ablation", "latency"],
                   choices=["rate", "plr", "ablation", "latency", "vis"],
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
    if "latency" in args.part:
        data = _load_json("latency_data.json")
        if data:
            plot_latency(data, out_dir)
    if "vis" in args.part:
        ours_npz_path = os.path.join(args.data_dir, "disentangle_data.npz")
        encodec_npz_path = os.path.join(args.data_dir, "disentangle_encodec_data.npz")
        if not os.path.exists(ours_npz_path):
            print(f"[skip] missing: {ours_npz_path}")
        elif not os.path.exists(encodec_npz_path):
            print(f"[skip] missing: {encodec_npz_path}")
        else:
            plot_disentangle(ours_npz_path, encodec_npz_path, out_dir)


if __name__ == "__main__":
    main()
