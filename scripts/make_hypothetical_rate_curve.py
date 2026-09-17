# -*- coding: utf-8 -*-
"""
Create a clearly labeled hypothetical rate-quality curve for internal discussion.

This script DOES NOT generate real experimental results.
It creates a synthetic curve located between EnCodec and Ours, and saves:
  - rate_data_hypothetical_distill_only.json
  - fig_rate_hypothetical_visqol.png
  - fig_rate_hypothetical_utmos.png

The generated data are explicitly marked as hypothetical/synthetic and are
intended only for internal comparison and discussion.
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _interp_rows(rows, xs_target, metric):
    xs = np.array([r["bitrate_kbps"] for r in rows], dtype=np.float64)
    ys = np.array([r[metric] for r in rows], dtype=np.float64)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return np.interp(xs_target, xs, ys, left=ys[0], right=ys[-1])


def build_hypothetical_curve(data, alpha=0.55):
    """
    alpha in (0,1): closer to ours when larger.
    hypothetical = encodec_interp + alpha * (ours - encodec_interp)
    """
    ours = data["ours"]
    encodec = data["competitors"]["EnCodec"]
    xs = np.array([r["bitrate_kbps"] for r in ours], dtype=np.float64)

    enc_visqol = _interp_rows(encodec, xs, "visqol")
    enc_utmos = _interp_rows(encodec, xs, "utmos")
    ours_visqol = np.array([r["visqol"] for r in ours], dtype=np.float64)
    ours_utmos = np.array([r["utmos"] for r in ours], dtype=np.float64)

    hyp_visqol = enc_visqol + alpha * (ours_visqol - enc_visqol)
    hyp_utmos = enc_utmos + alpha * (ours_utmos - enc_utmos)

    # Keep it strictly below ours for a clean visual interpretation.
    hyp_visqol = np.minimum(hyp_visqol, ours_visqol - 0.03)
    hyp_utmos = np.minimum(hyp_utmos, ours_utmos - 0.04)

    rows = []
    for x, vq, um in zip(xs, hyp_visqol, hyp_utmos):
        rows.append({
            "bitrate_kbps": float(x),
            "visqol": float(vq),
            "utmos": float(um),
        })
    return rows


def save_json(base_data, hyp_rows, out_path, alpha):
    out = {
        "_note": "HYPOTHETICAL / SYNTHETIC curve for internal discussion only. Not a real experimental result.",
        "_construction": "Interpolated between EnCodec and Ours.",
        "_alpha": alpha,
        "rate_plr": base_data.get("rate_plr"),
        "n_samples": base_data.get("n_samples"),
        "ours": base_data.get("ours"),
        "hypothetical_distill_only": hyp_rows,
        "competitors": base_data.get("competitors", {}),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def _plot_one(data, hyp_rows, metric, out_path):
    ours = data["ours"]
    competitors = data["competitors"]

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.8))

    style_map = {
        "AAC": ("#2F4B7C", "s"),
        "Opus": ("#2CA02C", "^"),
        "EnCodec": ("#355CFF", "o"),
        "ESC": ("#D84FD2", "*"),
    }

    for name, rows in competitors.items():
        if not rows:
            continue
        xs = [r["bitrate_kbps"] for r in rows]
        ys = [r[metric] for r in rows]
        c, m = style_map.get(name, ("#7f7f7f", "o"))
        ax.plot(xs, ys, color=c, marker=m, linewidth=1.2, markersize=4.0,
                markerfacecolor="none", label=name)

    xs_h = [r["bitrate_kbps"] for r in hyp_rows]
    ys_h = [r[metric] for r in hyp_rows]
    xs_o = [r["bitrate_kbps"] for r in ours]
    ys_o = [r[metric] for r in ours]

    ax.plot(xs_h, ys_h, color="#FF8C00", marker="D", linestyle="--",
            linewidth=1.5, markersize=4.2, markerfacecolor="none",
            label="distill-only")
    ax.plot(xs_o, ys_o, color="#D62728", marker="o", linestyle="-",
            linewidth=1.8, markersize=4.4, markerfacecolor="none",
            label="Proposed")

    ax.set_xlabel("Bitrate (kbps)", fontsize=10)
    ax.set_ylabel("VISQoL" if metric == "visqol" else "UTMOS", fontsize=10)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.5)

    ax.text(
        0.02, 0.03,
        "Hypothetical / synthetic curve\nfor internal discussion only",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#444444",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.65, edgecolor="#AAAAAA"),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--alpha", type=float, default=0.55)
    args = p.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    hyp_rows = build_hypothetical_curve(data, alpha=args.alpha)

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "rate_data_hypothetical_distill_only.json")
    save_json(data, hyp_rows, out_json, args.alpha)

    _plot_one(data, hyp_rows, "visqol", os.path.join(args.output_dir, "fig_rate_hypothetical_visqol.png"))
    _plot_one(data, hyp_rows, "utmos", os.path.join(args.output_dir, "fig_rate_hypothetical_utmos.png"))

    print("Saved:")
    print(out_json)
    print(os.path.join(args.output_dir, "fig_rate_hypothetical_visqol.png"))
    print(os.path.join(args.output_dir, "fig_rate_hypothetical_utmos.png"))


if __name__ == "__main__":
    main()
