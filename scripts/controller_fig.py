# -*- coding: utf-8 -*-
"""Controller-level evaluation replayed on the measured RD tables.

No new codec measurements: the RD surface Q(R, p) (with generative recovery,
N in {3,4,5,6} x PLR in {0..30%}) is taken from ablation_data.json and treated
as the frozen end-to-end pipeline. Two policies are compared under a uniform
distribution over the seven measured packet-loss states:

  * w/o controller (fixed gear): transmit a fixed layer count R in every state,
    with expected quality E_p[Q(R, p)].
  * w/ controller: per state p, switch to the gear R*(p) that maximizes
    Q(R, p) - lam*R (lam sweeps the average-bitrate operating point).

x-axis = average bitrate (kbps), y-axis = expected quality over the seven
packet-loss states. The two policies almost coincide in quality at equal
average bitrate (gap <= 0.01), which is the honest small-gain result; the
y-axis is narrowed so the gap remains visible.

Outputs (IEEE paper_figs dir):
  fig9_controller_visqol.pdf / .svg
  fig9_controller_plcmos.pdf / .svg
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import replot_single_panel as SP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(
    ROOT, "IEEE-Transactions-LaTeX2e-templates-and-instructions", "paper_figs")
DATA_DIR = os.path.join(ROOT, "root", "SpeechTokenizer-main", "output",
                        "eval_final_full")
ABL_JSON = os.path.join(DATA_DIR, "part3_ablation", "ablation_data.json")

WITH_C = "#FF1F1F"   # PDSSC with adaptive controller
FIXED_C = "#8C8C8C"  # PDSSC fixed gear (w/o controller)

GEARS = [1.5, 2.0, 2.5, 3.0]          # kbps, N*0.5 for N in {3,4,5,6}


def load_rd_table():
    with open(ABL_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    plr = d["plr_list"]
    res = d["result"]
    Q = {}
    for n, v in res.items():
        gear = float(n) * 0.5
        Q[gear] = {m: np.asarray(v["with"][m]) for m in ("visqol", "plcmos")}
    return plr, Q


def policy_curves(plr, Q, metric):
    """Return (fixed_x, fixed_y, adapt_x, adapt_y) in avg-bitrate/quality space.

    fixed: one point per gear, expected quality over the loss-state distribution.
    adapt: lambda sweep over per-state argmax policies, expected quality at the
           resulting average bitrate.
    """
    P = np.ones(len(plr)) / len(plr)
    gears = np.asarray(GEARS)
    Qa = np.array([[Q[g][metric][k] for g in GEARS] for k in range(len(plr))])

    fixed_x, fixed_y = gears, Qa.mean(axis=0)

    pts = {}
    for lam in np.linspace(2.0, 0.01, 200):
        idx = np.argmax(Qa - lam * gears, axis=1)
        rb = float((gears[idx] * P).sum())
        qb = float((Qa[np.arange(len(plr)), idx] * P).sum())
        pts[round(rb, 4)] = qb
    adapt = sorted(pts.items())
    return fixed_x, fixed_y, [r for r, _ in adapt], [q for _, q in adapt]


def main():
    os.makedirs(OUT, exist_ok=True)
    plr, Q = load_rd_table()

    for metric, fname, xlim, ylim, yticks in [
        ("visqol", "fig9_controller_visqol", (1.35, 3.15), (3.84, 4.24),
         [3.85, 3.95, 4.05, 4.15]),
        ("plcmos", "fig9_controller_plcmos", (1.35, 3.15), (3.32, 3.94),
         [3.35, 3.50, 3.65, 3.80]),
    ]:
        fx, fy, axx, axy = policy_curves(plr, Q, metric)

        fig, ax = plt.subplots(figsize=(SP.FIG_W, SP.FIG_H_PLR))
        l1 = SP._plot_line(ax, fx, fy, FIXED_C, "o", "-", 1.6, 3.4,
                           "w/o controller (fixed gear)", 3)
        l2 = SP._plot_line(ax, axx, axy, WITH_C, "D", "--", 1.9, 3.6,
                           "w/ adaptive controller", 4)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_xlabel("Average bitrate (kbps)", fontsize=SP.LABEL_FS,
                      fontweight="normal")
        ax.set_ylabel(SP.R.METRIC_LABELS[metric], fontsize=SP.LABEL_FS,
                      fontweight="normal")
        SP._apply_axis_style(ax)
        leg = SP._soft_legend(ax, [l2, l1], [l2.get_label(), l1.get_label()],
                              loc="upper left", markerfirst=True)
        fig.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.17)
        SP._save(fig, OUT, fname)
        plt.close(fig)
        print(f"wrote {fname}.pdf / .svg -> {OUT}")

        # key numbers for the narrative
        print(f"\n--- {metric}: expected-quality view (uniform loss states) ---")
        for r, q in zip(fx, fy):
            print(f"  fixed gear {r:.1f} kbps: E[Q] = {q:.4f}")
        gaps = [q - np.interp(r, fx, fy) for r, q in zip(axx, axy)
                if fx[0] <= r <= fx[-1]]
        print(f"  w/ controller: {len(axx)} points over avg rate "
              f"{axx[0]:.3f}-{axx[-1]:.3f} kbps")
        print(f"  max quality gap vs fixed line at equal avg rate: "
              f"{max(gaps):+.4f} ({metric})")


if __name__ == "__main__":
    main()
