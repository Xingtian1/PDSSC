# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg


SRC = Path(r"C:\Users\cheng\Desktop\SpeechTokenizer-main\root\SpeechTokenizer-main\output\eval_final_full\part4_vis\fig_vis_final.png")
OUT_DIR = Path(r"C:\Users\cheng\Desktop\SpeechTokenizer-main\IEEE-conference-template-062824\writing\paper_figs")
OUT_BASE = OUT_DIR / "fig_spec_case_final"


def main():
    img = mpimg.imread(SRC)
    h, w = img.shape[:2]

    # Manually tuned from the existing 3x3 comparison plate.
    lefts = [12, 615, 1218]
    rights = [600, 1203, 1806]
    tops = [46, 617, 1188]
    bottoms = [580, 1152, 1722]

    crops = [
        ("Original", 0, 0),
        ("PDSSC (1.5 kbps)", 0, 1),
        ("EnCodec (1.5 kbps)", 1, 0),
        ("ESC (1.5 kbps)", 1, 2),
        ("Opus + LBRR (8 kbps)", 2, 1),
        ("AAC + LFR-PLC (20 kbps)", 2, 2),
    ]

    # Same highlighted time-frequency region across all panels.
    rx0, ry0, rw, rh = 0.54, 0.18, 0.18, 0.24

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    axes = axes.flatten()

    for ax, (title, row, col) in zip(axes, crops):
        l, r = lefts[col], rights[col]
        t, b = tops[row], bottoms[row]
        panel = img[t:b, l:r]
        ph, pw = panel.shape[:2]
        ax.imshow(panel, aspect="auto")
        ax.set_title(title, fontsize=8.2, pad=3)
        ax.axis("off")

        rect = Rectangle(
            (rx0 * pw, ry0 * ph),
            rw * pw,
            rh * ph,
            linewidth=1.5,
            edgecolor="#E53935",
            facecolor="none",
        )
        ax.add_patch(rect)

    fig.subplots_adjust(left=0.02, right=0.985, top=0.94, bottom=0.03, wspace=0.03, hspace=0.13)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
