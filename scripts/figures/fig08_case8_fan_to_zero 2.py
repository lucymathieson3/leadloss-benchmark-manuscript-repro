#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "Peak-Picking-v2" / "LeadLoss"
FIG_HELPERS = REPO / "papers" / "2025-peak-picking" / "scripts" / "figures"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(FIG_HELPERS) not in sys.path:
    sys.path.insert(0, str(FIG_HELPERS))

import fig03_fig05_cdc_goodness_grids as grid
import render_case8_revised_concordia as conc


KS_DIR = HERE / "cdc_run_outputs_ABC" / "ks_diagnostics"
OUT_STUB = HERE / "Case8_Revised_Combined"


def draw_concordia(ax) -> None:
    spots = conc.load_spots()
    weth = conc.load_wetherill()
    concordant_rows, discordant_rows, _reverse_rows = conc.split_wetherill_rows(weth, spots)
    concordant, _, _ = conc.classify_spots(spots)

    tvals = np.linspace(0, 4500, 400)
    cx, cy = zip(*(conc.base.wetherill_xy(tt) for tt in tvals))
    ax.plot(cx, cy, color="slategray", lw=1.0, zorder=0)

    for age in conc.guide_ages_ma(concordant):
        x_u, y_u = conc.base.wetherill_xy(age)
        ax.plot([x_u, 0.0], [y_u, 0.0], ls="--", lw=0.8, color="lightslategray", zorder=1)

    conc.draw_population(ax, concordant_rows, is_conc=True)
    conc.draw_population(
        ax,
        discordant_rows,
        is_conc=False,
        clip_patch=conc.below_concordia_clip_patch(ax),
    )

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel(r"$^{207}\mathrm{Pb}/^{235}\mathrm{U}$", fontsize=9)
    ax.set_ylabel(r"$^{206}\mathrm{Pb}/^{238}\mathrm{U}$", fontsize=9)
    ax.tick_params(direction="in", labelsize=7)

    handles = [
        Line2D([], [], ls="", marker="s", markersize=6, markerfacecolor=conc.base.COL_CONC, markeredgecolor="black", label="Concordant"),
        Line2D([], [], ls="", marker="s", markersize=6, markerfacecolor=(0.72, 0.58, 0.86, 0.90), markeredgecolor="black", label="Discordant"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, borderaxespad=0.3, handlelength=1.0, handletextpad=0.4)
    ax.annotate(
        "Fan-like discordance\nno common lower intercept",
        xy=(7.3, 0.11),
        xytext=(10.2, 0.22),
        fontsize=7,
        color="0.35",
        arrowprops=dict(
            arrowstyle="->",
            color="0.45",
            lw=0.8,
            connectionstyle="arc3,rad=-0.25",
        ),
        ha="left",
        va="center",
    )


def draw_goodness(ax) -> None:
    x, _S_raw, S_pen = grid.load_npz_both("8A", KS_DIR)
    med = np.nanmedian(S_pen, axis=0)

    for y in S_pen:
        ax.plot(x, y, color="0.75", lw=0.4, alpha=0.55, zorder=1)
    ax.plot(x, med, color="k", lw=1.6, zorder=2)

    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Pb-loss age (Ma)", fontsize=9)
    ax.set_ylabel(r"Normalised goodness, $S$", fontsize=9)
    ax.tick_params(direction="in", labelsize=7)

    handles = [
        Line2D([0], [0], color="0.75", lw=1.2, label="all runs $S(t)$"),
        Line2D([0], [0], color="k", lw=1.5, label="median $S(t)$"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    ax.annotate(
        "No peak\n= no discrete Pb-loss age",
        xy=(460, 0.27),
        xytext=(1030, 0.58),
        fontsize=7,
        color="0.35",
        arrowprops=dict(
            arrowstyle="->",
            color="0.45",
            lw=0.8,
            connectionstyle="arc3,rad=0.25",
        ),
        ha="center",
        va="center",
    )


def render() -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.15),
        gridspec_kw={"wspace": 0.32, "left": 0.08, "right": 0.985, "bottom": 0.18, "top": 0.92},
    )

    draw_concordia(axes[0])
    draw_goodness(axes[1])

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_STUB.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    render()
