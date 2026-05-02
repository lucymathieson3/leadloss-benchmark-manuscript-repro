#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
INPUT_WETH = (
    ROOT
    / "data"
    / "inputs"
    / "case8_fan_to_zero"
    / "machine_readable_csv"
    / "case8_fan_to_zero_synth_weth.csv"
)
KS_DIR = ROOT / "data" / "derived" / "case8_fan_to_zero" / "ks_diagnostics"
OUT_STUB = FIG_DIR / "fig08_case8_fan_to_zero"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fig02_synthetic_cases1to4 as base
import fig04_fig06_cdc_goodness_grids as grid


def draw_concordia(ax) -> None:
    df = pd.read_csv(INPUT_WETH).rename(columns={
        "pb207_u235_ratio": "x",
        "pb206_u238_ratio": "y",
        "pb207_u235_ratio_1sigma_abs": "x_err",
        "pb206_u238_ratio_1sigma_abs": "y_err",
        "true_upper_intercept_ma": "t_up_true",
        "true_lower_intercept_ma": "t_low_true",
        "case_id": "Case",
        "tier": "Tier",
        "sample_id": "Sample",
    })
    df = df[df["Tier"].astype(str).str.upper() == "A"].copy()

    tvals = np.linspace(0, 4500, 400)
    cx, cy = zip(*(base.wetherill_xy(tt) for tt in tvals))
    ax.plot(cx, cy, color="slategray", lw=1.0, zorder=0)

    guide_ages = (
        df.loc[df["is_concordant"], "t_up_true"]
        .dropna()
        .astype(float)
        .sort_values()
        .unique()
    )
    for age in guide_ages:
        x_u, y_u = base.wetherill_xy(age)
        ax.plot([x_u, 0.0], [y_u, 0.0], ls="--", lw=0.8, color="lightslategray", zorder=1)

    for x, y, sx, sy, is_c in df[["x", "y", "x_err", "y_err", "is_concordant"]].itertuples(index=False):
        patch = base.ellipse_patch(x, y, sx, sy, is_conc=bool(is_c), rho=base.RHO_CONST)
        if not bool(is_c):
            patch.set_facecolor((0.72, 0.58, 0.86, 0.90))
            patch.set_edgecolor("black")
            patch.set_linewidth(0.20)
            patch.set_zorder(4.5)
        ax.add_patch(patch)
        ax.plot(x, y, marker="o", ms=1.3, mfc="black", mec="none", zorder=6)

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel(r"$^{207}\mathrm{Pb}/^{235}\mathrm{U}$", fontsize=9)
    ax.set_ylabel(r"$^{206}\mathrm{Pb}/^{238}\mathrm{U}$", fontsize=9)
    ax.tick_params(direction="in", labelsize=7)

    handles = [
        Line2D([], [], ls="", marker="s", markersize=6, markerfacecolor=base.COL_CONC, markeredgecolor="black", label="Concordant"),
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
    x, _s_raw, s_pen = grid.load_npz_both("8A", KS_DIR)
    med = np.nanmedian(s_pen, axis=0)

    for y in s_pen:
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
    FIG_DIR.mkdir(parents=True, exist_ok=True)
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
