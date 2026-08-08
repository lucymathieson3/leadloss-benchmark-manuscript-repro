#!/usr/bin/env python3
"""
fig_concordia_193435.py

Tera-Wasserburg concordia diagram for SHRIMP U-Pb analyses of sample 193435
(GSWA Geochronology Record 960; Wingate et al., 2011), the natural case study
for the ensemble CDC workflow. Plots all 31 analyses classified concordant
(teal) / discordant (grey) by the 2-sigma error-ellipse concordance test, with
2-sigma error crosses, and marks the two recovered Pb-loss ages (454, 1113 Ma).

Input : 193435_tw_concordia_classified.csv  (204-corrected T-W ratios +
        1-sigma abs uncertainties + concordance classification), as archived in
        data/derived/193435_cdc_2026-04-20/ of the reproducibility repository.
Output: concordia_193435.{pdf,svg,png}

Usage : python fig_concordia_193435.py \
            --csv data/derived/193435_cdc_2026-04-20/193435_tw_concordia_classified.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams

# Conventional U decay constants (Jaffey et al. 1971; Steiger & Jaeger 1977)
L235, L238, RU = 9.8485e-4, 1.55125e-4, 137.818   # per Ma; 238U/235U


def wetherill(t):
    """Wetherill concordia coordinates (207Pb/235U, 206Pb/238U) at age t (Ma)."""
    return np.exp(L235 * t) - 1.0, np.exp(L238 * t) - 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=Path("data/derived/193435_cdc_2026-04-20/"
                                 "193435_tw_concordia_classified.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--stub", default="figS1_concordia_193435")
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    args = ap.parse_args()

    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 8.5, "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
    })

    df = pd.read_csv(args.csv)
    u = df["u238_pb206_ratio_204corr"].to_numpy()
    su = df["u238_pb206_ratio_204corr_1sigma_abs"].to_numpy()
    v = df["pb207_pb206_ratio_204corr"].to_numpy()
    sv = df["pb207_pb206_ratio_204corr_1sigma_abs"].to_numpy()
    is_c = (df["classification"] == "concordant").to_numpy()

    y = 1.0 / u                                    # 206Pb/238U
    x = RU * v / u                                 # 207Pb/235U
    sy = su / u**2
    sx = np.sqrt((RU / u)**2 * sv**2 + (RU * v / u**2)**2 * su**2)

    INK, SOFT = "#1f1f1f", "#555555"
    TEAL, TEAL_E = "#2f8f7f", "#1c554b"
    GREY, GREY_E = "#a9adad", "#5a5f5f"
    CROSS, RED = "#8c9090", "#9c5148"

    fig, ax = plt.subplots(figsize=(8.6 / 2.54, 8.0 / 2.54))
    t = np.linspace(2, 2980, 900)
    cx, cy = wetherill(t)
    ax.plot(cx, cy, color=INK, lw=1.1, zorder=3)
    for tm in (500, 1000, 1500, 2000, 2500):
        mx, my = wetherill(tm)
        ax.plot(mx, my, "o", ms=2.6, color=INK, zorder=6)
        off, ha = ((5, -7), "left") if tm == 500 else ((-5, 1), "right")
        ax.annotate(f"{tm}", (mx, my), textcoords="offset points",
                    xytext=off, ha=ha, va="center", fontsize=7, color=SOFT)
    for age, lab, off, ha in [(453.72, "454 Ma", (0, -11), "center"),
                              (1112.67, "1113 Ma", (6, -2), "left")]:
        mx, my = wetherill(age)
        ax.plot(mx, my, marker="D", ms=4.0, mfc=RED, mec="white", mew=0.5,
                zorder=8)
        ax.annotate(lab, (mx, my), textcoords="offset points", xytext=off,
                    ha=ha, fontsize=7, color=RED)
    ax.errorbar(x, y, xerr=2 * sx, yerr=2 * sy, fmt="none", ecolor=CROSS,
                elinewidth=0.6, capsize=0, zorder=4)
    ax.scatter(x[~is_c], y[~is_c], s=15, facecolor=GREY, edgecolor=GREY_E,
               linewidth=0.4, zorder=5)
    ax.scatter(x[is_c], y[is_c], s=15, facecolor=TEAL, edgecolor=TEAL_E,
               linewidth=0.4, zorder=6)

    ax.set_xlim(0, 13.9)
    ax.set_ylim(0, 0.545)
    ax.set_xlabel(r"$^{207}\mathrm{Pb}/^{235}\mathrm{U}$")
    ax.set_ylabel(r"$^{206}\mathrm{Pb}/^{238}\mathrm{U}$")
    ax.tick_params(length=3.2, width=0.6, color=INK)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", ls="none", mfc=TEAL, mec=TEAL_E, ms=6,
               label=f"Concordant ({int(is_c.sum())})"),
        Line2D([0], [0], marker="o", ls="none", mfc=GREY, mec=GREY_E, ms=6,
               label=f"Discordant ({int((~is_c).sum())})"),
        Line2D([0], [0], marker="D", ls="none", mfc=RED, mec="white", ms=6,
               label="Recovered Pb-loss age")],
        loc="lower right", frameon=False, fontsize=6.6, borderaxespad=0.5)

    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in args.formats:
        fig.savefig(args.outdir / f"{args.stub}.{ext}", dpi=300,
                    bbox_inches="tight")
    print("wrote", [str(args.outdir / f'{args.stub}.{e}') for e in args.formats])


if __name__ == "__main__":
    main()
