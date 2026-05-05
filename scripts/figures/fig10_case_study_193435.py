#!/usr/bin/env python3
"""
case_study_figure.py  (v6 — 2026-04-20)

Three-panel case-study figure tying the CDC ensemble result for sample
193435 (Burtville Terrane, eastern Yilgarn) to independent regional age
constraints.

  Panel A  CDC ensemble goodness-of-fit curve for 193435: Monte-Carlo
           realisations + the ensemble median. Two accepted
           peaks are annotated with their CIs ("stability bounds") and
           direct/winner support fractions from the peak-picker.

  Panel B  Sample 185138 (GSWA 738) zircon best-age distribution — all
           19 zircons, shown as a histogram (grey) with an Isoplot-style
           PDP curve (red) using each grain's analytical 1σ. Within the
           0–1400 Ma window this reduces to the two Group-C grains
           (496 & 1066 Ma), labelled inline. Archean cores (n = 17)
           plot off-scale above 1400 Ma and are noted.

  Panel C  Regional Western Australia + central Australia *Pb-loss-
           relevant* events for 0–1400 Ma, compiled from GSWA +
           Geoscience Australia sources. Only the two classes of event that most directly
           drive U–Pb Pb-loss in zircon are shown:
             - Thermal / cooling (Ar–Ar, Rb–Sr, K–Ar) — records the
               times at which the host rocks cooled through closure
               temperatures, typically during post-orogenic thermal
               relaxation or late fluid events.
             - Direct overprint / reheat / hydrothermal / dyke
               emplacement (incl. U–Pb baddeleyite dyke ages) — the
               most direct analogues for Pb-loss drivers.
           Crystallisation ages of unrelated plutons and metamorphic
           zircon rims are deliberately excluded because they don't
           drive Pb-loss in previously-crystallised detrital zircon.

           Each layer is drawn as a stacked histogram plus a PDP curve
           (one-sigma Gaussians, peak-normalised to the histogram).

Vertical shaded bands behind panels B and C mark the CDC stability
bounds from Panel A (444–484 Ma light blue; 826–1255 Ma light orange).

Inputs:
  papers/2025-peak-picking/data/raw/185138_spot_ages.csv
  data/inputs/case_study_193435/machine_readable_csv/
      regional_cooling_ages.csv
  papers/2025-peak-picking/data/derived/193435_cdc_2026-04-20/
      193435_curve.csv, 193435_spaghetti.csv

Outputs:
  papers/2025-peak-picking/outputs/figures/case_study_figure.(png|pdf|svg)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Paths assume this script lives at <archive>/scripts/figures/case_study_figure.py
# and reads from <archive>/data/inputs/case_study_193435/machine_readable_csv/ and
# <archive>/data/derived/193435_cdc_2026-04-20/. Outputs go to
# <archive>/05_Final_Manuscript_Figures/.
PAPER_DIR = Path(__file__).resolve().parents[2]

DEFAULT_SPOTS_CSV = (
    PAPER_DIR
    / "data"
    / "inputs"
    / "case_study_193435"
    / "machine_readable_csv"
    / "185138_spot_ages.csv"
)
DEFAULT_REGIONAL_CSV = (
    PAPER_DIR
    / "data"
    / "inputs"
    / "case_study_193435"
    / "machine_readable_csv"
    / "regional_cooling_ages.csv"
)
DEFAULT_CDC_DIR = PAPER_DIR / "data" / "derived" / "193435_cdc_2026-04-20"
DEFAULT_OUTDIR = PAPER_DIR / "05_Final_Manuscript_Figures"


# ---------------------------------------------------------------------------
# Style — muted publication palette
# ---------------------------------------------------------------------------

INK = "#1f1f1f"
INK_SOFT = "#555555"
MID_GREY = "#8a8a8a"
PALE_GREY = "#c9c9c9"
SPAG_GREY = "#b8b8b8"
HIST_FILL = "#d9d9d9"

METAMORPH_NAVY = "#2c3e70"  # U–Pb metamorphic zircon / monazite layer
COOL_TEAL = "#2a7a7a"       # Ar–Ar / Rb–Sr / K–Ar cooling layer
OVERPRINT_PLUM = "#8c3a5c"  # Direct overprint / reheat / dyke / hydrothermal

LEAD_RED = "#b2272c"        # Group-C / peak marker
PEAK_LINE = "#b2272c"

# Muted stability-band fills — desaturated to read as neutral shading,
# not saturated pastel. Peak 1 is a cool slate, Peak 2 a warm taupe.
BAND_PEAK1 = "#b4bfcc"
BAND_PEAK2 = "#c6b79a"
BAND_ALPHA = 0.28


def apply_style():
    # Latin Modern Roman — the LaTeX book serif — gives a distinctly
    # "typeset paper" feel and avoids the default matplotlib serif look.
    # Math is rendered in
    # the matching Computer Modern / LM Math set via mathtext.
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = [
        "Latin Modern Roman", "CMU Serif", "Nimbus Roman",
        "Bitstream Charter", "Liberation Serif", "DejaVu Serif",
    ]
    rcParams["mathtext.fontset"] = "cm"
    rcParams["font.size"] = 8.5
    rcParams["axes.labelsize"] = 8.5
    rcParams["axes.titlesize"] = 8.5
    rcParams["legend.fontsize"] = 7.5
    rcParams["xtick.labelsize"] = 8
    rcParams["ytick.labelsize"] = 8
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.top"] = True
    rcParams["ytick.right"] = False
    rcParams["axes.linewidth"] = 0.6
    rcParams["xtick.major.width"] = 0.6
    rcParams["ytick.major.width"] = 0.6
    rcParams["axes.grid"] = False
    rcParams["lines.solid_capstyle"] = "round"
    # PDF output embeds TrueType/Type42 so fonts render correctly when
    # the file is viewed on a machine without LMR installed.
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42


# ---------------------------------------------------------------------------
# Peaks from the stability-freeze GUI run for 193435
# ---------------------------------------------------------------------------

PEAKS = [
    dict(label="Peak 1", age_ma=453.72, ci_low_ma=443.66, ci_high_ma=483.90,
         direct_pct=96, winner_pct=24, band_color=BAND_PEAK1),
    dict(label="Peak 2", age_ma=1112.67, ci_low_ma=825.95, ci_high_ma=1254.78,
         direct_pct=88, winner_pct=73, band_color=BAND_PEAK2),
]


# ---------------------------------------------------------------------------
# Pb-loss-relevant event classification for Panel C.
#
# We DELIBERATELY exclude "age of igneous crystallization" events because Pb
# in zircon is lost during post-crystallisation thermal / fluid perturbation
# — metamorphism, cooling through closure temperatures, dyke-driven
# hydrothermal circulation, or direct reheating — not during emplacement of
# unrelated plutons elsewhere.
# ---------------------------------------------------------------------------

COOLING_ANALYSIS_TYPES = {
    "Ar-Ar hornblende", "Ar-Ar", "K-Ar", "Rb-Sr biotite",
}
# "Metamorphism" is tagged by ANALYSIS_EVENTDATED containing the word
# "metamorph"; the typical analysis types are U-Pb zircon (metamorphic),
# U-Pb monazite, titanite, etc.
OVERPRINT_KEYWORDS = (
    "overprint", "reheat", "reset", "reworked",
    "hydrotherm", "alteration",
    "mafic intrusion", "dyke",
)

# Tectonic unit filter for "WA + central Australia".
WA_CENTRAL_UNIT_SUBSTRINGS = [
    "Pinjarra", "Albany-Fraser", "Musgrave", "Yeneena", "Paterson",
    "Bentley", "Arunta", "Amadeus", "Leeuwin", "Northampton",
    "Capricorn", "Rudall", "Madura", "Coompana", "Officer", "Canning",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_cdc_curve(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    return df["age_ma"].to_numpy(float), df["goodness"].to_numpy(float)


def load_cdc_spaghetti(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not csv_path.is_file():
        return np.empty(0), np.empty((0, 0))
    df = pd.read_csv(csv_path)
    ages = df["age_ma"].to_numpy(float)
    runs = df.drop(columns=["age_ma"]).to_numpy(float).T
    return ages, runs


def load_spot_ages(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "spot_id": "spot",
        "figure_group_code": "group",
        "th_u_ratio": "th_u",
        "f204_percent": "f204_pct",
        "discordance_percent": "disc_pct",
        "age_206pb_238u_ma": "age_206_238_ma",
        "age_206pb_238u_1sigma_ma": "age_206_238_sig_ma",
        "age_207pb_206pb_ma": "age_207_206_ma",
        "age_207pb_206pb_1sigma_ma": "age_207_206_sig_ma",
        "preferred_age_ma": "best_age_ma",
        "preferred_age_1sigma_ma": "best_age_sig_ma",
        "preferred_age_system": "best_age_system",
        "source_reference": "source",
    })
    for col in ("age_207_206_ma", "age_207_206_sig_ma",
                "age_206_238_ma", "age_206_238_sig_ma",
                "best_age_ma", "best_age_sig_ma"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["group"] = df["group"].astype(str)
    return df


def load_regional_master(csv_path: Path, x_max: float) -> pd.DataFrame:
    """Load the WA geochron master, filter to WA + central Australia units
    and 0–x_max Ma, and classify each record as ``metamorphism``,
    ``cooling``, ``overprint``, or ``exclude`` (for crystallisation /
    detrital / otherwise-irrelevant ages).
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.rename(columns={
        "sample_id": "SAMPLE_ID",
        "tectonic_unit": "TECTONIC_UNIT",
        "region": "REGION",
        "analysis_type": "ANALYSIS_TYPE",
        "closure_temperature_band": "CLOSURE_T_BAND",
        "analysis_event_dated": "ANALYSIS_EVENTDATED",
        "analysis_age_ma": "ANALYSIS_AGE",
        "analysis_uncertainty_ma": "ANALYSIS_UNCERTAINTY",
        "analysis_uncertainty_confidence_level": "ANALYSIS_CONFIDLEVEL",
        "latitude": "LATITUDE",
        "longitude": "LONGITUDE",
        "source_reference": "SOURCE_REFERENCE",
    })
    df["age_ma"] = pd.to_numeric(df["ANALYSIS_AGE"], errors="coerce")
    df["sigma_ma"] = pd.to_numeric(df["ANALYSIS_UNCERTAINTY"], errors="coerce")

    unit = df["TECTONIC_UNIT"].astype(str)
    in_wa = unit.apply(lambda u: any(s in u for s in WA_CENTRAL_UNIT_SUBSTRINGS))
    df = df[in_wa].copy()
    df = df.dropna(subset=["age_ma"])
    df = df[(df["age_ma"] >= 0) & (df["age_ma"] <= x_max)]

    ed = df["ANALYSIS_EVENTDATED"].astype(str).str.lower()
    at = df["ANALYSIS_TYPE"].astype(str)

    def classify(row_type: str, row_event: str) -> str:
        t = row_type.lower()
        # Baddeleyite = dyke emplacement = overprint-relevant
        if "baddeleyite" in t:
            return "overprint"
        # Direct overprint / reset / hydrothermal / dyke keywords
        if any(k in row_event for k in OVERPRINT_KEYWORDS):
            return "overprint"
        # Cooling (Ar-Ar, Rb-Sr, K-Ar) or explicit "cooling age"
        if row_type in COOLING_ANALYSIS_TYPES or "cooling" in row_event:
            return "cooling"
        # Metamorphism — by event label, or by U-Pb "(plot_ready metamorphic)"
        if "metamorph" in row_event:
            return "metamorphism"
        if "metamorphic" in t:
            return "metamorphism"
        return "exclude"

    df["kind"] = [classify(t, e) for t, e in zip(at, ed)]
    df = df[df["kind"] != "exclude"].copy()

    # Sigma floor: where missing/zero, substitute max(1% of age, 10 Ma).
    bad = ~np.isfinite(df["sigma_ma"]) | (df["sigma_ma"] <= 0)
    df.loc[bad, "sigma_ma"] = np.maximum(0.01 * df.loc[bad, "age_ma"], 10.0)

    return df[["SAMPLE_ID", "TECTONIC_UNIT", "ANALYSIS_TYPE",
               "ANALYSIS_EVENTDATED", "kind", "age_ma", "sigma_ma"]]


# ---------------------------------------------------------------------------
# U–Pb ratios from ages (for Wetherill concordia inset)
# ---------------------------------------------------------------------------

# Steiger & Jäger (1977) decay constants, in /Ma.
LAMBDA_238 = 1.55125e-4
LAMBDA_235 = 9.8485e-4
# 238U/235U (Hiess et al., 2012).
U238_U235 = 137.818


def tw_from_ages(t68_ma: np.ndarray, t76_ma: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Tera–Wasserburg coordinates (238U/206Pb*, 207Pb/206Pb*) from a
    pair of ages: 206/238 age fixes the x-ratio; 207/206 age fixes y."""
    t68 = np.asarray(t68_ma, float)
    t76 = np.asarray(t76_ma, float)
    r68 = np.exp(LAMBDA_238 * t68) - 1.0
    x = 1.0 / r68
    y = (1.0 / U238_U235) * (
        (np.exp(LAMBDA_235 * t76) - 1.0) /
        (np.exp(LAMBDA_238 * t76) - 1.0)
    )
    return x, y


def tw_concordia_curve(t_grid_ma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tera–Wasserburg concordia curve, parameterised by age."""
    r68 = np.exp(LAMBDA_238 * t_grid_ma) - 1.0
    x = 1.0 / r68
    y = (1.0 / U238_U235) * (
        (np.exp(LAMBDA_235 * t_grid_ma) - 1.0) /
        (np.exp(LAMBDA_238 * t_grid_ma) - 1.0)
    )
    return x, y


# ---------------------------------------------------------------------------
# PDP / KDE
# ---------------------------------------------------------------------------

def pdp(ages_ma: np.ndarray, sigmas_ma: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Isoplot-style PDP: sum of Gaussians, each centred on an age with
    σ = its analytical 1σ. Normalised to integrate to 1 over the grid.
    """
    ages_ma = np.asarray(ages_ma, float)
    sigmas_ma = np.asarray(sigmas_ma, float)
    y = np.zeros_like(x, dtype=float)
    n = 0
    for a, s in zip(ages_ma, sigmas_ma):
        if not (np.isfinite(a) and np.isfinite(s) and s > 0):
            continue
        y += np.exp(-0.5 * ((x - a) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        n += 1
    if n > 0:
        y /= n
    return y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def shade_stability_bands(ax):
    for p in PEAKS:
        ax.axvspan(p["ci_low_ma"], p["ci_high_ma"],
                   color=p["band_color"], alpha=BAND_ALPHA,
                   zorder=0, linewidth=0)


# Candidate tectono-thermal drivers that lie inside each peak's CI window.
# Peak 2 in particular has multiple converging candidates; the sample
# location (−28.51° S, 123.02° E, Bailey Range) lies inside the recognised
# extent of the Warakurna LIP (Wingate et al., 2004).
PEAK1_DRIVERS = [
    dict(name="Alice Springs Orogeny", age_ma=450, note="distal (?)"),
    dict(name="Paterson / Pinjarra cooling", age_ma=470, note="regional"),
]
PEAK2_DRIVERS = [
    dict(name="Warakurna LIP", age_ma=1075, note="on-site (Fig. 59)"),
    dict(name="older Pinjarra Orogeny", age_ma=1050, note="margin"),
    dict(name="Musgrave–Albany-Fraser II", age_ma=1180, note="overlap"),
]


def annotate_band_drivers(ax):
    """Write a compact label above each stability band listing candidate
    tectono-thermal drivers that lie inside the CI.  Uses the axis
    transform in x and axis-fraction 0.98 in y so the labels sit flush
    at the top of whatever panel they're drawn into."""
    # Peak 1 — short, hedged
    ax.text(
        0.5 * (PEAKS[0]["ci_low_ma"] + PEAKS[0]["ci_high_ma"]), 0.985,
        "Alice Springs? /\nPaterson–Pinjarra cooling",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=6.0, color=INK_SOFT,
        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                  ec="none", alpha=0.82),
        zorder=7,
    )
    # Peak 2 — named LIP first, on-site emphasis
    ax.text(
        0.5 * (PEAKS[1]["ci_low_ma"] + PEAKS[1]["ci_high_ma"]), 0.985,
        "Warakurna LIP (on site)\nolder Pinjarra / Musgrave",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=6.0, color=INK_SOFT,
        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                  ec="none", alpha=0.82),
        zorder=7,
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_cdc(ax, curve_x, curve_y, spag_x, spag_runs, *, max_spaghetti=200):
    shade_stability_bands(ax)
    annotate_band_drivers(ax)

    if spag_runs.size and spag_runs.ndim == 2:
        R = spag_runs.shape[0]
        if R > max_spaghetti:
            idx = np.linspace(0, R - 1, max_spaghetti).astype(int)
            sp = spag_runs[idx]
        else:
            sp = spag_runs
        for i in range(sp.shape[0]):
            ax.plot(spag_x, sp[i], color=SPAG_GREY, lw=0.35,
                    alpha=0.15, zorder=1)
        mc_handle = Line2D([0], [0], color=SPAG_GREY, lw=1.0, alpha=0.85,
                           label=f"MC realisations (n={R})")
    else:
        mc_handle = None

    ax.plot(curve_x, curve_y, color=INK, lw=1.3, zorder=4,
            label="Ensemble median")

    curve_ymax = float(np.nanmax(curve_y)) if curve_y.size else 1.0
    curve_ymin = float(np.nanmin(curve_y)) if curve_y.size else 0.0
    yr = curve_ymax - curve_ymin if curve_ymax > curve_ymin else 1.0
    ax.set_ylim(curve_ymin - 0.08 * yr, curve_ymax + 0.55 * yr)

    for i, p in enumerate(PEAKS):
        ax.axvline(p["age_ma"], color=PEAK_LINE, lw=0.9, alpha=0.95,
                   zorder=5)
        g_here = np.interp(p["age_ma"], curve_x, curve_y)
        ax.plot([p["age_ma"]], [g_here], marker="v", color=PEAK_LINE,
                ms=5.5, mec="white", mew=0.5, zorder=6)

        label_text = (
            f"{p['age_ma']:.0f} Ma\n"
            f"CI {p['ci_low_ma']:.0f}–{p['ci_high_ma']:.0f} Ma\n"
            f"direct {p['direct_pct']}%, winner {p['winner_pct']}%"
        )
        ha = "left" if i == 0 else "right"
        x_off = 60 if i == 0 else -60
        ax.annotate(
            label_text,
            xy=(p["age_ma"], g_here), xycoords="data",
            xytext=(p["age_ma"] + x_off,
                    curve_ymin + 0.96 * (
                        curve_ymax + 0.55 * yr - curve_ymin)),
            textcoords="data",
            color=INK, fontsize=6.8, ha=ha, va="top",
            arrowprops=dict(arrowstyle="-", color=PEAK_LINE, lw=0.5,
                            shrinkA=0, shrinkB=2),
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="none", alpha=0.85),
        )

    ax.set_ylabel("Ensemble goodness of fit")
    ax.set_title(
        "(a) CDC ensemble for sample 193435 (Burtville Terrane)",
        loc="left", fontsize=8.5, pad=3,
    )

    handles = [Line2D([0], [0], color=INK, lw=1.3, label="Ensemble median")]
    if mc_handle is not None:
        handles.append(mc_handle)
    # Peak labels read from 193435_peaks.csv so legend stays in sync with data.
    try:
        _peaks_path = DEFAULT_CDC_DIR / "193435_peaks.csv"
        _peaks_df = pd.read_csv(_peaks_path)
        _p1 = _peaks_df.iloc[0]
        _p2 = _peaks_df.iloc[1]
        _p1_label = f"Peak 1 stability ({_p1['ci_low_ma']:.0f}–{_p1['ci_high_ma']:.0f} Ma)"
        _p2_label = f"Peak 2 stability ({_p2['ci_low_ma']:.0f}–{_p2['ci_high_ma']:.0f} Ma)"
    except Exception:
        # Fallback to manuscript values if peaks.csv is absent or malformed.
        _p1_label = "Peak 1 stability (444–484 Ma)"
        _p2_label = "Peak 2 stability (826–1255 Ma)"
    handles.append(Patch(facecolor=BAND_PEAK1, alpha=BAND_ALPHA,
                         edgecolor="none",
                         label=_p1_label))
    handles.append(Patch(facecolor=BAND_PEAK2, alpha=BAND_ALPHA,
                         edgecolor="none",
                         label=_p2_label))
    ax.legend(handles=handles, loc="lower left", frameon=True,
              framealpha=0.9, edgecolor="none",
              handletextpad=0.5, borderaxespad=0.5, fontsize=6.8)


def panel_sample_hist(ax, spots: pd.DataFrame, x_min: float, x_max: float,
                      x_grid: np.ndarray, *, bin_width_ma: float = 40.0):
    """Histogram + PDP of sample 185138 best ages in the 0–x_max window.

    The PDP uses each grain's ``best_age_sig_ma`` as Gaussian σ
    (Isoplot-style) and is peak-normalised to the histogram maximum
    so both curves share a count-like y-axis.
    """
    shade_stability_bands(ax)

    ages_best = spots["best_age_ma"].to_numpy(float)
    sigmas_best = spots["best_age_sig_ma"].to_numpy(float)
    good = np.isfinite(ages_best) & np.isfinite(sigmas_best) & (sigmas_best > 0)
    ages_best = ages_best[good]
    sigmas_best = sigmas_best[good]

    in_win = (ages_best >= x_min) & (ages_best <= x_max)
    ages_win = ages_best[in_win]
    sigmas_win = sigmas_best[in_win]

    bins = np.arange(x_min, x_max + bin_width_ma, bin_width_ma)
    counts, _, _ = ax.hist(
        ages_win, bins=bins, color=HIST_FILL, edgecolor=INK_SOFT,
        linewidth=0.45, zorder=1,
        label=f"185138 zircons (n={len(ages_win)} in window)")

    peak_count = float(counts.max()) if counts.size else 1.0
    peak_count = max(peak_count, 1.0)
    # Extra headroom above the PDP/label band so the concordia inset
    # has clear space and doesn't collide with Group-C age labels.
    ax.set_ylim(0, peak_count * 3.6)

    # Isoplot-style PDP scaled so its peak matches the histogram peak.
    y_pdp = pdp(ages_win, sigmas_win, x_grid)
    ymax = float(np.nanmax(y_pdp)) if y_pdp.size else 0.0
    if ymax > 0:
        y_pdp_scaled = y_pdp / ymax * peak_count
        ax.fill_between(x_grid, 0, y_pdp_scaled,
                        color=LEAD_RED, alpha=0.18, lw=0, zorder=2)
        ax.plot(x_grid, y_pdp_scaled, color=LEAD_RED, lw=1.2, zorder=3,
                label=r"Sample PDP (best-age $\pm$ 1$\sigma$)")

    # Inline labels for the two Group-C grains — these are the analyses
    # GSWA's 185138 record flags as "probable processing contaminants",
    # but which in this paper we argue are real age-matched Pb-loss
    # overgrowths (cf. 193435's CDC peaks at 454 & 1113 Ma; see Discussion).
    # Place the left grain above its peak, and the right grain below —
    # the right-hand label would otherwise collide with the concordia
    # inset anchored in the upper-right of this panel.
    groupC = spots[spots["group"] == "C"]
    for _, row in groupC.iterrows():
        a = row["best_age_ma"]
        if not np.isfinite(a) or a < x_min or a > x_max:
            continue
        if a < 700.0:
            # 496 Ma — above-peak label, arrow down
            ax.annotate(
                f"Group-C  {a:.0f} Ma",
                xy=(a, peak_count * 1.02), xytext=(a, peak_count * 1.30),
                textcoords="data",
                color=LEAD_RED, fontsize=6.8, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=LEAD_RED, lw=0.5,
                                shrinkA=0, shrinkB=2),
            )
        else:
            # 1066 Ma — sits under the concordia inset, so anchor the
            # label below/left of the PDP peak with a horizontal stub
            ax.annotate(
                f"Group-C  {a:.0f} Ma",
                xy=(a, peak_count * 0.55),
                xytext=(a - 130.0, peak_count * 1.30),
                textcoords="data",
                color=LEAD_RED, fontsize=6.8, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=LEAD_RED, lw=0.5,
                                shrinkA=0, shrinkB=2,
                                connectionstyle="arc3,rad=0.0"),
            )

    n_above = int(np.sum(ages_best > x_max))
    n_within = int(np.sum((ages_best >= x_min) & (ages_best <= x_max)))
    note = (f"{n_within}/{len(ages_best)} zircons in 0–{x_max:.0f} Ma  •  "
            f"{n_above} Archean cores (~2.6–2.76 Ga) off-scale above")
    ax.text(0.01, 0.98, note, transform=ax.transAxes,
            ha="left", va="top", fontsize=6.4, color=INK_SOFT)

    ax.set_ylabel("Count")
    ax.set_title(
        "(b) Sample 185138 zircon best-age distribution (GSWA 738)",
        loc="left", fontsize=8.5, pad=3,
    )
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.62),
              frameon=False, handletextpad=0.5, borderaxespad=0.0,
              fontsize=6.8)

    # Wetherill concordia inset (upper-right of panel B).
    _concordia_inset(ax, spots)


def _concordia_inset(parent_ax, spots: pd.DataFrame) -> None:
    """Tera–Wasserburg concordia inset for sample 185138, styled after
    GSWA Record 2018/1 Figure 2 (Wingate et al.). Marker conventions:
      Group 1 (magmatic)                    — yellow filled squares
      Group 2 (radiogenic Pb loss)          — black filled squares
      Group C (probable processing contam.) — crossed diamonds
      Group D (ungrouped, disc >10%)        — crossed squares
    """
    # Position the inset in the upper-right quadrant, tall enough that the
    # TW curve isn't vertically compressed. Plot-area aspect ≈ 1.2 : 1
    # matches the shape of GSWA Record 2018/1 Figure 2.
    inset = parent_ax.inset_axes([0.62, 0.42, 0.36, 0.55])

    # Heavy grey concordia curve with filled tick circles every 100 Ma
    # — the published GSWA style.
    t_curve = np.linspace(400.0, 3100.0, 600)
    xc, yc = tw_concordia_curve(t_curve)
    inset.plot(xc, yc, color="#9a9a9a", lw=2.6, zorder=2,
               solid_capstyle="round")

    t_ticks_small = np.arange(500.0, 3001.0, 100.0)
    xt, yt = tw_concordia_curve(t_ticks_small)
    inset.plot(xt, yt, marker="o", linestyle="none",
               ms=2.0, mfc="white", mec=INK, mew=0.5, zorder=3)

    for t_tick, pad in ((500, (4, -8)), (1000, (-2, -10)),
                        (1500, (-12, 2)), (2000, (-12, 2)),
                        (2500, (-14, 0))):
        xa, ya = tw_concordia_curve(np.array([float(t_tick)]))
        inset.annotate(f"{t_tick}", xy=(xa[0], ya[0]),
                       xytext=pad, textcoords="offset points",
                       fontsize=5.6, color=INK, ha="center", va="center")

    # Compute TW coordinates for every grain and plot by group.
    t68 = spots["age_206_238_ma"].to_numpy(float)
    t76 = spots["age_207_206_ma"].to_numpy(float)
    groups = spots["group"].astype(str).to_numpy()
    good = np.isfinite(t68) & np.isfinite(t76)
    t68, t76, groups = t68[good], t76[good], groups[good]
    x, y = tw_from_ages(t68, t76)

    g1 = groups == "1"
    g2 = groups == "2"
    gC = groups == "C"
    gD = groups == "D"

    if g1.any():
        inset.plot(x[g1], y[g1], marker="s", linestyle="none",
                   ms=4.5, mfc="#e8c84a", mec=INK, mew=0.5,
                   zorder=5, label=f"Group 1 magmatic (n={int(g1.sum())})")
    if g2.any():
        inset.plot(x[g2], y[g2], marker="s", linestyle="none",
                   ms=4.5, mfc=INK, mec=INK, mew=0.5,
                   zorder=5, label=f"Group 2 Pb loss (n={int(g2.sum())})")
    # Crossed diamonds and squares — draw open marker then overlay an ×.
    if gC.any():
        inset.plot(x[gC], y[gC], marker="D", linestyle="none",
                   ms=5.0, mfc="white", mec=INK, mew=0.6, zorder=5,
                   label=f"Group C contam. (n={int(gC.sum())})")
        inset.plot(x[gC], y[gC], marker="x", linestyle="none",
                   ms=3.6, color=INK, mew=0.8, zorder=6)
    if gD.any():
        inset.plot(x[gD], y[gD], marker="s", linestyle="none",
                   ms=4.5, mfc="white", mec=INK, mew=0.6, zorder=5,
                   label=f"Group D ungrouped (n={int(gD.sum())})")
        inset.plot(x[gD], y[gD], marker="x", linestyle="none",
                   ms=3.2, color=INK, mew=0.8, zorder=6)

    inset.set_xlabel(r"$^{238}$U/$^{206}$Pb*", fontsize=6.3, labelpad=1)
    inset.set_ylabel(r"$^{207}$Pb/$^{206}$Pb*", fontsize=6.3, labelpad=1)
    inset.tick_params(axis="both", labelsize=5.6, length=2, width=0.4)
    for s in inset.spines.values():
        s.set_linewidth(0.45)
    inset.set_xlim(1, 14)
    inset.set_ylim(0.04, 0.22)
    inset.text(0.50, 0.97,
               f"185138  Tera--Wasserburg  (n={int(good.sum())})",
               transform=inset.transAxes, ha="center", va="top",
               fontsize=6.2, color=INK)
    # Legend tucked into the lower-left quadrant, which is clear of both
    # the concordia curve (which exits the top-left at ~3 Ga) and the
    # single Group-C grain at ~500 Ma in the far right.
    inset.legend(loc="lower left", bbox_to_anchor=(0.30, 0.02),
                 frameon=False, fontsize=5.0,
                 handletextpad=0.35, borderaxespad=0.2,
                 labelspacing=0.25)


# ---------------------------------------------------------------------------
# Location / tectonic inset for Panel C
#
# A small schematic map of Western + central Australia that answers the reader's
# natural question "why is Warakurna LIP relevant to this sample?".  Shows
# (i) the Australian continental outline (simplified hand-coded polygon),
# (ii) the approximate extent of the Yilgarn Craton, (iii) the recognised
# extent of the Warakurna LIP (Wingate et al., 2004; Smithies et al., 2005),
# and (iv) the location of sample 193435 (Bailey Range, Burtville Terrane,
# eastern Yilgarn — inside the Warakurna footprint).  All polygons are
# schematic and drawn from published map outlines; this inset is a
# locator, not a data map.
# ---------------------------------------------------------------------------

# Hand-coded Australia/WA outline — simplified coastline points in (lon, lat).
# Drawn from a 1:50M AusBoundaries figure; deliberately low resolution so the
# inset stays legible at small sizes.
_AUS_OUTLINE = [
    (114.1, -21.8),  # North West Cape
    (115.8, -20.3),  # Port Hedland
    (119.4, -19.9),  # Eighty Mile Beach south
    (122.5, -17.7),  # Broome / Cape Leveque entry
    (124.0, -16.5),  # Derby
    (126.5, -14.5),  # Kimberley coast
    (129.0, -14.9),  # NT corner approach
    (129.0, -11.9),  # NT top (Joseph Bonaparte)
    (132.5, -12.2),  # Darwin region
    (135.3, -14.9),  # Arnhem edge
    (136.0, -17.0),  # Gulf of Carpentaria W shore
    (138.0, -19.0),  # NT/QLD border inland
    (138.0, -26.0),  # SA/NT/QLD tri-junction
    (141.0, -26.0),  # eastern extent of panel
    (141.0, -32.0),  # eastern extent (SA border)
    (138.5, -35.0),  # SA coast near Adelaide
    (136.0, -35.0),  # Eyre Peninsula
    (133.5, -32.0),  # Great Australian Bight N
    (129.0, -32.0),  # WA/SA border on coast
    (123.0, -33.8),  # south of Eucla / Esperance
    (120.5, -33.9),  # Esperance
    (117.5, -35.0),  # Albany
    (115.1, -34.4),  # Cape Leeuwin
    (115.8, -31.9),  # Perth
    (114.6, -28.8),  # Geraldton
    (113.3, -26.1),  # Shark Bay
    (113.7, -24.8),  # Carnarvon / Ningaloo
    (114.1, -21.8),  # back to North West Cape
]

# Yilgarn Craton — simplified polygon.  Coordinates from GSWA 1:2.5M tectonic
# map (simplified for an inset).
_YILGARN_POLY = [
    (116.0, -26.5),   # NW corner (Narryer boundary approx)
    (118.0, -26.0),
    (121.5, -25.8),   # NE — Laverton/Leonora area
    (123.5, -26.5),
    (124.0, -29.0),   # E edge — against Officer / Musgrave transition
    (123.7, -32.5),   # SE — Albany-Fraser contact
    (122.0, -33.7),   # S coast
    (118.0, -34.0),
    (116.2, -33.5),
    (115.9, -31.0),   # Perth / Pinjarra boundary
    (116.5, -28.5),
    (116.0, -26.5),
]

# Warakurna LIP recognised extent — after Wingate et al. (2004, EPSL) and
# Smithies et al. (2005, Precambrian Research).  The LIP is defined by
# ~1075 Ma mafic intrusions and volcanism across the Musgrave Province,
# western Arunta, Bentley Basin and eastern Yilgarn; its footprint is
# approximate and this polygon is a generalisation of published extent maps.
_WARAKURNA_POLY = [
    (122.5, -22.5),   # NW — into Canning / Paterson
    (126.0, -21.5),
    (130.0, -22.0),   # N — western Arunta
    (134.5, -22.5),
    (136.5, -24.5),   # NE — central Arunta
    (136.0, -27.5),   # E — Musgrave tail
    (133.5, -28.5),
    (130.0, -28.7),
    (127.0, -29.0),
    (124.0, -29.1),   # W — edge over Burtville Terrane
    (122.8, -27.5),
    (122.5, -25.0),
    (122.5, -22.5),
]

# Sample 193435 location — converted from MGA Zone 51, 502441E 6845871N.
SAMPLE_LON, SAMPLE_LAT = 123.0249, -28.5139


def _location_inset(parent_ax) -> None:
    """Compact WA + central-Australia locator map pinned in the upper-left
    of Panel C.  Sized just large enough that the Yilgarn, Warakurna
    footprint, and sample dot are individually resolvable."""
    inset = parent_ax.inset_axes([0.015, 0.40, 0.245, 0.57])

    # Australia outline — light grey fill, subtle ink edge.
    aus_x = [p[0] for p in _AUS_OUTLINE]
    aus_y = [p[1] for p in _AUS_OUTLINE]
    inset.fill(aus_x, aus_y, facecolor="#eeeeee", edgecolor=INK_SOFT,
               linewidth=0.5, zorder=2)

    # Warakurna LIP extent — desaturated plum (matches Panel C overprint).
    wk_x = [p[0] for p in _WARAKURNA_POLY]
    wk_y = [p[1] for p in _WARAKURNA_POLY]
    inset.fill(wk_x, wk_y, facecolor=OVERPRINT_PLUM, alpha=0.28,
               edgecolor=OVERPRINT_PLUM, linewidth=0.6, zorder=3)

    # Yilgarn Craton outline — drawn as an outline only so the Warakurna
    # overlap (eastern Yilgarn) remains visually obvious.
    yg_x = [p[0] for p in _YILGARN_POLY]
    yg_y = [p[1] for p in _YILGARN_POLY]
    inset.plot(yg_x, yg_y, color=COOL_TEAL, linewidth=0.8,
               linestyle=(0, (3, 1.2)), zorder=4)

    # Sample dot — red star with white halo for pop.
    inset.plot([SAMPLE_LON], [SAMPLE_LAT], marker="*",
               markersize=8, markerfacecolor=LEAD_RED,
               markeredgecolor="white", markeredgewidth=0.7, zorder=6)
    inset.annotate(
        "193435",
        xy=(SAMPLE_LON, SAMPLE_LAT),
        xytext=(4, -5), textcoords="offset points",
        fontsize=5.6, color=LEAD_RED, ha="left", va="top",
        fontweight="bold", zorder=7,
    )

    # Feature labels — tuned for legibility against the shaded polygons.
    inset.text(128.5, -25.5, "Warakurna\nLIP", fontsize=5.2,
               color=OVERPRINT_PLUM, ha="center", va="center",
               fontstyle="italic", zorder=5)
    inset.text(119.0, -30.5, "Yilgarn\nCraton", fontsize=5.2,
               color=COOL_TEAL, ha="center", va="center",
               fontstyle="italic", zorder=5)
    inset.text(133.5, -16.0, "NT", fontsize=5.0, color=INK_SOFT,
               ha="center", va="center")
    inset.text(117.0, -24.0, "WA", fontsize=5.0, color=INK_SOFT,
               ha="center", va="center")
    inset.text(137.5, -30.5, "SA", fontsize=5.0, color=INK_SOFT,
               ha="center", va="center")

    # North arrow (top-left, inside axes).
    inset.annotate(
        "", xy=(114.8, -13.0), xytext=(114.8, -15.0),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6,
                        mutation_scale=6),
    )
    inset.text(114.8, -12.0, "N", fontsize=5.4, color=INK,
               ha="center", va="bottom", fontweight="bold")

    # Axis cosmetics — clip to WA + central Australia, aspect by latitude.
    inset.set_xlim(112.5, 141.5)
    inset.set_ylim(-36.5, -11.0)
    # aspect of 1.15 roughly compensates for lon-vs-lat at -25° S
    # (cos(-25°) ≈ 0.91 → lat-per-lon ≈ 1.1).
    inset.set_aspect(1.15)
    inset.set_xticks([])
    inset.set_yticks([])
    for s in inset.spines.values():
        s.set_linewidth(0.45)
        s.set_color(INK_SOFT)

    inset.text(
        0.5, 1.02,
        "Sample locality and Warakurna extent",
        transform=inset.transAxes, ha="center", va="bottom",
        fontsize=5.4, color=INK,
    )


def panel_regional_pdp(ax, regional: pd.DataFrame, x: np.ndarray,
                       x_min: float, x_max: float,
                       *, bin_width_ma: float = 40.0, rug_frac: float = 0.06):
    """Two-layer histogram + PDP of Pb-loss-relevant regional events:
      - Thermal / cooling (Ar–Ar, Rb–Sr, K–Ar)
      - Direct overprint / reheat / hydrothermal / dyke

    Metamorphism and crystallisation are excluded upstream — crystalline
    growth of unrelated plutons doesn't drive Pb-loss, and the
    metamorphism catalogue is dominated by Pinjarran events off-target
    for this sample. Each layer is drawn as a stacked count histogram
    (cooling below, overprint on top) plus a PDP curve peak-normalised
    to the combined-histogram maximum.
    """
    shade_stability_bands(ax)

    cool = regional[regional["kind"] == "cooling"]
    over = regional[regional["kind"] == "overprint"]

    cool_ages = cool["age_ma"].to_numpy(float)
    over_ages = over["age_ma"].to_numpy(float)

    bins = np.arange(x_min, x_max + bin_width_ma, bin_width_ma)
    counts, _, _ = ax.hist(
        [cool_ages, over_ages],
        bins=bins, stacked=True,
        color=[COOL_TEAL, OVERPRINT_PLUM],
        edgecolor=INK_SOFT, linewidth=0.35,
        alpha=0.72, zorder=2,
        label=[f"Thermal / cooling (n={len(cool)})",
               f"Overprint / reheat / dyke (n={len(over)})"],
    )
    # counts is (2, nbins) for stacked=True; we want the combined top
    # envelope for scaling the PDPs.
    if counts.ndim == 2:
        combined = counts.sum(axis=0)
    else:
        combined = counts
    peak_count = float(combined.max()) if combined.size else 1.0
    peak_count = max(peak_count, 1.0)
    ax.set_ylim(-rug_frac * 2.6 * peak_count, peak_count * 1.55)

    # PDPs — Isoplot-style, each peak-normalised to the histogram max
    # so they sit neatly on the same count axis.
    y_cool = pdp(cool_ages, cool["sigma_ma"].to_numpy(float), x)
    y_over = pdp(over_ages, over["sigma_ma"].to_numpy(float), x)

    def _scale(y):
        m = float(np.nanmax(y)) if y.size else 0.0
        return (y / m) * peak_count if m > 0 else y

    y_cool_s = _scale(y_cool)
    y_over_s = _scale(y_over)

    ax.plot(x, y_cool_s, color=COOL_TEAL, lw=1.3, zorder=4,
            linestyle=(0, (3, 1.5)),
            label="Cooling PDP")
    ax.plot(x, y_over_s, color=OVERPRINT_PLUM, lw=1.4, zorder=4,
            linestyle=(0, (1.5, 1.2)),
            label="Overprint PDP")

    # Rug ticks at the bottom — one row per layer.
    rug_y_cool = -rug_frac * peak_count
    rug_y_over = -rug_frac * 2.1 * peak_count
    if len(cool):
        ax.plot(cool_ages, np.full(len(cool_ages), rug_y_cool),
                marker="|", linestyle="none", mew=0.8, ms=5,
                color=COOL_TEAL, alpha=0.9, zorder=5)
    if len(over):
        ax.plot(over_ages, np.full(len(over_ages), rug_y_over),
                marker="|", linestyle="none", mew=1.0, ms=5.5,
                color=OVERPRINT_PLUM, alpha=0.95, zorder=5)

    ax.set_ylabel("Count")
    ax.set_xlabel("Age (Ma)")
    ax.set_title(
        "(c) WA + central Australia Pb-loss drivers "
        "(cooling + overprint, GSWA + Geoscience Australia)",
        loc="left", fontsize=8.5, pad=3,
    )
    # Data-source footer: identifies the compilation and filter criteria
    # directly on the figure. Full source details
    # (per-record source, filter rules, record counts) are documented in
    # papers/2025-peak-picking/data/raw/README_panel_c_data.md.  Placed
    # below the x-axis label so it reads as a footnote, not plot content.
    ax.text(
        0.0, -0.32,
        f"Source: GSWA Geochronology + GA SHRIMP, filtered to Pb-loss drivers "
        f"(cooling + overprint) in WA + central-Australia tectonic units "
        f"(n={len(regional)}); see data/raw/README_panel_c_data.md",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=5.8, color=INK_SOFT, style="italic",
        clip_on=False, zorder=7,
    )
    ax.legend(loc="upper right", frameon=False,
              handletextpad=0.5, borderaxespad=0.4, fontsize=6.8,
              ncol=1)

    # NOTE: the locator / Warakurna-extent inset is prepared separately in
    # QGIS (with proper coordinates, grid lines, and a published LIP
    # polygon) and will be composited on the final figure, so the
    # hand-coded schematic inset is not drawn here.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots-csv", type=Path, default=DEFAULT_SPOTS_CSV)
    ap.add_argument("--regional-csv", type=Path, default=DEFAULT_REGIONAL_CSV)
    ap.add_argument("--curve-csv", type=Path,
                    default=DEFAULT_CDC_DIR / "193435_curve.csv")
    ap.add_argument("--spaghetti-csv", type=Path,
                    default=DEFAULT_CDC_DIR / "193435_spaghetti.csv")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--stub", default="case_study_figure")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    ap.add_argument("--x-min", type=float, default=0.0)
    ap.add_argument("--x-max", type=float, default=1400.0)
    ap.add_argument("--grid-n", type=int, default=2000)
    ap.add_argument("--bin-width-sample", type=float, default=40.0)
    ap.add_argument("--max-spaghetti", type=int, default=200)
    ap.add_argument("--width-cm", type=float, default=19.0)
    args = ap.parse_args(argv)

    apply_style()

    curve_x, curve_y = load_cdc_curve(args.curve_csv)
    spag_x, spag_runs = load_cdc_spaghetti(args.spaghetti_csv)
    spots = load_spot_ages(args.spots_csv)
    regional = load_regional_master(args.regional_csv, x_max=args.x_max)

    # Drop metamorphism after loading — keep only cooling + overprint
    # as Pb-loss drivers for Panel C.
    regional = regional[regional["kind"].isin(["cooling", "overprint"])].copy()

    print(f"[case_study_figure] regional 0–{args.x_max:.0f} Ma: "
          f"{len(regional)} Pb-loss drivers "
          f"(cooling={int((regional['kind']=='cooling').sum())}, "
          f"overprint={int((regional['kind']=='overprint').sum())})")

    x = np.linspace(args.x_min, args.x_max, args.grid_n)

    width_in = args.width_cm / 2.54
    height_in = width_in * 1.18
    fig, axes = plt.subplots(
        3, 1, figsize=(width_in, height_in), sharex=True,
        gridspec_kw=dict(height_ratios=[1.15, 1.15, 1.0], hspace=0.20),
    )
    ax_a, ax_b, ax_c = axes

    panel_cdc(ax_a, curve_x, curve_y, spag_x, spag_runs,
              max_spaghetti=args.max_spaghetti)
    panel_sample_hist(ax_b, spots, args.x_min, args.x_max, x,
                      bin_width_ma=args.bin_width_sample)
    panel_regional_pdp(ax_c, regional, x, args.x_min, args.x_max,
                       bin_width_ma=args.bin_width_sample)

    for ax in axes:
        ax.set_xlim(args.x_min, args.x_max)
        ax.tick_params(length=3.5, width=0.55)
        for spine in ("top", "right"):
            ax.spines[spine].set_linewidth(0.5)
            ax.spines[spine].set_color(INK_SOFT)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        out = args.outdir / f"{args.stub}.{fmt}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"[case_study_figure] wrote {out}")

    plt.close(fig)
    print("[case_study_figure] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
