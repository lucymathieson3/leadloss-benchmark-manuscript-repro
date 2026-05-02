#!/usr/bin/env python3
"""sensitivity_analysis_extended.py

Extended parameter sensitivity analysis for the CDC ensemble pipeline.

Loads pre-computed per-run goodness surfaces from the benchmark NPZ diagnostics
and re-runs *only* the ensemble catalogue step with varied parameter settings.
No Monte Carlo rerun required.

Sweeps the four gate-keeping thresholds already reported in the manuscript
robustness analysis, plus six additional shape and pipeline parameters that
were not previously varied. Together these cover all overridable arguments
of build_ensemble_catalogue() that could in principle affect peak retention.

Parameters varied (one at a time, others held at defaults):
  Already in manuscript:
    1. f_p               — ensemble prominence threshold
    2. per_run_prom_frac — per-run prominence threshold
    3. support_min       — minimum MC support fraction
    4. f_v               — valley-merge depth threshold
  New in this sweep:
    5. r_min             — absolute minimum supporting runs
    6. delta_min         — flat-surface abstention floor
    7. f_d               — peak-distance fraction
    8. per_run_min_dist  — per-run minimum peak separation (nodes)
    9. per_run_min_width — per-run minimum peak width (nodes)

Outputs:
  - CSV table per parameter
  - Combined summary CSV

Author: Lucy Mathieson
"""

from __future__ import annotations

import csv
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ARCHIVE_ROOT = _SCRIPT_DIR.parents[1]   # <archive>/scripts/tables/this.py -> <archive>
_DIAG_DIR = _ARCHIVE_ROOT / "data" / "derived" / "ks_diagnostics"
_OUT_DIR = _ARCHIVE_ROOT / "data" / "derived" / "sensitivity_extended"

# This script needs the LeadLoss application source (process.ensemble) on
# PYTHONPATH. In the reproducibility archive, a pinned src/ snapshot can be
# bundled at <archive>/src. If that is absent, fall back to LEADLOSS_SRC.
import os
_ARCHIVE_SRC = _ARCHIVE_ROOT / "src"
_LEADLOSS_SRC = os.environ.get("LEADLOSS_SRC")
if _ARCHIVE_SRC.exists():
    sys.path.insert(0, str(_ARCHIVE_SRC))
    print(f"[info] importing LeadLoss from bundled archive src: {_ARCHIVE_SRC}", flush=True)
elif _LEADLOSS_SRC and Path(_LEADLOSS_SRC).exists():
    sys.path.insert(0, _LEADLOSS_SRC)
    print(f"[info] importing LeadLoss from: {_LEADLOSS_SRC}", flush=True)

_UTIL_DIR = _SCRIPT_DIR.parents[0] / "_util"
if str(_UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(_UTIL_DIR))

try:
    from process.ensemble import build_ensemble_catalogue  # type: ignore
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: cannot import process.ensemble.\n"
        "This script reproduces Table S1 (parameter sensitivity) by re-running\n"
        "the ensemble peak-detection step against the bundled goodness-surface\n"
        "diagnostics. It needs the LeadLoss application source available.\n\n"
        "Two ways to make it work:\n"
        "  1. Bundle <archive>/src in this reproducibility archive, or\n"
        "  2. Set the LEADLOSS_SRC environment variable to point at the\n"
        "     <leadloss-checkout>/src directory. Example:\n"
        "       export LEADLOSS_SRC=/path/to/LeadLoss/src\n\n"
        "If you only need the published sensitivity numbers, they are already\n"
        "in data/derived/sensitivity/ as CSVs and in the Supp PDF Table S1."
    ) from exc

from benchmark_definitions import CASES_TRUE  # type: ignore

# ── Defaults (from cdc_config.py) ─────────────────────────────────────
DEFAULTS = dict(
    smooth_frac=0.01,
    f_d=0.10,
    f_p=0.10,
    f_v=0.50,
    f_w=0.10,
    w_min_nodes=3,
    support_min=0.10,
    r_min=5,
    f_r=0.25,
    per_run_prom_frac=0.06,
    per_run_min_dist=3,
    per_run_min_width=3,
    per_run_require_full_prom=False,
    height_frac=0.0,
    delta_min=0.05,
    merge_per_hump=True,
    merge_shoulders=True,
)

# ── Representative cases ──────────────────────────────────────────────
# Full sweep: all 7 cases × 3 tiers for comprehensive coverage.
# A reduced set (e.g. ["1B", "2B", "6B"]) can be used for quick checks.
CASES = [
    f"{c}{t}" for c in ["1", "2", "3", "4", "5", "6", "7"]
    for t in ["A", "B", "C"]
]

# ── Parameters to sweep ───────────────────────────────────────────────
# Already-reported gate-keeping thresholds, plus six additional shape /
# pipeline parameters not previously varied.
SWEEPS: Dict[str, Tuple[str, List[float]]] = {
    "f_p": ("Ensemble prominence (f_p)", [0.03, 0.05, 0.10, 0.15, 0.20]),
    "per_run_prom_frac": ("Per-run prominence", [0.02, 0.04, 0.06, 0.10, 0.15]),
    "support_min": ("Minimum support (f_support)", [0.05, 0.10, 0.15, 0.20, 0.30]),
    "f_v": ("Valley merge depth (f_v)", [0.20, 0.35, 0.50, 0.65, 0.80]),
    "r_min": ("Minimum supporting runs (r_min)", [3, 5, 7, 10, 15]),
    "delta_min": ("Flat-surface floor (delta_min)", [0.02, 0.05, 0.08, 0.10, 0.15]),
    "f_d": ("Peak-distance fraction (f_d)", [0.05, 0.07, 0.10, 0.13, 0.20]),
    "per_run_min_dist": ("Per-run min peak distance (nodes)", [2, 3, 4, 5, 7]),
    "per_run_min_width": ("Per-run min peak width (nodes)", [2, 3, 4, 5, 7]),
    # height_frac excluded: default is 0 (disabled), so varying it tests a
    # feature that is off by design rather than a tuning choice.
}

# ── Truth-window scoring (matches manuscript Tables 3–8 logic) ────────
HALF_MIN = 50.0
HALF_CAP = 120.0


def _truth_windows(case_id: str) -> List[Tuple[float, float, float]]:
    """Return (true_age, window_lo, window_hi) for each true event."""
    true_ages = sorted(CASES_TRUE[case_id])
    n = len(true_ages)
    windows = []
    for i, t in enumerate(true_ages):
        dists = []
        if i > 0:
            dists.append(t - true_ages[i - 1])
        if i < n - 1:
            dists.append(true_ages[i + 1] - t)
        half = min(dists) / 2.0 if dists else HALF_CAP
        half = max(HALF_MIN, min(half, HALF_CAP))
        windows.append((t, t - half, t + half))
    return windows


def _load_surfaces(case_tier: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-computed penalised goodness surfaces from NPZ bundle."""
    npz_path = _DIAG_DIR / f"{case_tier}_runs_S.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing: {npz_path}")
    d = np.load(npz_path)
    return d["age_Ma"], d["S_runs_pen"], d["optima_Ma"]


def _score(
    catalogue: List[Dict],
    case_id: str,
) -> Dict:
    """Score a catalogue against true events.

    Returns dict with:
      n_peaks, peak_ages_str, n_recovered, n_covered, recovered_events, covered_events
    """
    windows = _truth_windows(case_id)
    n_true = len(windows)

    peak_ages = []
    for row in catalogue:
        age = float(row.get("age_ma", row.get("age_Ma", 0)))
        peak_ages.append(age)

    recovered = []
    covered = []
    for true_age, wlo, whi in windows:
        # Find best peak inside truth window
        best = None
        best_dist = float("inf")
        for row in catalogue:
            age = float(row.get("age_ma", row.get("age_Ma", 0)))
            if wlo <= age <= whi:
                dist = abs(age - true_age)
                if dist < best_dist:
                    best = row
                    best_dist = dist
        if best is not None:
            recovered.append(true_age)
            ci_lo = float(best.get("ci_low", best.get("ci_lo", 0)))
            ci_hi = float(best.get("ci_high", best.get("ci_hi", 0)))
            if ci_lo <= true_age <= ci_hi:
                covered.append(true_age)

    return dict(
        n_peaks=len(catalogue),
        peak_ages_str=", ".join(f"{a:.0f}" for a in sorted(peak_ages)),
        n_recovered=len(recovered),
        n_true=n_true,
        n_covered=len(covered),
        recovered_str=", ".join(f"{a:.0f}" for a in recovered),
        covered_str=", ".join(f"{a:.0f}" for a in covered),
    )


def _run_one(
    case_tier: str,
    age_grid: np.ndarray,
    S_runs: np.ndarray,
    optima_ma: np.ndarray,
    overrides: Dict,
) -> List[Dict]:
    """Run ensemble catalogue with given parameter overrides."""
    kw = dict(DEFAULTS)
    kw.update(overrides)
    # Extract case number for naming
    case_num = case_tier[:-1]
    tier = case_tier[-1]
    return build_ensemble_catalogue(
        sample_name=f"Case{case_num}",
        tier=tier,
        age_grid=age_grid,
        goodness_runs=S_runs,
        orientation="max",
        optima_ma=optima_ma,
        **kw,
    ) or []


def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load surfaces
    surfaces = {}
    for ct in CASES:
        print(f"Loading {ct} ...", flush=True)
        surfaces[ct] = _load_surfaces(ct)

    all_rows = []

    for param_name, (label, values) in SWEEPS.items():
        print(f"\n{'='*60}")
        print(f"Sweeping: {label} ({param_name})")
        print(f"  Values: {values}")
        print(f"  Default: {DEFAULTS[param_name]}")
        print(f"{'='*60}")

        param_rows = []
        for ct in CASES:
            case_id = ct[:-1]  # "1", "2", "6"
            true_ages = CASES_TRUE[case_id]
            age_grid, S_runs, optima_ma = surfaces[ct]

            for val in values:
                overrides = {param_name: val}
                cat = _run_one(ct, age_grid, S_runs, optima_ma, overrides)
                sc = _score(cat, case_id)

                is_default = (val == DEFAULTS[param_name])
                row = dict(
                    param=param_name,
                    param_label=label,
                    value=val,
                    is_default="*" if is_default else "",
                    case=ct,
                    true_ages=", ".join(f"{a:.0f}" for a in true_ages),
                    **sc,
                )
                param_rows.append(row)
                all_rows.append(row)

                tag = " [DEFAULT]" if is_default else ""
                rec = f"{sc['n_recovered']}/{sc['n_true']}"
                cov = f"{sc['n_covered']}/{sc['n_true']}"
                print(
                    f"  {ct}  {param_name}={val:<6}  "
                    f"peaks={sc['n_peaks']}  ages=[{sc['peak_ages_str']:>20s}]  "
                    f"rec={rec}  cov={cov}{tag}"
                )

        # Write per-parameter CSV
        csv_path = _OUT_DIR / f"sensitivity_{param_name}.csv"
        _write_csv(csv_path, param_rows)

    # Write combined CSV
    combined_path = _OUT_DIR / "sensitivity_combined.csv"
    _write_csv(combined_path, all_rows)
    print(f"\nWrote combined results to {combined_path}")

    # Print summary table
    _print_summary(all_rows)

    return 0


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _print_summary(rows: List[Dict]) -> None:
    """Print a compact summary highlighting stability."""
    print("\n" + "=" * 80)
    print("SENSITIVITY SUMMARY")
    print("=" * 80)
    print(
        "Results are stable if peak ages and event recovery do not change "
        "across the tested range."
    )
    print()

    for param_name, (label, values) in SWEEPS.items():
        print(f"\n--- {label} ({param_name}) ---")
        print(f"{'Case':<6} {'Default':>8} | ", end="")
        print("  ".join(f"{v:>6}" for v in values))
        print("-" * (18 + 8 * len(values)))

        for ct in CASES:
            case_rows = [
                r for r in rows
                if r["param"] == param_name and r["case"] == ct
            ]
            default_val = DEFAULTS[param_name]

            ages_by_val = {}
            for r in case_rows:
                ages_by_val[r["value"]] = r["peak_ages_str"]

            # Check if all values give same peak ages
            unique_results = set(ages_by_val.values())
            stable = len(unique_results) == 1

            print(f"{ct:<6} {'STABLE' if stable else 'VARIES':>8} | ", end="")
            for v in values:
                tag = "*" if v == default_val else " "
                ages = ages_by_val.get(v, "?")
                # Compact: just show n_peaks
                n = len(ages.split(",")) if ages else 0
                rec = None
                for r in case_rows:
                    if r["value"] == v:
                        rec = f"{r['n_recovered']}/{r['n_true']}"
                        break
                print(f" {n}p {rec or '?':>4}{tag}", end="")
            print()

    print()


if __name__ == "__main__":
    raise SystemExit(main())
