#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CaseBundle:
    sample_id: str
    age_ma: np.ndarray
    s_raw_mean: np.ndarray
    s_pen_mean: np.ndarray
    legacy_age_ma: float
    legacy_ci_low_ma: float
    legacy_ci_high_ma: float
    ensemble_rows: pd.DataFrame


def default_paper_dir(script_path: str) -> Path:
    p = Path(script_path).resolve()
    cand = p.parents[2]
    if (cand / "data").is_dir() and (cand / "scripts").is_dir():
        return cand
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "scripts").is_dir():
            return parent
    return cand


def tie_aware_argmax(values: np.ndarray) -> int:
    values = np.asarray(values, float)
    max_value = np.nanmax(values)
    idx = np.flatnonzero(values == max_value)
    if idx.size == 0:
        return int(np.nanargmax(values))
    return int((idx.min() + idx.max()) // 2)


def standardise_runs_matrix(surfaces: np.ndarray, n_ages: int) -> np.ndarray:
    surfaces = np.asarray(surfaces, float)
    if surfaces.ndim != 2:
        raise ValueError(f"Expected 2D surfaces, got shape {surfaces.shape}")
    if surfaces.shape[1] == n_ages:
        return surfaces
    if surfaces.shape[0] == n_ages:
        return surfaces.T
    raise ValueError(f"Cannot align surfaces {surfaces.shape} with age grid of length {n_ages}")


def load_case_bundle(bundle_dir: Path, sample_id: str) -> CaseBundle:
    bundle_dir = bundle_dir.expanduser().resolve()
    runs_npz = bundle_dir / f"{sample_id}_runs_S.npz"
    catalogue_csv = bundle_dir / "ensemble_catalogue_rows.csv"

    if not runs_npz.exists():
        raise FileNotFoundError(f"Missing runs bundle: {runs_npz}")
    if not catalogue_csv.exists():
        raise FileNotFoundError(f"Missing ensemble rows CSV: {catalogue_csv}")

    z = np.load(runs_npz, allow_pickle=True)
    age_ma = np.asarray(z["age_Ma"], float) if "age_Ma" in z.files else np.asarray(z["age_ma"], float)
    s_raw = standardise_runs_matrix(z["S_runs_raw"], n_ages=age_ma.size)
    s_pen = standardise_runs_matrix(z["S_runs_pen"], n_ages=age_ma.size)

    s_raw_mean = np.nanmean(s_raw, axis=0)
    s_pen_mean = np.nanmean(s_pen, axis=0)
    legacy_age_ma = float(age_ma[tie_aware_argmax(s_pen_mean)])

    if "optima_Ma" in z.files:
        optima = np.asarray(z["optima_Ma"], float)
        optima = optima[np.isfinite(optima)]
    else:
        optima = []
        for row in s_pen:
            optima.append(float(age_ma[tie_aware_argmax(row)]))
        optima = np.asarray(optima, float)
    legacy_ci_low_ma, legacy_ci_high_ma = np.quantile(optima, [0.025, 0.975])

    ensemble_rows = pd.read_csv(catalogue_csv)
    ensemble_rows.columns = ensemble_rows.columns.str.strip()

    return CaseBundle(
        sample_id=sample_id,
        age_ma=age_ma,
        s_raw_mean=s_raw_mean,
        s_pen_mean=s_pen_mean,
        legacy_age_ma=legacy_age_ma,
        legacy_ci_low_ma=float(legacy_ci_low_ma),
        legacy_ci_high_ma=float(legacy_ci_high_ma),
        ensemble_rows=ensemble_rows,
    )


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, outdir: Path, stub: str, formats: list[str]) -> list[Path]:
    ensure_output_dir(outdir)
    written = []
    for ext in formats:
        ext = ext.strip().lstrip(".")
        if not ext:
            continue
        outpath = outdir / f"{stub}.{ext}"
        fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02)
        written.append(outpath)
    return written
