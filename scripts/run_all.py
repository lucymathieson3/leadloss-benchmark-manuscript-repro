#!/usr/bin/env python3
"""run_all.py — One-command reproduction runner for the Zenodo archive.

Layout assumed:

    <archive>/
      data/inputs/, data/derived/
      scripts/figures/, scripts/tables/, scripts/_util/
      05_Final_Manuscript_Figures/, 06_Table_Exports/, outputs/

Steps:
  1. Optionally clean previously-generated outputs in outputs/figures.
  2. Extract KS diagnostics tarball into data/derived/ks_diagnostics/ if needed.
  3. Run all table scripts (the benchmark scripts write regenerated outputs into outputs/tables/).
  4. Run all figure scripts (write PDF/SVG/PNG into outputs/figures/).

Run from anywhere:

    python scripts/run_all.py            # regenerate everything
    python scripts/run_all.py --clean    # also wipe previous figure outputs
    python scripts/run_all.py --dry-run  # print commands only

Figure scripts run headlessly via MPLBACKEND=Agg.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def archive_root() -> Path:
    # <archive>/scripts/run_all.py -> archive root is parents[1]
    return Path(__file__).resolve().parents[1]


def run(cmd, *, cwd: Path, env: dict, dry_run: bool = False) -> None:
    cmd_str = [str(c) for c in cmd]
    print(">>>", " ".join(cmd_str))
    if dry_run:
        return
    subprocess.run(cmd_str, cwd=str(cwd), env=env, check=True)


def can_run_table_s1(root: Path) -> bool:
    """Table S1 rerun needs LeadLoss application source or an installed package."""
    bundled_src = root / "src"
    if bundled_src.is_dir():
        return True

    env_src = os.environ.get("LEADLOSS_SRC", "").strip()
    if env_src and Path(env_src).expanduser().is_dir():
        return True

    try:
        return importlib.util.find_spec("process.ensemble") is not None
    except ModuleNotFoundError:
        return False


def extract_ks_bundle(root: Path, *, dry_run: bool = False) -> None:
    tar_path = root / "data" / "derived" / "ks_diagnostics_npz.tar.gz"
    dest_dir = root / "data" / "derived"
    ks_dir = dest_dir / "ks_diagnostics"

    if ks_dir.exists() and any(ks_dir.rglob("*.npz")):
        print(f"[ks] ok: {ks_dir} already populated")
        return

    if not tar_path.exists():
        print(f"[ks] no tarball at {tar_path} (skipping)")
        return

    print(f"[ks] extracting {tar_path} -> {dest_dir}")
    if dry_run:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=dest_dir)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run all tables + figures for the manuscript Zenodo archive."
    )
    ap.add_argument("--clean", action="store_true",
                    help="Delete outputs/figures contents before running.")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Skip extracting the KS diagnostics tarball.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands, do not execute.")
    ap.add_argument("--upgrade-sample-id", default="2A",
                    help="Sample ID for Fig 9 (CDC upgrade diagnostic). Default: 2A.")
    args = ap.parse_args()

    root = archive_root()
    fig_dir = root / "outputs" / "figures"
    tab_dir = root / "outputs" / "tables"
    ks_dir = root / "data" / "derived" / "ks_diagnostics"
    dd_dir = root / "data" / "derived" / "reimink_discordance_dating"

    if args.clean and fig_dir.exists():
        print(f"[clean] wiping {fig_dir}")
        if not args.dry_run:
            for child in fig_dir.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)

    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONHASHSEED", "0")

    if not args.skip_extract:
        extract_ks_bundle(root, dry_run=args.dry_run)

    py = sys.executable
    fig = root / "scripts" / "figures"
    tab = root / "scripts" / "tables"

    # Tables (run first; CSVs feed downstream sanity checks).
    table_cmds = [
        [py, tab / "tables01_02_benchmark_definitions.py"],
        [py, tab / "tables03_to_07_benchmark_results.py"],
        [py, tab / "table08_runtime_comparison.py"],
    ]
    if can_run_table_s1(root):
        table_cmds.append([py, tab / "tableS1_sensitivity.py"])
    else:
        print("[tableS1] skipping rerun: LeadLoss application source not bundled; archived Table S1 outputs remain available")

    # Figures. Fig 1 (workflow flowchart) is hand-edited, no script.
    figure_cmds = [
        [py, fig / "fig02_synthetic_cases1to4.py"],
        [py, fig / "fig03_synthetic_cases5to7.py",
         "--save-fig", "--fig-dir", fig_dir, "--formats", "svg,png,pdf", "--overwrite"],
        [py, fig / "fig04_fig06_cdc_goodness_grids.py",
         "--ks-dir", ks_dir, "--no-show", "--fig-dir", fig_dir],
        [py, fig / "fig05_fig07_dd_likelihood_grids.py",
         "--dd-dir", dd_dir, "--no-show", "--fig-dir", fig_dir],
        [py, fig / "fig08_case8_fan_to_zero.py"],
        [py, fig / "fig09_cdc_upgrade.py",
         "--sample-id", args.upgrade_sample_id, "--no-show", "--fig-dir", fig_dir],
        [py, fig / "fig10_case_study_193435.py", "--outdir", fig_dir],
        [py, fig / "figS1_concordia_193435.py", "--outdir", fig_dir, "--stub", "figS1_concordia_193435"],
    ]

    for cmd in table_cmds:
        run(cmd, cwd=root, env=env, dry_run=args.dry_run)
    for cmd in figure_cmds:
        run(cmd, cwd=root, env=env, dry_run=args.dry_run)

    print("[run_all] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
