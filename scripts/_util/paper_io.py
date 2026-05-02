from pathlib import Path
import argparse
import sys
from typing import Union


def infer_paper_dir(script_file: Union[str, Path]) -> Path:
    """
    Locate the paper/archive root containing data/ and (optionally) tables/.

    Tries two strategies in order:
      1. Walk upward looking for a folder named "2025-peak-picking" (internal
         repo layout used during development).
      2. Walk upward looking for a folder that contains a data/ subdirectory
         (archive layout shipped to Zenodo, where the root is whatever the
         downloader unpacked it to).
    """
    p = Path(script_file).resolve()
    for parent in p.parents:
        if parent.name == "2025-peak-picking":
            return parent
    for parent in p.parents:
        if (parent / "data").is_dir():
            return parent
    raise RuntimeError(
        "Could not locate the paper/archive root. Expected a parent folder "
        "either named '2025-peak-picking' or containing a 'data/' subdirectory."
    )


def add_src_to_path(script_file: Union[str, Path]) -> Path:
    """
    Ensure the repo's ./src folder is on sys.path so paper scripts can import package code.
    Falls back to a bundled archive-root src/ snapshot, then LEADLOSS_SRC env var,
    when no internal repo src/ is present.
    Returns the src_dir Path (may not exist on disk in the Zenodo archive).
    """
    import os
    paper_dir = infer_paper_dir(script_file)
    # Internal repo layout: .../LeadLoss/papers/<paper>/scripts/...
    # Walk up two more levels to look for src/ at repo root.
    candidate = paper_dir
    for _ in range(2):
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    src_dir = candidate / "src"

    if not src_dir.is_dir():
        bundled_src = paper_dir / "src"
        if bundled_src.is_dir():
            src_dir = bundled_src

    if not src_dir.is_dir():
        env_src = os.environ.get("LEADLOSS_SRC")
        if env_src:
            src_dir = Path(env_src)

    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def parse_paper_args(script_file: Union[str, Path]):
    # Add src/ if present (safe no-op if missing)
    add_src_to_path(script_file)

    paper_dir = infer_paper_dir(script_file)
    data_dir  = paper_dir / "data"
    # Archive ships manuscript-facing exports at <root>/06_Table_Exports
    # and <root>/05_Final_Manuscript_Figures.
    # If a dedicated 'outputs' folder exists (internal repo), use it; otherwise
    # fall back to the archive root so downstream scripts write to the right place.
    out_dir   = paper_dir / "outputs" if (paper_dir / "outputs").is_dir() else paper_dir

    # IMPORTANT: parse_known_args so table scripts can add their own flags
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paper-dir", dest="paper_dir", type=Path, default=paper_dir)
    parser.add_argument("--data-dir",  dest="data_dir",  type=Path, default=data_dir)
    parser.add_argument("--out-dir",   dest="out_dir",   type=Path, default=out_dir)

    args, _unknown = parser.parse_known_args()
    return args
