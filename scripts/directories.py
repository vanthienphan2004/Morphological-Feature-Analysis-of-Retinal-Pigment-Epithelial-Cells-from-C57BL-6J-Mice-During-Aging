"""
Directory utilities module for RPE image data processing.

This module provides utility functions for managing output directories.
"""

import json
from pathlib import Path
from typing import Tuple, Optional


def resolve_output_dirs(base: Path, config_path: Optional[Path] = None) -> Tuple[Path, Path, Path, Path]:
    """
    Resolve and create output directories (out_root, reports, models, plots).

    Args:
        base: Project root (usually Path(__file__).resolve().parents[0]).
        config_path: Optional path to config.json (if None, will check base/config.json).

    Returns:
        Tuple of (out_root, reports_dir, models_dir, plots_dir).
    """
    # Determine project root by searching upwards for config.json; fall back to provided base
    project_root: Optional[Path] = None
    for current_path in (base, *base.parents):
        if (current_path / 'config.json').exists():
            project_root = current_path
            break
    if project_root is None:
        project_root = base

    config_file_path = config_path or (project_root / 'config.json')
    if config_file_path.exists():
        try:
            with open(config_file_path, 'r') as file_handle:
                config_data = json.load(file_handle)
            output_root = Path(config_data.get('paths', {}).get('output_directory', project_root / 'analysis_results'))
        except Exception:
            output_root = project_root / 'analysis_results'
    else:
        output_root = project_root / 'analysis_results'

    reports_dir = output_root / 'reports'
    models_dir = output_root / 'models'
    plots_dir = output_root / 'plots'
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return output_root, reports_dir, models_dir, plots_dir

