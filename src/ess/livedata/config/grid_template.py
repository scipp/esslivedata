# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Grid specification data model and raw template loading."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

import structlog
import yaml

if TYPE_CHECKING:
    from ess.livedata.dashboard.plot_orchestrator import PlotCell

logger = structlog.get_logger()


@dataclass(frozen=True)
class GridSpec:
    """
    Grid configuration without runtime state.

    Used for pre-defined templates and for persisting configurations across sessions.
    """

    name: str
    title: str
    description: str
    nrows: int
    ncols: int
    cells: tuple[PlotCell, ...]

    @property
    def min_rows(self) -> int:
        """Minimum rows required to fit all cells."""
        if not self.cells:
            return self.nrows
        return max(cell.geometry.row + cell.geometry.row_span for cell in self.cells)

    @property
    def min_cols(self) -> int:
        """Minimum columns required to fit all cells."""
        if not self.cells:
            return self.ncols
        return max(cell.geometry.col + cell.geometry.col_span for cell in self.cells)


def load_raw_grid_templates(instrument: str) -> list[dict[str, Any]]:
    """
    Load raw grid template YAML files for an instrument.

    Templates are loaded from the instrument's ``grid_templates/`` subdirectory
    without validation.

    Parameters
    ----------
    instrument
        Name of the instrument (e.g., 'dummy', 'dream').

    Returns
    -------
    :
        List of raw template dicts. Empty if no templates exist.
    """
    templates: list[dict[str, Any]] = []

    try:
        package = f'ess.livedata.config.instruments.{instrument}'
        instrument_files = resources.files(package)
        templates_dir = instrument_files.joinpath('grid_templates')

        if not templates_dir.is_dir():
            logger.debug('No grid_templates directory for instrument %s', instrument)
            return templates

        for item in templates_dir.iterdir():
            if item.is_file() and item.name.endswith('.yaml'):
                raw = _load_template_file(item)
                if raw is not None:
                    templates.append(raw)

    except ModuleNotFoundError:
        logger.warning('Instrument package not found: %s', instrument)
    except Exception:
        logger.exception('Failed to load grid templates for instrument %s', instrument)

    logger.info(
        'Loaded %d raw grid template(s) for instrument %s',
        len(templates),
        instrument,
    )
    return templates


def _load_template_file(file_path: resources.abc.Traversable) -> dict[str, Any] | None:
    """Load a single YAML template file, returning None on failure."""
    try:
        with file_path.open() as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            logger.warning('Template file %s does not contain a dict', file_path.name)
            return None

        return config

    except Exception:
        logger.exception('Failed to load template file %s', file_path.name)
        return None
