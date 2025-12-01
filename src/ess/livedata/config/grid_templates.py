# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Grid specification data model and raw template loading.

GridSpec is the universal representation of a grid configuration without runtime
state. It is used for:

1. **Templates**: Pre-defined configurations shipped with the package that users
   can select when creating a new grid.
2. **Persistence**: Restoring grid configurations from the config store on reload.

This module provides:

- :py:class:`GridSpec`: The data model for grid configurations
- :py:func:`load_raw_grid_templates`: Load raw YAML templates from package resources

Parsing raw templates into validated GridSpec objects is handled by
:py:class:`~ess.livedata.dashboard.plot_orchestrator.PlotOrchestrator`, which has
access to the plotter registry needed for validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from ess.livedata.dashboard.plot_orchestrator import PlotCell

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridSpec:
    """
    A validated grid specification ready for use.

    GridSpec is the universal representation of a grid configuration without
    runtime state (no CellIds). It serves two purposes:

    1. **Templates**: Pre-defined configurations loaded from YAML files that
       users can select when creating a new grid.
    2. **Persistence**: Serialized grid configurations restored on reload.

    The ``cells`` field contains validated PlotCell objects. GridSpecs are
    created by parsing raw data via
    :py:meth:`~ess.livedata.dashboard.plot_orchestrator.PlotOrchestrator._parse_grid_specs`.

    When applied, a GridSpec is converted to runtime state by calling
    ``add_grid()`` followed by ``add_plot()`` for each cell.
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
    Load raw grid template data for an instrument from package resources.

    This function loads YAML files without validation. The raw data is parsed
    into validated :py:class:`GridSpec` objects by
    :py:class:`~ess.livedata.dashboard.plot_orchestrator.PlotOrchestrator`.

    Templates are YAML files in the instrument's ``grid_templates/`` subdirectory.
    Each file should contain a grid configuration matching the serialization format:

    .. code-block:: yaml

        title: Monitor Overview
        nrows: 2
        ncols: 3
        cells:
          - geometry: {row: 0, col: 0, row_span: 1, col_span: 2}
            config:
              workflow_id: dummy/monitor_data/monitor_histogram/1
              output_name: histogram
              source_names: [monitor1]
              plot_name: lines
              params: {}

    Parameters
    ----------
    instrument
        Name of the instrument (e.g., 'dummy', 'dream').

    Returns
    -------
    :
        List of raw template dicts. Empty if the instrument has no templates
        or the grid_templates directory doesn't exist.
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
    """
    Load a single template file as raw dict.

    Parameters
    ----------
    file_path
        Path to the YAML template file.

    Returns
    -------
    :
        Raw template dict if loading succeeded, None otherwise.
    """
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
