# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Grid specification loading and parsing utilities.

GridSpec is the universal representation of a grid configuration without runtime
state. It is used for:

1. **Templates**: Pre-defined configurations shipped with the package that users
   can select when creating a new grid.
2. **Persistence**: Restoring grid configurations from the config store on reload.

Templates are loaded as raw YAML at startup, then parsed into validated GridSpec
objects when the dashboard services are initialized.

Usage::

    # At app startup (before services exist)
    raw_templates = load_raw_grid_templates(instrument)

    # After PlotOrchestrator is created
    specs = parse_grid_specs(raw_templates, plot_orchestrator)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from ess.livedata.dashboard.plot_orchestrator import PlotCell, PlotOrchestrator

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
    created by parsing raw data via :py:func:`parse_grid_specs`.

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

    This function loads YAML files without validation. Use
    :py:func:`parse_grid_specs` to convert the raw data into validated
    :py:class:`GridSpec` objects.

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


def parse_grid_specs(
    raw_specs: list[dict[str, Any]],
    orchestrator: PlotOrchestrator,
) -> list[GridSpec]:
    """
    Parse raw grid dicts into validated GridSpec objects.

    This function validates cells using the orchestrator's plotter registry.
    Cells with unknown plotters are skipped (logged as warnings).

    Parameters
    ----------
    raw_specs
        List of raw grid dicts from :py:func:`load_raw_grid_templates` or
        from persisted configurations.
    orchestrator
        PlotOrchestrator used to validate and parse cell configurations.

    Returns
    -------
    :
        List of validated GridSpec objects.
    """
    specs: list[GridSpec] = []

    for raw in raw_specs:
        spec = _parse_single_spec(raw, orchestrator)
        if spec is not None:
            specs.append(spec)

    logger.info('Parsed %d grid spec(s)', len(specs))
    return specs


def _parse_single_spec(
    raw: dict[str, Any],
    orchestrator: PlotOrchestrator,
) -> GridSpec | None:
    """
    Parse a single raw dict into a GridSpec.

    Parameters
    ----------
    raw
        Raw grid dict.
    orchestrator
        PlotOrchestrator used to validate cell configurations.

    Returns
    -------
    :
        GridSpec if parsing succeeded, None otherwise.
    """
    try:
        # Use title as display name, falling back to 'Untitled'
        name = raw.get('title', 'Untitled')

        # Parse cells using orchestrator's validation
        raw_cells = raw.get('cells', [])
        cells = []
        for cell_data in raw_cells:
            parsed = orchestrator.parse_raw_cell(cell_data)
            if parsed is not None:
                cells.append(parsed)

        return GridSpec(
            name=name,
            title=raw.get('title', name),
            description=raw.get('description', ''),
            nrows=raw.get('nrows', 3),
            ncols=raw.get('ncols', 3),
            cells=tuple(cells),
        )

    except Exception:
        logger.exception('Failed to parse grid spec: %s', raw.get('title', 'unknown'))
        return None
