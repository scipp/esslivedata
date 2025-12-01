# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Grid template loading utilities.

Grid templates are pre-defined plot grid configurations shipped with the package
that users can select when creating a new grid. Templates are loaded once at
dashboard startup and shared across all browser sessions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridTemplate:
    """
    A grid template loaded from package resources.

    Templates are immutable blueprints for creating plot grids. When a user
    selects a template, a copy is made which they can then modify.

    Parameters
    ----------
    name
        Display name for the template (shown in UI dropdown).
    config
        Raw configuration dict matching the grid serialization format.
        Contains: title, nrows, ncols, cells.
    """

    name: str
    config: dict[str, Any]

    @property
    def title(self) -> str:
        """Grid title from the template config."""
        return self.config.get('title', self.name)

    @property
    def nrows(self) -> int:
        """Number of rows in the template grid."""
        return self.config.get('nrows', 3)

    @property
    def ncols(self) -> int:
        """Number of columns in the template grid."""
        return self.config.get('ncols', 3)

    @property
    def cells(self) -> list[dict[str, Any]]:
        """Cell configurations from the template."""
        return self.config.get('cells', [])

    @property
    def min_rows(self) -> int:
        """Minimum rows required to fit all template cells."""
        if not self.cells:
            return self.nrows
        return max(
            cell['geometry']['row'] + cell['geometry']['row_span']
            for cell in self.cells
        )

    @property
    def min_cols(self) -> int:
        """Minimum columns required to fit all template cells."""
        if not self.cells:
            return self.ncols
        return max(
            cell['geometry']['col'] + cell['geometry']['col_span']
            for cell in self.cells
        )


def load_grid_templates(instrument: str) -> list[GridTemplate]:
    """
    Load grid templates for an instrument from package resources.

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
        List of grid templates. Empty if the instrument has no templates
        or the grid_templates directory doesn't exist.
    """
    templates: list[GridTemplate] = []

    try:
        package = f'ess.livedata.config.instruments.{instrument}'
        instrument_files = resources.files(package)
        templates_dir = instrument_files.joinpath('grid_templates')

        if not templates_dir.is_dir():
            logger.debug('No grid_templates directory for instrument %s', instrument)
            return templates

        for item in templates_dir.iterdir():
            if item.is_file() and item.name.endswith('.yaml'):
                template = _load_template_file(item)
                if template is not None:
                    templates.append(template)

    except ModuleNotFoundError:
        logger.warning('Instrument package not found: %s', instrument)
    except Exception:
        logger.exception('Failed to load grid templates for instrument %s', instrument)

    logger.info(
        'Loaded %d grid template(s) for instrument %s',
        len(templates),
        instrument,
    )
    return templates


def _load_template_file(file_path: resources.abc.Traversable) -> GridTemplate | None:
    """
    Load a single template file.

    Parameters
    ----------
    file_path
        Path to the YAML template file.

    Returns
    -------
    :
        GridTemplate if loading succeeded, None otherwise.
    """
    try:
        with file_path.open() as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            logger.warning('Template file %s does not contain a dict', file_path.name)
            return None

        # Use title as display name, falling back to filename
        name = config.get('title', file_path.name.removesuffix('.yaml'))

        return GridTemplate(name=name, config=config)

    except Exception:
        logger.exception('Failed to load template file %s', file_path.name)
        return None
