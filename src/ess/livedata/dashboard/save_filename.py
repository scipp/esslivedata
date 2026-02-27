# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Descriptive filenames for the Bokeh SaveTool on dashboard plots."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec

from .plot_orchestrator import PlotCell
from .widgets.plot_widgets import get_workflow_display_info

_MAX_SOURCES_IN_FILENAME = 3


def _sanitize_for_filename(text: str) -> str:
    """Replace characters that are problematic in filenames."""
    # Replace runs of whitespace or path-unsafe characters with a single hyphen
    return re.sub(r'[\s/\\:*?"<>|()]+', '-', text).strip('-')


def build_save_filename(
    instrument: str,
    source_titles: list[str],
    output_titles: list[str],
) -> str:
    """
    Build a descriptive filename for the Bokeh SaveTool.

    Produces a filename like "DREAM_I-Q_Mantle-SANS" from instrument,
    output titles, and source titles. Source titles are included only
    when there are few enough to keep the name readable. Timestamps are
    omitted because the file-system creation time serves the same
    purpose.

    Parameters
    ----------
    instrument:
        Instrument name (will be uppercased).
    source_titles:
        Human-readable source/detector titles.
    output_titles:
        Human-readable output titles.

    Returns
    -------
    :
        Filename string (without extension).
    """
    parts = [instrument.upper()]
    if output_titles:
        parts.append('-'.join(_sanitize_for_filename(t) for t in sorted(output_titles)))
    if source_titles and len(source_titles) <= _MAX_SOURCES_IN_FILENAME:
        parts.append('-'.join(_sanitize_for_filename(t) for t in sorted(source_titles)))
    return '_'.join(parts)


def make_save_filename_hook(
    filename: str,
) -> Callable[[Any, Any], None]:
    """
    Create a HoloViews hook that sets the SaveTool filename on a Bokeh figure.

    Parameters
    ----------
    filename:
        Filename (without extension) to set on the SaveTool.

    Returns
    -------
    :
        A hook function compatible with ``hv.Element.opts(hooks=[...])``.
    """

    def hook(plot: Any, element: Any) -> None:
        del element
        from bokeh.models.tools import SaveTool

        for tool in plot.state.toolbar.tools:
            if isinstance(tool, SaveTool):
                tool.filename = filename

    return hook


def build_save_filename_from_cell(
    cell: PlotCell,
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    get_source_title: Callable[[str], str],
) -> str | None:
    """Build a descriptive SaveTool filename from a plot cell's layer configs.

    Resolves human-readable titles for outputs and sources so the
    filename uses e.g. "I-Q" and "Mantle" rather than raw identifiers.
    Returns None if no non-static layers exist.
    """
    source_titles: list[str] = []
    output_titles: list[str] = []
    instrument: str | None = None
    for layer in cell.layers:
        config = layer.config
        if config.is_static():
            continue
        if instrument is None:
            instrument = config.workflow_id.instrument
        source_titles.extend(get_source_title(s) for s in config.source_names)
        _, output_title = get_workflow_display_info(
            workflow_registry, config.workflow_id, config.output_name
        )
        output_titles.append(output_title)
    if instrument is None:
        return None
    return build_save_filename(instrument, source_titles, output_titles)
