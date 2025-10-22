# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Lightweight monitor workflow spec registration (no heavy dependencies)."""

from __future__ import annotations

import pydantic

from .. import parameter_models
from ..config.instrument import Instrument
from ..handlers.workflow_factory import SpecHandle


class MonitorDataParams(pydantic.BaseModel):
    """Parameters for monitor histogram workflow."""

    toa_edges: parameter_models.TOAEdges = pydantic.Field(
        title="Time of Arrival Edges",
        description="Time of arrival edges for histogramming.",
        default=parameter_models.TOAEdges(
            start=0.0,
            stop=1000.0 / 14,
            num_bins=100,
            unit=parameter_models.TimeUnit.MS,
        ),
    )


def register_monitor_workflow_specs(
    instrument: Instrument, source_names: list[str]
) -> SpecHandle | None:
    """
    Register monitor workflow specs (lightweight, no heavy dependencies).

    This is the first phase of two-phase registration. Call this from
    instrument specs.py modules.

    If the workflow is already registered (e.g., auto-registered in
    Instrument.__post_init__()), returns the existing handle.

    Parameters
    ----------
    instrument
        The instrument to register the workflow specs for.
    source_names
        List of monitor names (source names) for which to register the workflow.
        If empty, returns None without registering.

    Returns
    -------
    SpecHandle for later factory attachment, or None if no monitors.
    """
    if not source_names:
        return None

    from ..config.workflow_spec import WorkflowId

    workflow_id = WorkflowId(
        instrument=instrument.name,
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )
    if workflow_id in instrument.workflow_factory._workflow_specs:
        return SpecHandle(workflow_id=workflow_id, _factory=instrument.workflow_factory)

    return instrument.register_spec(
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
        title="Beam monitor data",
        description="Histogrammed and time-integrated beam monitor data. The monitor "
        "is histogrammed or rebinned into specified time-of-arrival (TOA) bins.",
        source_names=source_names,
        params=MonitorDataParams,
    )
