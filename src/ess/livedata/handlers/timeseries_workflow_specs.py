# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Lightweight timeseries workflow spec registration (no heavy dependencies)."""

from __future__ import annotations

from ..config.instrument import Instrument
from ..handlers.workflow_factory import SpecHandle


def register_timeseries_workflow_specs(
    instrument: Instrument, source_names: list[str]
) -> SpecHandle | None:
    """
    Register timeseries workflow specs (lightweight, no heavy dependencies).

    This is the first phase of two-phase registration. Call this from
    instrument specs.py modules.

    If the workflow is already registered (e.g., auto-registered in
    Instrument.__post_init__()), returns the existing handle.

    Parameters
    ----------
    instrument
        The instrument to register the workflow specs for.
    source_names
        List of log data source names (e.g., f144 attribute names) for which to
        register the workflow. If empty, returns None without registering.

    Returns
    -------
    SpecHandle for later factory attachment, or None if no timeseries sources.
    """
    if not source_names:
        return None

    from ..config.workflow_spec import WorkflowId

    workflow_id = WorkflowId(
        instrument=instrument.name,
        namespace='timeseries',
        name='timeseries_data',
        version=1,
    )
    if workflow_id in instrument.workflow_factory._workflow_specs:
        from ..handlers.workflow_factory import SpecHandle

        return SpecHandle(workflow_id=workflow_id, _factory=instrument.workflow_factory)

    return instrument.register_spec(
        namespace='timeseries',
        name='timeseries_data',
        version=1,
        title="Timeseries data",
        description="Accumulated log data as timeseries.",
        source_names=source_names,
    )
