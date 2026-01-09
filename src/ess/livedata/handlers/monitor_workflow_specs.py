# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Lightweight monitor workflow spec registration (no heavy dependencies)."""

from __future__ import annotations

import pydantic
import scipp as sc

from .. import parameter_models
from ..config.instrument import Instrument
from ..config.workflow_spec import WorkflowOutputsBase
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
    toa_range: parameter_models.TOARange = pydantic.Field(
        title="Time of Arrival Range",
        description="Time of arrival range for ratemeter output.",
        default=parameter_models.TOARange(),
    )


class MonitorHistogramOutputs(WorkflowOutputsBase):
    """Outputs for the monitor histogram workflow."""

    cumulative: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['time_of_arrival'], shape=[0], unit='counts'),
            coords={'time_of_arrival': sc.arange('time_of_arrival', 0, unit='ms')},
        ),
        title='Cumulative Counts',
        description='Time-integrated monitor counts accumulated over all time.',
    )
    current: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['time_of_arrival'], shape=[0], unit='counts'),
            coords={
                'time_of_arrival': sc.arange('time_of_arrival', 0, unit='ms'),
                'time': sc.scalar(0, unit='ns'),
            },
        ),
        title='Current Counts',
        description='Monitor counts for the current time window since last update.',
    )
    counts_total: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.scalar(0, unit='counts'),
            coords={'time': sc.scalar(0, unit='ns')},
        ),
        title='Total counts',
        description='Total monitor counts in the current time window.',
    )
    counts_in_toa_range: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.scalar(0, unit='counts'),
            coords={'time': sc.scalar(0, unit='ns')},
        ),
        title='Counts in TOA Range',
        description='Number of monitor events within the configured TOA range filter.',
    )


def register_monitor_workflow_specs(
    instrument: Instrument, source_names: list[str]
) -> SpecHandle | None:
    """
    Register monitor workflow specs (lightweight, no heavy dependencies).

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

    return instrument.register_spec(
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
        title="Beam monitor data",
        description="Histogrammed and time-integrated beam monitor data. The monitor "
        "is histogrammed or rebinned into specified time-of-arrival (TOA) bins.",
        source_names=source_names,
        params=MonitorDataParams,
        outputs=MonitorHistogramOutputs,
    )


def create_monitor_workflow_factory(source_name: str, params: MonitorDataParams):
    """
    Factory function for monitor workflow from MonitorDataParams.

    This is a wrapper around create_monitor_workflow that unpacks the params.
    Defined here so the params type hint can be properly resolved by the
    workflow factory registration system.
    """
    from .monitor_workflow import create_monitor_workflow

    return create_monitor_workflow(
        source_name=source_name,
        edges=params.toa_edges.get_edges(),
        toa_range=params.toa_range.range_ns if params.toa_range.enabled else None,
    )
