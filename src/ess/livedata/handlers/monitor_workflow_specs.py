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


class MonitorRatemeterParams(pydantic.BaseModel):
    """Parameters for monitor ratemeter workflow."""

    toa_range: parameter_models.TOARange = pydantic.Field(
        title="Time of Arrival Range",
        description="Time of arrival range to include.",
        default=parameter_models.TOARange(),
    )


class MonitorRatemeterOutputs(WorkflowOutputsBase):
    """Outputs for monitor ratemeter workflow."""

    monitor_counts: sc.DataArray = pydantic.Field(
        title="Monitor Counts",
        description="Monitor counts within the specified TOA range.",
    )


class MonitorHistogramOutputs(WorkflowOutputsBase):
    """Outputs for the monitor histogram workflow."""

    cumulative: sc.DataArray = pydantic.Field(
        title='Cumulative Counts',
        description='Time-integrated monitor counts accumulated over all time.',
    )
    current: sc.DataArray = pydantic.Field(
        title='Current Counts',
        description='Monitor counts for the current time window since last update.',
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


def register_monitor_ratemeter_spec(
    instrument: Instrument, source_names: list[str]
) -> SpecHandle | None:
    """
    Register monitor ratemeter workflow spec.

    Parameters
    ----------
    instrument
        The instrument to register the workflow spec for.
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
        name='monitor_ratemeter',
        version=1,
        title='Monitor Ratemeter',
        description='Monitor counts within a specified time-of-arrival range.',
        source_names=source_names,
        params=MonitorRatemeterParams,
        outputs=MonitorRatemeterOutputs,
    )
