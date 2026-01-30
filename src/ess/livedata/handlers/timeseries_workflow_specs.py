# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Lightweight timeseries workflow spec registration (no heavy dependencies)."""

from __future__ import annotations

import pydantic
import scipp as sc

from ..config.instrument import Instrument
from ..config.workflow_spec import WorkflowOutputsBase
from ..handlers.workflow_factory import SpecHandle


class TimeseriesOutputs(WorkflowOutputsBase):
    """Outputs for the timeseries workflow."""

    delta: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.scalar(0.0),
            coords={'time': sc.scalar(0, unit='ns')},
        ),
        title='Delta',
        description='New timeseries data since last update.',
    )


def register_timeseries_workflow_specs(
    instrument: Instrument, source_names: list[str]
) -> SpecHandle | None:
    """
    Register timeseries workflow specs (lightweight, no heavy dependencies).

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

    return instrument.register_spec(
        namespace='timeseries',
        name='timeseries_data',
        version=1,
        title="Timeseries data",
        description="Accumulated log data as timeseries.",
        source_names=source_names,
        outputs=TimeseriesOutputs,
    )
