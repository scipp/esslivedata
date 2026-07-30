# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Lightweight timeseries workflow spec registration (no heavy dependencies)."""

from __future__ import annotations

from typing import ClassVar

import pydantic
import scipp as sc

from ..config.instrument import Instrument
from ..config.workflow_spec import (
    TIMESERIES,
    OutputView,
    SeriesOutput,
    WorkflowOutputsBase,
)
from .workflow_factory import SpecHandle


class TimeseriesOutputs(WorkflowOutputsBase):
    """Outputs for the timeseries workflow.

    The template defines a 0-D DataArray with a scalar ``time`` coordinate.
    Conceptually, each timeseries value is a timestamped scalar; in practice,
    ``TimeseriesStreamProcessor.finalize()`` returns batches (1-D along ``time``)
    for efficiency. Either way the ``time`` coordinate is real data — the
    per-sample wall-clock timestamps — which is what
    :attr:`Temporality.series` declares.
    """

    output_views: ClassVar[tuple[OutputView, ...]] = (
        OutputView(
            name='delta',
            title='Timeseries',
            fields=('delta',),
            description='Timestamped device values; plots accumulate the history.',
        ),
    )

    delta: SeriesOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.scalar(0.0),
            coords={'time': sc.scalar(0, unit='ns')},
        ),
        title='Update',
        description='New value updates since the last frame.',
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
        group=TIMESERIES,
        name='timeseries_data',
        version=1,
        title="Readings",
        description=(
            "Time-stamped value updates from a single device or sensor. "
            "Plot the latest value, or accumulate updates into a timeseries."
        ),
        source_names=source_names,
        outputs=TimeseriesOutputs,
        reset_on_run_transition=False,
        # Reset would clear only the job's publication bookkeeping while the
        # shared ToNXlog preprocessor keeps its cumulative buffer, so the next
        # finalize re-emits the full history as a delta and every downstream
        # buffer duplicates it. Restarting the workflow already provides
        # correct clear-and-refill semantics via the generation flip.
        supports_reset=False,
    )
