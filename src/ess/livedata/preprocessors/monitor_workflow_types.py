# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Sciline types for monitor workflow using StreamProcessor."""

from typing import NewType

import sciline
import scipp as sc

from .accumulation_mode import AccumulationMode

# Input type: Use standard NeXusData[NXmonitor, SampleRun] from ess.reduce
# The specific monitor is determined by NeXusName[NXmonitor] = source_name
# Same pattern as detector workflows use NeXusName[NXdetector] = source_name

# Intermediate type (computed once, routed to both accumulators)
MonitorHistogram = NewType('MonitorHistogram', sc.DataArray)


class AccumulatedMonitorHistogram(
    sciline.Scope[AccumulationMode, sc.DataArray],
    sc.DataArray,  # type: ignore[misc]
):
    """Monitor histogram parametrized by accumulation mode.

    - AccumulatedMonitorHistogram[Cumulative]: Accumulated forever
      (EternalAccumulator)
    - AccumulatedMonitorHistogram[Current]: Current window only (clears
      after finalize)
    """


class MonitorCountsTotal(
    sciline.Scope[AccumulationMode, sc.DataArray],
    sc.DataArray,  # type: ignore[misc]
):
    """Total monitor counts as 0D scalar, parametrized by accumulation mode."""


class MonitorCountsInRange(
    sciline.Scope[AccumulationMode, sc.DataArray],
    sc.DataArray,  # type: ignore[misc]
):
    """Monitor counts within configured range as 0D scalar.

    Parametrized by accumulation mode.
    """


# Configuration types (mode-agnostic names)
HistogramEdges = NewType('HistogramEdges', sc.Variable)
HistogramRangeLow = NewType('HistogramRangeLow', sc.Variable)
HistogramRangeHigh = NewType('HistogramRangeHigh', sc.Variable)
