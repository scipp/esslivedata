# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Sciline types for monitor workflow using StreamProcessor."""

from typing import NewType

import scipp as sc

# Input type: Use standard NeXusData[NXmonitor, SampleRun] from ess.reduce
# The specific monitor is determined by NeXusName[NXmonitor] = source_name
# Same pattern as detector workflows use NeXusName[NXdetector] = source_name

# Intermediate type (computed once, routed to two accumulators)
MonitorHistogram = NewType('MonitorHistogram', sc.DataArray)

# Output types (identity transforms for routing to different accumulators)
CumulativeMonitorHistogram = NewType('CumulativeMonitorHistogram', sc.DataArray)
WindowMonitorHistogram = NewType('WindowMonitorHistogram', sc.DataArray)

# Ratemeter outputs (derived from window histogram)
MonitorCountsTotal = NewType('MonitorCountsTotal', sc.DataArray)
MonitorCountsInRange = NewType('MonitorCountsInRange', sc.DataArray)

# Configuration types (mode-agnostic names)
HistogramEdges = NewType('HistogramEdges', sc.Variable)
HistogramRangeLow = NewType('HistogramRangeLow', sc.Variable)
HistogramRangeHigh = NewType('HistogramRangeHigh', sc.Variable)
