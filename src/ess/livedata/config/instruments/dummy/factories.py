# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Dummy instrument factory implementations (heavy).

This module contains factory implementations with heavy dependencies.
Only imported by backend services.
"""

from typing import NewType

import sciline
import scipp as sc

from ess.livedata.config import instrument_registry
from ess.livedata.handlers.detector_data_handler import (
    DetectorLogicalView,
    LogicalViewConfig,
)
from ess.livedata.handlers.detector_view_specs import DetectorViewParams
from ess.livedata.handlers.monitor_data_handler import attach_monitor_workflow_factory
from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow
from ess.livedata.handlers.timeseries_handler import attach_timeseries_workflow_factory

from .specs import (
    monitor_workflow_handle,
    panel_0_view_handle,
    timeseries_workflow_handle,
    total_counts_handle,
)

# Get instrument from registry (already registered by specs.py)
instrument = instrument_registry['dummy']

# Add detector (uses explicit detector_number so no file needed)
instrument.add_detector(
    'panel_0',
    detector_number=sc.arange('yx', 1, 128**2 + 1, unit=None).fold(
        dim='yx', sizes={'y': -1, 'x': 128}
    ),
)

# Create detector view (heavy: uses DetectorLogicalView)
_panel_0_config = LogicalViewConfig(
    name='panel_0_xy',
    title='Panel 0',
    description='',
    source_names=['panel_0'],
)
_panel_0_view = DetectorLogicalView(instrument=instrument, config=_panel_0_config)


@panel_0_view_handle.attach_factory()
def _panel_0_view_factory(source_name: str, params: DetectorViewParams):
    """Factory for panel_0 detector view."""
    return _panel_0_view.make_view(source_name, params=params)


# Total counts workflow
Events = NewType('Events', sc.DataArray)
TotalCounts = NewType('TotalCounts', sc.DataArray)


def _total_counts(events: Events) -> TotalCounts:
    """Calculate total counts from events."""
    return TotalCounts(events.to(dtype='int64').sum())


_total_counts_workflow = sciline.Pipeline((_total_counts,))


@total_counts_handle.attach_factory()
def _total_counts_processor() -> StreamProcessorWorkflow:
    """Dummy processor for development and testing."""
    return StreamProcessorWorkflow(
        base_workflow=_total_counts_workflow.copy(),
        dynamic_keys={'panel_0': Events},
        target_keys={'total_counts': TotalCounts},
        accumulators=(TotalCounts,),
    )


# Attach monitor and timeseries workflow factories
attach_monitor_workflow_factory(monitor_workflow_handle)
attach_timeseries_workflow_factory(timeseries_workflow_handle)
