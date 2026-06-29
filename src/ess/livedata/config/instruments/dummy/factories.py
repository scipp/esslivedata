# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Dummy instrument factory implementations.
"""

from typing import NewType

import scipp as sc

from ess.livedata.config import Instrument

from . import specs

# Total counts workflow types
Events = NewType('Events', sc.DataArray)
TotalCounts = NewType('TotalCounts', sc.DataArray)


def _total_counts(events: Events) -> TotalCounts:
    """Calculate total counts from events."""
    return TotalCounts(events.to(dtype='int64').sum())


def setup_factories(instrument: Instrument) -> None:
    """Initialize dummy-specific factories and workflows."""
    import sciline

    from ess.livedata.handlers.area_detector_view import AreaDetectorView
    from ess.livedata.handlers.detector_view import (
        DetectorViewFactory,
        InstrumentDetectorSource,
        LogicalViewConfig,
    )
    from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow

    # Configure detector with explicit detector_number
    instrument.configure_detector(
        'panel_0',
        detector_number=sc.arange('yx', 1, 128**2 + 1, unit=None).fold(
            dim='yx', sizes={'y': -1, 'x': 128}
        ),
    )

    # Create detector view for event-mode panel_0 using Sciline-based factory
    _panel_0_view = DetectorViewFactory(
        data_source=InstrumentDetectorSource(instrument),
        view_config=LogicalViewConfig(),  # Identity transform
    )

    specs.panel_0_view_handle.attach_factory()(_panel_0_view.make_workflow)

    # Area detector view for area_panel (ad00 images)
    specs.area_panel_view_handle.attach_factory()(AreaDetectorView.view_factory())

    # Total counts workflow
    _total_counts_workflow = sciline.Pipeline((_total_counts,))

    @specs.total_counts_handle.attach_factory()
    def _total_counts_processor() -> StreamProcessorWorkflow:
        """Dummy processor for development and testing."""
        return StreamProcessorWorkflow(
            base_workflow=_total_counts_workflow.copy(),
            dynamic_keys={'panel_0': Events},
            target_keys={'total_counts': TotalCounts},
            accumulators=(TotalCounts,),
        )
