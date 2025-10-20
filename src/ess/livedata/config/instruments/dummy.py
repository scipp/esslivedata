# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Detector configuration for a dummy instrument used for development and testing.
"""

from typing import NewType

import pydantic
import sciline
import scipp as sc

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.env import StreamingEnv
from ess.livedata.config.workflow_spec import WorkflowOutputsBase
from ess.livedata.handlers.detector_data_handler import (
    DetectorLogicalView,
    LogicalViewConfig,
)
from ess.livedata.handlers.monitor_data_handler import register_monitor_workflows
from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow
from ess.livedata.handlers.timeseries_handler import register_timeseries_workflows
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from ._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping

detectors_config = {'fakes': {'panel_0': (1, 128**2)}}


def _make_dummy_detectors() -> StreamLUT:
    """Dummy detector mapping."""
    return {InputStreamKey(topic='dummy_detector', source_name='panel_0'): 'panel_0'}


Events = NewType('Events', sc.DataArray)
TotalCounts = NewType('TotalCounts', sc.DataArray)


def _total_counts(events: Events) -> TotalCounts:
    """Calculate total counts from events."""
    return TotalCounts(events.to(dtype='int64').sum())


_total_counts_workflow = sciline.Pipeline((_total_counts,))


class TotalCountsOutputs(WorkflowOutputsBase):
    """Outputs for the total counts workflow."""

    total_counts: sc.DataArray = pydantic.Field(
        title='Total Counts',
        description='Sum of all detector counts.',
    )


instrument = Instrument(
    name='dummy', f144_attribute_registry={'motion1': {'units': 'mm'}}
)
register_monitor_workflows(instrument=instrument, source_names=['monitor1', 'monitor2'])
register_timeseries_workflows(instrument=instrument, source_names=['motion1'])

instrument_registry.register(instrument)
instrument.add_detector(
    'panel_0',
    detector_number=sc.arange('yx', 1, 128**2 + 1, unit=None).fold(
        dim='yx', sizes={'y': -1, 'x': 128}
    ),
)
_panel_0_config = LogicalViewConfig(
    name='panel_0_xy',
    title='Panel 0',
    description='',
    source_names=['panel_0'],
)
_panel_0_view = DetectorLogicalView(instrument=instrument, config=_panel_0_config)


@instrument.register_workflow(
    name='total_counts',
    version=1,
    title='Total counts',
    description='Dummy workflow that simply computes the total counts.',
    source_names=['panel_0'],
    outputs=TotalCountsOutputs,
)
def _total_counts_processor() -> StreamProcessorWorkflow:
    """Dummy processor for development and testing."""
    return StreamProcessorWorkflow(
        base_workflow=_total_counts_workflow.copy(),
        dynamic_keys={'panel_0': Events},
        target_keys={'total_counts': TotalCounts},
        accumulators=(TotalCounts,),
    )


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'dummy', detector_names=list(detectors_config['fakes'])
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='dummy'),
        detectors=_make_dummy_detectors(),
    ),
}
