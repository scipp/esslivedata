# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Dummy instrument spec registration.
"""

import pydantic
import scipp as sc

from ess.livedata.config import F144Stream, Instrument, instrument_registry
from ess.livedata.config.workflow_spec import DETECTORS, WorkflowOutputsBase
from ess.livedata.preprocessors.detector_view_specs import (
    DetectorROIAuxSources,
    DetectorViewOutputs,
    DetectorViewParams,
)
from ess.livedata.preprocessors.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)


class TotalCountsOutputs(WorkflowOutputsBase):
    """Outputs for the total counts workflow."""

    total_counts: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts')),
        title='Total',
        description='Sum of all detector counts.',
    )


detector_names = ['panel_0', 'area_panel']

f144_streams: dict[str, F144Stream] = {
    'motion1': F144Stream(source='motion1', topic='dummy_motion', units='mm'),
    'motion2': F144Stream(source='motion2', topic='dummy_motion', units='deg'),
}

instrument = Instrument(
    name='dummy',
    detector_names=detector_names,
    monitors=['monitor1', 'monitor2'],
    streams=f144_streams,
)

# Register instrument
instrument_registry.register(instrument)

# Register monitor workflow spec (TOA-only, no TOF lookup tables)
register_monitor_workflow_specs(
    instrument, ['monitor1', 'monitor2'], params=TOAOnlyMonitorDataParams
)

# Register detector view spec.
# Note: We don't use register_detector_view_specs here because dummy uses
# DetectorLogicalView which doesn't follow the projection pattern.
panel_0_view_handle = instrument.register_spec(
    group=DETECTORS,
    name='panel_0_xy',
    version=1,
    title='Panel 0',
    description=(
        'Detector image for the dummy "panel_0" logical detector, projected '
        'onto its pixel grid. Counts can be binned by time-of-arrival and '
        'restricted to regions of interest.'
    ),
    source_names=['panel_0'],
    aux_sources=DetectorROIAuxSources(),
    params=DetectorViewParams,
    outputs=DetectorViewOutputs,
)

# Register area detector view spec
area_panel_view_handle = instrument.register_spec(
    group=DETECTORS,
    name='area_panel_xy',
    version=1,
    title='Area Panel',
    description='Area detector image view',
    source_names=['area_panel'],
    params=None,
    outputs=DetectorViewOutputs,
)

# Register total counts workflow spec
total_counts_handle = instrument.register_spec(
    name='total_counts',
    version=1,
    title='Total counts',
    description='Dummy workflow that simply computes the total counts.',
    source_names=['panel_0'],
    outputs=TotalCountsOutputs,
    params=None,
)
