# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Dummy instrument spec registration.
"""

import pydantic
import scipp as sc

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.workflow_spec import WorkflowOutputsBase
from ess.livedata.handlers.detector_view_specs import DetectorViewParams


class TotalCountsOutputs(WorkflowOutputsBase):
    """Outputs for the total counts workflow."""

    total_counts: sc.DataArray = pydantic.Field(
        title='Total Counts',
        description='Sum of all detector counts.',
    )


detector_names = ['panel_0']

instrument = Instrument(
    name='dummy',
    detector_names=detector_names,
    monitors=['monitor1', 'monitor2'],
    f144_attribute_registry={'motion1': {'units': 'mm'}},
)

# Register instrument
instrument_registry.register(instrument)

# Register detector view spec.
# Note: We don't use register_detector_view_specs here because dummy uses
# DetectorLogicalView which doesn't follow the projection pattern.
panel_0_view_handle = instrument.register_spec(
    namespace='detector_data',
    name='panel_0_xy',
    version=1,
    title='Panel 0',
    description='',
    source_names=['panel_0'],
    params=DetectorViewParams,
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
