# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument spec registration.
"""

import pydantic
import scipp as sc

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.workflow_spec import WorkflowOutputsBase
from ess.livedata.parameter_models import TOAEdges

from .views import get_multiblade_view

detector_names = ['multiblade_detector']

instrument = Instrument(
    name='estia',
    detector_names=detector_names,
    monitors=[],
    f144_attribute_registry={},
)

instrument_registry.register(instrument)

instrument.add_logical_view(
    name='estia_multiblade_detector_view',
    title='Multiblade Detector',
    description='Counts folded into strip, blade, and wire dimensions',
    source_names=['multiblade_detector'],
    transform=get_multiblade_view,
    roi_support=False,
    output_ndim=3,
)


class EstiaSpectrumViewParams(pydantic.BaseModel):
    """Parameters for ESTIA spectrum view."""

    toa_edges: TOAEdges = pydantic.Field(
        title='Time of arrival edges',
        description='Histogram bin edges for the time-of-arrival axis.',
        default_factory=lambda: TOAEdges(start=0.0, stop=1000.0 / 14, num_bins=100),
    )


class SpectrumViewOutputs(WorkflowOutputsBase):
    """Outputs for ESTIA spectrum view workflow."""

    spectrum_view: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(
                dims=['blade', 'wire', 'event_time_offset'],
                shape=[0, 0, 0],
                unit='counts',
            ),
            coords={'event_time_offset': sc.arange('event_time_offset', 0, unit='ms')},
        ),
        title='Spectrum View',
        description='Spectrum view showing accumulated counts in time-of-arrival vs. '
        'blade and wire.',
    )


spectrum_view_handle = instrument.register_spec(
    name='spectrum_view',
    version=1,
    title='Spectrum view',
    description='Spectrum view with configurable time-of-arrival bins. Counts within '
    'the same strip are summed.',
    source_names=['multiblade_detector'],
    params=EstiaSpectrumViewParams,
    outputs=SpectrumViewOutputs,
)
