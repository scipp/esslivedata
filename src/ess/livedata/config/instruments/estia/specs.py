# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument spec registration.
"""

import scipp as sc

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.handlers.detector_view_specs import SpectrumViewSpec
from ess.livedata.handlers.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)

from .._ess import GENERIC_CBM_DESCRIPTION_NOTE, GENERIC_CBM_MONITORS
from .views import get_multiblade_view

detector_names = ['multiblade_detector']

instrument = Instrument(
    name='estia',
    detector_names=detector_names,
    monitors=list(GENERIC_CBM_MONITORS),
    f144_attribute_registry={},
)

instrument_registry.register(instrument)

monitor_handle = register_monitor_workflow_specs(
    instrument,
    instrument.monitors,
    params=TOAOnlyMonitorDataParams,
    extra_description=GENERIC_CBM_DESCRIPTION_NOTE,
)


def _estia_spectrum_transform(histogram: sc.DataArray) -> sc.DataArray:
    """Sum over the ``strip`` axis (constant scattering angle)."""
    return histogram.sum('strip')


instrument.add_logical_view(
    name='estia_multiblade_detector_view',
    title='Multiblade Detector',
    description='Counts folded into strip, blade, and wire dimensions',
    source_names=['multiblade_detector'],
    transform=get_multiblade_view,
    roi_support=False,
    output_ndim=3,
    spectrum_view=SpectrumViewSpec(
        transform=_estia_spectrum_transform,
        output_dims=['blade', 'wire'],
        extra_description='Summed across strips, yielding per-blade, per-wire spectra.',
    ),
)
