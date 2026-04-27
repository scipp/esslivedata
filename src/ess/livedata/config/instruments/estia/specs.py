# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument spec registration.
"""

import scipp as sc

from ess.livedata.config import Instrument, SourceMetadata, instrument_registry
from ess.livedata.handlers.detector_view_specs import SpectrumViewSpec
from ess.livedata.handlers.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)

from .._ess import GENERIC_CBM_DESCRIPTION_NOTE, GENERIC_CBM_MONITORS
from .views import get_multiblade_view

detector_names = ['multiblade_detector']

# f144 log streams for ESTIA. The detector rotation is the only time-dependent
# transformation in the multiblade detector's depends_on chain; sample-detector
# distance is invariant under this rotation, so the value is currently exposed
# only for plotting, not for geometry. The PV channel ``.RBV`` (readback) is
# taken from ``coda_estia_999999_00027641.hdf`` under
# ``/entry/instrument/detector_arm/detector_rotation/value`` (NXpositioner).
# The same positioner also publishes ``.VAL`` (setpoint) and ``.DMOV``
# (done-moving) on the same topic; not exposed here.
f144_log_streams = {
    'detector_rotation': {
        'source': 'ESTIA-DtRot:MC-RotZ01:Mtr.RBV',
        'topic': 'estia_motion',
        'units': 'deg',
    },
}

instrument = Instrument(
    name='estia',
    detector_names=detector_names,
    monitors=list(GENERIC_CBM_MONITORS),
    f144_attribute_registry={
        name: {'units': info['units']} for name, info in f144_log_streams.items()
    },
    source_metadata={
        'detector_rotation': SourceMetadata(
            title='Detector Rotation',
            description='Multiblade detector bank rotation angle.',
        ),
    },
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
