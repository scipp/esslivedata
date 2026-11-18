# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
FREIA instrument spec registration.

FREIA is a horizontal-sample reflectometer with a single Multi-Blade detector.
Only logical views are registered; there is no geometry artifact, geometric
projection, reduction workflow or wavelength lookup table yet. See ``views`` for
how the detector's pixel offsets define the fold.
"""

from ess.livedata.config import (
    Instrument,
    filter_authorized_streams,
    instrument_registry,
    name_streams,
)
from ess.livedata.config.device_contract import DETECTOR_VIEW_DEVICES
from ess.livedata.workflows.detector_view_specs import SpectrumViewSpec
from ess.livedata.workflows.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)

from .streams_parsed import PARSED_STREAMS
from .views import get_multiblade_view, get_spectrum_view

detector_names = ['multiblade_detector']

#: Inclusive ``detector_number`` range of the single bank, as in
#: ``coda_freia_999999_00030110.hdf``. Stated here rather than derived from
#: ``ess.freia``'s bank sizes, which specs may not import.
detector_pixel_range = (1, 65536)

#: Named after their NeXus groups. Their Kafka source names are ``cbm1``..``cbm6``.
monitor_names = [f'monitor_{i}' for i in range(1, 7)]

streams = name_streams(filter_authorized_streams(PARSED_STREAMS))

instrument = Instrument(
    name='freia',
    detector_names=detector_names,
    monitors=monitor_names,
    streams=streams,
)

instrument_registry.register(instrument)

register_monitor_workflow_specs(
    instrument,
    instrument.monitors,
    params=TOAOnlyMonitorDataParams,
)

instrument.add_logical_view(
    name='freia_multiblade_detector_view',
    title='Multiblade Detector',
    description='Counts folded into blade, wire, and strip dimensions',
    source_names=detector_names,
    transform=get_multiblade_view,
    roi_support=False,
    output_ndim=3,
    spectrum_view=SpectrumViewSpec(
        transform=get_spectrum_view,
        output_dims=['blade', 'wire'],
        extra_description='Summed across strips, yielding per-blade, per-wire spectra.',
    ),
    device_outputs=DETECTOR_VIEW_DEVICES,
)
