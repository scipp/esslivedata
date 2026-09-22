# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
FREIA instrument spec registration.

FREIA is a horizontal-sample reflectometer with a single Multi-Blade detector.
Only geometry-free views are registered: the reference file places every
component, the source included, at the origin, so neither a geometric projection
nor a reduction workflow nor a wavelength lookup table has the positions it needs.
See ``views`` for what the detector's pixel offsets do and do not tell us.
"""

import math

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
from .views import DETECTOR_BANK_SIZES, get_multiblade_view, get_spectrum_view

detector_names = list(DETECTOR_BANK_SIZES)

#: Inclusive ``detector_number`` range of the single bank.
detector_pixel_range = (
    1,
    math.prod(DETECTOR_BANK_SIZES['multiblade_detector'].values()),
)

#: Named after their NeXus groups, without the ``MISSING_ESSCONFIG_`` prefix the
#: file writer puts on 4-6. Their Kafka source names upstream are
#: ``monitor_<i>_events``, which no other instrument uses and which looks like a
#: placeholder, so the PROD mapping assumes the conventional ``cbm<i>``.
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
    description='Counts folded into strip, blade, and wire dimensions',
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
