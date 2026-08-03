# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
BEER instrument spec registration.
"""

from ess.livedata.config import (
    Instrument,
    SourceMetadata,
    filter_authorized_streams,
    instrument_registry,
    name_streams,
)
from ess.livedata.workflows.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)

from .streams_parsed import PARSED_STREAMS
from .views import get_bank_view, get_panel_view

#: Contiguous pixel-ID range per detector bank, matching ``detector_number`` in
#: the NeXus baseline. On the wire the banks are the sources ``detector_a``
#: (south) and ``detector_b`` (north); they face each other at +/-90 degrees,
#: 2 m from the sample.
detector_pixel_ranges = {
    'beer_detector_s2': (1, 12_000_000),
    'beer_detector_n2': (12_000_001, 24_000_000),
}

detector_names = list(detector_pixel_ranges)

#: Beam monitors, named after their Kafka source names so that what the topic
#: carries is what the dashboard shows. ``cbm1`` and ``cbm2`` publish da00
#: histograms, ``hereon`` publishes ev44 events; both schemas are routed to the
#: same monitor workflow.
monitor_names = ['cbm1', 'cbm2', 'hereon']

#: The hexapod axes are the only NeXus groups whose leaf names are unique on
#: their own, so the auto-derived names would drop the parent and yield bare
#: ``position_x/value``. Restore a common prefix so they are identifiable in the
#: timeseries UI.
_hexapod_rename = {
    f'/entry/instrument/symetrie_beer_hexapod/{axis}/{leaf}': f'hexapod/{axis}/{leaf}'
    for axis in (
        'position_x',
        'position_y',
        'position_z',
        'rotation_x',
        'rotation_y',
        'rotation_z',
        'rotary_stage',
    )
    for leaf in ('value', 'target_value', 'idle_flag')
}

instrument = Instrument(
    name='beer',
    detector_names=detector_names,
    monitors=monitor_names,
    streams=name_streams(
        filter_authorized_streams(PARSED_STREAMS), rename=_hexapod_rename
    ),
    source_metadata={
        'beer_detector_s2': SourceMetadata(
            title='South Bank', description='Detector bank at -90 degrees.'
        ),
        'beer_detector_n2': SourceMetadata(
            title='North Bank', description='Detector bank at +90 degrees.'
        ),
        'cbm1': SourceMetadata(title='Beam Monitor 1'),
        'cbm2': SourceMetadata(title='Beam Monitor 2'),
        'hereon': SourceMetadata(title='Hereon Monitor'),
    },
)

instrument_registry.register(instrument)

register_monitor_workflow_specs(
    instrument, monitor_names, params=TOAOnlyMonitorDataParams
)

instrument.add_logical_view(
    name='bank_view',
    title='Detector Bank',
    description='Bank image, summed over the 12 panels and binned to 4 mm pixels.',
    source_names=detector_names,
    transform=get_bank_view,
    reduction_dim=['panel', 'y_bin', 'x_bin'],
)

instrument.add_logical_view(
    name='panel_view',
    title='Detector Panels',
    description='Per-panel images of a bank, binned to 8 mm pixels.',
    source_names=detector_names,
    transform=get_panel_view,
    reduction_dim=['y_bin', 'x_bin'],
    output_ndim=3,
    roi_support=False,
)
