# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
NMX instrument spec registration.
"""

from ess.livedata.config import (
    Instrument,
    filter_authorized_streams,
    instrument_registry,
    name_streams,
)
from ess.livedata.config.device_contract import COUNTS_TOTAL_DEVICE
from ess.livedata.workflows.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)

from .streams_parsed import PARSED_STREAMS

# Detector panel names
detector_names = ['detector_panel_0', 'detector_panel_1', 'detector_panel_2']

#: Side length one panel is ingested at. At full resolution a panel publishes
#: 26 MB per update, more than the broker or the dashboard wants, and grouping
#: it costs hundreds of milliseconds regardless of how many events arrived.
PANEL_IMAGE_RESOLUTION = 320
#: Side length of one panel's readout grid. Fixed, unlike the Timepix3 panels
#: of TBL and ODIN, so the ingest is given the stride outright rather than
#: inferring it from the event ids. That also makes the panel enumeration in
#: ``factories.py`` a check on this number rather than merely a consumer of it.
PANEL_RESOLUTION = 1280

# Create instrument with detectors and monitors
instrument = Instrument(
    name='nmx',
    detector_names=detector_names,
    monitors=['monitor1', 'monitor2'],
    streams=name_streams(filter_authorized_streams(PARSED_STREAMS)),
)

# Register instrument
instrument_registry.register(instrument)

# Register monitor workflow spec (TOA-only, no TOF lookup tables)
register_monitor_workflow_specs(
    instrument, ['monitor1', 'monitor2'], params=TOAOnlyMonitorDataParams
)

for _panel in detector_names:
    instrument.configure_detector_downsampling(
        _panel,
        resolution=PANEL_IMAGE_RESOLUTION,
        source_resolution=PANEL_RESOLUTION,
    )

# Register detector view spec for the panel_xy view
instrument.add_logical_view(
    name='panel_xy',
    title='Detector counts',
    description=(
        f'{PANEL_IMAGE_RESOLUTION}x{PANEL_IMAGE_RESOLUTION} image per panel,'
        f' downsampled from the {PANEL_RESOLUTION}x{PANEL_RESOLUTION} readout'
        ' as the events are ingested.'
    ),
    source_names=detector_names,
    device_outputs=COUNTS_TOTAL_DEVICE,
)
