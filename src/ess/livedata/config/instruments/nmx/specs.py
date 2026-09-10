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
from ess.livedata.config.device_contract import DETECTOR_VIEW_DEVICES
from ess.livedata.workflows.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)

from .streams_parsed import PARSED_STREAMS

# Detector panel names
detector_names = ['detector_panel_0', 'detector_panel_1', 'detector_panel_2']

#: Side length one panel is ingested at, a 2x2 binning of the readout. At full
#: resolution a panel publishes 26 MB per update and costs hundreds of
#: milliseconds to group, regardless of how many events arrived. Coarser
#: targets cost proportionally less, but the readout is only 0.4 mm per pixel
#: and this is the view the instrument judges its diffraction pattern from.
PANEL_IMAGE_RESOLUTION = 640
#: Side length of one panel's readout grid. Fixed, unlike the Timepix3 panels
#: of TBL and ODIN, so the ingest is given the stride outright rather than
#: inferring it from the event ids.
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
    device_outputs=DETECTOR_VIEW_DEVICES,
)
