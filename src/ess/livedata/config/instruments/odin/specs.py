# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN instrument spec registration.
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
from .views import name_image_dims

#: Side length the Timepix3 panel is ingested at; see name_image_dims.
IMAGE_RESOLUTION = 512

instrument = Instrument(
    name='odin',
    detector_names=['timepix3'],
    monitors=['monitor1', 'monitor2'],
    streams=name_streams(filter_authorized_streams(PARSED_STREAMS)),
)

instrument_registry.register(instrument)

# Register monitor workflow spec (TOA-only, no TOF lookup tables)
register_monitor_workflow_specs(
    instrument, ['monitor1', 'monitor2'], params=TOAOnlyMonitorDataParams
)

instrument.configure_detector_downsampling('timepix3', resolution=IMAGE_RESOLUTION)

# Detector view spec registration (with ROI support)
instrument.add_logical_view(
    name='odin_detector_xy',
    title='Timepix3 XY Detector Counts',
    description=(
        f'{IMAGE_RESOLUTION}x{IMAGE_RESOLUTION} image, downsampled from full'
        ' resolution as the events are ingested. The full resolution is read'
        ' from the event ids, so it may be revised upward shortly after a run'
        ' starts if the first events cover only part of the panel. Cumulative'
        ' results from before such a revision are wrong and are not discarded'
        ' automatically: reset or restart the workflow to clear them.'
    ),
    source_names=['timepix3'],
    transform=name_image_dims,
    roi_support=True,
    device_outputs=COUNTS_TOTAL_DEVICE,
)
