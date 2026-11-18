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
from ess.livedata.config.device_contract import DETECTOR_VIEW_DEVICES
from ess.livedata.workflows.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)

from .streams_parsed import PARSED_STREAMS
from .views import name_image_dims

#: Side length the Timepix3 panel is ingested at; see name_image_dims.
TIMEPIX3_IMAGE_RESOLUTION = 512
#: Largest grid the Timepix3 panel can read out. The readout is
#: reconfigured during operation, so this bounds the inferred streamed
#: resolution rather than stating it.
TIMEPIX3_PANEL_RESOLUTION = 4096

#: Choppers feeding the wavelength-LUT cascade, in beam order. The geometry
#: artifact also holds a ``t0`` chopper, left out because that chopper does not
#: exist yet; its NeXus entry is a placeholder on PVs that never connect
#: (``ODIN:PLACEHOLDER:T0:*``). The cascade fires only once every configured
#: chopper has locked, so listing a device that cannot stream would hold the LUT
#: permanently pending. Add it when the chopper is installed.
#:
#: The artifact's ``wfm1``/``wfm2`` axle positions carry a hand repair: the run
#: file chains their offsets (absolute: -53.725 / -53.275 m) onto an
#: NXlog on the ``ODIN-ChpLin:MC-LinX-01:Mtr`` readback, which the generator
#: trims to length 0, so the positions come out empty. The artifact ends both
#: chains at the offset instead (``depends_on='.'``). The two positioners also
#: read the same PV and ``make_f144_log_lut`` keys on (topic, source), so only
#: one of the two streams would be fed were the chains live.
ODIN_CHOPPERS = [
    'wfm1',
    'wfm2',
    'bpc1',
    'bpc2',
    'foc1',
    'foc2',
    'foc3',
    'foc4',
    'foc5',
]

instrument = Instrument(
    name='odin',
    detector_names=['timepix3'],
    monitors=['monitor1', 'monitor2'],
    choppers=ODIN_CHOPPERS,
    streams=name_streams(filter_authorized_streams(PARSED_STREAMS)),
)

instrument_registry.register(instrument)

# TOA-only. Wavelength mode needs a flight path. The geometry file places
# beam_monitor_1..3, but no group is named monitor1 or monitor2 (#1346).
register_monitor_workflow_specs(
    instrument, ['monitor1', 'monitor2'], params=TOAOnlyMonitorDataParams
)

instrument.configure_detector_downsampling(
    'timepix3',
    resolution=TIMEPIX3_IMAGE_RESOLUTION,
    source_resolution=TIMEPIX3_PANEL_RESOLUTION,
    reconfigurable=True,
)

# Detector view spec registration (with ROI support)
instrument.add_logical_view(
    name='odin_detector_xy',
    title='Timepix3 XY Detector Counts',
    description=(
        f'{TIMEPIX3_IMAGE_RESOLUTION}x{TIMEPIX3_IMAGE_RESOLUTION} image,'
        ' downsampled from the streamed resolution as the events are'
        ' ingested. The streamed resolution is read from the event ids; a'
        ' readout reconfiguration restarts the cumulative image, since counts'
        ' taken before and after it are not commensurable.'
    ),
    source_names=['timepix3'],
    transform=name_image_dims,
    roi_support=True,
    device_outputs=DETECTOR_VIEW_DEVICES,
)
