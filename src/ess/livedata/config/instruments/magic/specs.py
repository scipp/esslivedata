# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
MAGIC instrument spec registration.

MAGIC (single-crystal magnetism diffractometer) has two detector banks shaped
like vertical cylinders, resembling the DREAM mantle rotated onto Y. This module
registers the geometry-free logical detector views (wire and strip views), the
cylinder-Y mantle projection of each bank, and the beam monitor.
"""

import math

from ess.livedata.config import (
    Instrument,
    SourceMetadata,
    filter_authorized_streams,
    instrument_registry,
    name_streams,
)
from ess.livedata.config.device_contract import DETECTOR_VIEW_DEVICES
from ess.livedata.workflows.detector_view_specs import TOAOnlyDetectorViewParams
from ess.livedata.workflows.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)

from .streams_parsed import PARSED_STREAMS
from .views import DETECTOR_BANK_SIZES, get_strip_view, get_wire_view


def _pixel_ranges() -> dict[str, tuple[int, int]]:
    """Inclusive ``detector_number`` range per bank.

    The banks are numbered contiguously from 1 in declaration order, so the
    ranges follow from the voxel counts and cannot drift away from the folds
    the logical views apply.
    """
    ranges = {}
    start = 1
    for name, sizes in DETECTOR_BANK_SIZES.items():
        stop = start + math.prod(sizes.values()) - 1
        ranges[name] = (start, stop)
        start = stop + 1
    return ranges


#: Inclusive ``detector_number`` range per bank. Incoming ``event_id`` is assumed
#: to fall inside the range of the bank it arrives on. The reference files
#: ``coda_magic_999999_*.hdf`` violate this for bank A: its ``event_id``s only
#: cover ids 1-238336, so the upper half of the bank is empty. This is a
#: producer-side bug to fix upstream, not worked around here (scipp/esslivedata#1475).
detector_pixel_ranges = _pixel_ranges()
detector_names = list(detector_pixel_ranges)

#: The cave monitor, Kafka source ``cbm1``. The bunker monitor is not mapped: its
#: Kafka source in the run files is the placeholder ``bbm1_todo``.
monitor_names = ['cave_beam_monitor']


instrument = Instrument(
    name='magic',
    detector_names=detector_names,
    monitors=monitor_names,
    streams=name_streams(filter_authorized_streams(PARSED_STREAMS)),
    source_metadata={
        'magic_detector_a': SourceMetadata(
            title='Main bank',
            description='Main detector bank (~490k voxels).',
        ),
        'magic_detector_b': SourceMetadata(
            title='Polarization bank',
            description='Detector bank behind the analyzer (~130k voxels).',
        ),
        'cave_beam_monitor': SourceMetadata(title='Cave Beam Monitor'),
    },
)

instrument_registry.register(instrument)

register_monitor_workflow_specs(
    instrument, monitor_names, params=TOAOnlyMonitorDataParams
)


wire_view_handle = instrument.add_logical_view(
    name='wire_view',
    title='Wire view',
    description='Sum over strips to show counts per wire.',
    source_names=detector_names,
    transform=get_wire_view,
    roi_support=False,
    reduction_dim='strip',
)
strip_view_handle = instrument.add_logical_view(
    name='strip_view',
    title='Strip view',
    description='Sum over all dimensions except strip to show counts per strip.',
    source_names=detector_names,
    transform=get_strip_view,
    output_ndim=1,
    roi_support=False,
    reduction_dim='wire/segment',
)

# Both banks are vertical (Y-axis) cylinders; project onto the mantle. Pixel
# positions are the file's own ``[xyz]_pixel_offset``, which sit at the voxel
# centroids, halfway between the inner and outer radial faces of each cuboid voxel
# (8-10 mm from either). Which face is the active surface has not been
# confirmed against the hardware.
projection_handle = instrument.register_detector_view(
    name='detector_projection',
    title='Detector Projection',
    description='Projection of the cylindrical detector banks onto their mantle.',
    source_names=detector_names,
    params=TOAOnlyDetectorViewParams,
    # The projection, not the wire or strip view, carries the NICOS devices.
    device_outputs=DETECTOR_VIEW_DEVICES,
)
