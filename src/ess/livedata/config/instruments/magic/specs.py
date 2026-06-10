# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
MAGIC instrument spec registration.

MAGIC (single-crystal magnetism diffractometer) has two detector banks shaped
like vertical cylinders, resembling the DREAM mantle. This module registers the
geometry-free logical detector views (wire and strip views). Monitor workflows
and the cylinder-Y projection are deferred (the latter needs an upstream
essreduce addition).
"""

from ess.livedata.config import (
    Instrument,
    SourceMetadata,
    instrument_registry,
)

from .views import get_strip_view, get_wire_view

detector_names = ['magic_detector_a', 'magic_detector_b']


instrument = Instrument(
    name='magic',
    detector_names=detector_names,
    source_metadata={
        'magic_detector_a': SourceMetadata(
            title='Main bank',
            description='Main detector bank (~500k voxels).',
        ),
        'magic_detector_b': SourceMetadata(
            title='Polarization bank',
            description='Detector bank behind the analyzer (~130k voxels).',
        ),
    },
)

instrument_registry.register(instrument)


instrument.add_logical_view(
    name='wire_view',
    title='Wire view',
    description='Sum over strips to show counts per wire.',
    source_names=detector_names,
    transform=get_wire_view,
    roi_support=False,
    reduction_dim='strip',
)
instrument.add_logical_view(
    name='strip_view',
    title='Strip view',
    description='Sum over all dimensions except strip to show counts per strip.',
    source_names=detector_names,
    transform=get_strip_view,
    output_ndim=1,
    roi_support=False,
    reduction_dim='other',
)
