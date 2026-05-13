# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Auto-generated NeXus f144 stream declarations.

Do not edit by hand. Regenerate with
``python -m ess.livedata.nexus_helpers <geometry.nxs> --generate``.

Source: geometry-loki-2026-04-13.nxs
"""

from ess.livedata.config import F144Stream

PARSED_STREAMS: list[F144Stream] = [
    F144Stream(
        stream_name='detector_carriage',
        nexus_path='/entry/instrument/detector_carriage/value',
        source='LOKI-DtCar1:MC-LinX-01:Mtr.RBV',
        topic='loki_motion',
        units='mm',
    ),
]
