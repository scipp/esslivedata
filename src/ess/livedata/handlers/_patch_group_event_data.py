# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Monkey-patch ``ess.reduce.nexus.group_event_data`` to be idempotent.

When the preprocessor groups events by detector pixel via ``GroupByPixel``,
downstream workflows call ``assemble_detector_data`` which calls
``group_event_data`` again. Without this patch the second call fails because
the data is already binned by ``detector_number``.

This patch makes ``group_event_data`` detect already-grouped input and
skip the grouping step, but still fold to the target shape defined by
``detector_number.sizes`` (e.g., flat 1D → multi-dimensional).

This is a temporary measure until ``ess.reduce`` is updated upstream.
"""

import ess.reduce.nexus._nexus_loader as _nexus_loader
import scipp as sc

_original_group_event_data = _nexus_loader.group_event_data


def _idempotent_group_event_data(
    *, event_data: sc.DataArray, detector_number: sc.Variable
) -> sc.DataArray:
    if 'detector_number' in event_data.coords:
        # Already grouped — data has a detector_number coordinate from a
        # previous group_event_data call. Re-fold to target shape if needed
        # (e.g., preprocessor produced flat 1D, workflow expects multi-dim).
        if event_data.sizes != detector_number.sizes:
            return event_data.fold(dim='detector_number', sizes=detector_number.sizes)
        return event_data
    return _original_group_event_data(
        event_data=event_data, detector_number=detector_number
    )


_nexus_loader.group_event_data = _idempotent_group_event_data
