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

Skipping the grouping makes the preprocessor's pixel order the workflow's:
``assemble_detector_data`` then assigns the ``EmptyDetector`` coords, such as
positions, by array position. The preprocessor groups by
``Instrument.get_detector_number`` while the workflow's ``EmptyDetector`` may
come from the geometry file, so the patch checks that both list the same
detector numbers in the same order, and raises otherwise.

This is a temporary measure until ``ess.reduce`` is updated upstream.

Whatever replaces it must keep top-level coords: the upstream implementation
rebuilds the array and drops them, whereas this one passes them through, and
``DownsamplePixelIds`` relies on that to carry ``SOURCE_RESOLUTION`` to the
cumulative accumulator's reset. Unlike ``DETECTOR_TRANSFORM``, which the
workflow re-stamps from a static pipeline value, the source resolution is only
known to the preprocessor and has nowhere else to come from.
``tests/services/detector_data_test.py`` covers the resulting reset end to end.
"""

import ess.reduce.nexus._nexus_loader as _nexus_loader
import numpy as np
import scipp as sc

_original_group_event_data = _nexus_loader.group_event_data


def _idempotent_group_event_data(
    *, event_data: sc.DataArray, detector_number: sc.Variable
) -> sc.DataArray:
    if 'detector_number' in event_data.coords:
        # Already grouped — data has a detector_number coordinate from a
        # previous group_event_data call. Re-fold to target shape if needed
        # (e.g., preprocessor produced flat 1D, workflow expects multi-dim).
        grouped = event_data.coords['detector_number'].values.ravel()
        if not np.array_equal(grouped, detector_number.values.ravel()):
            raise ValueError(
                'Events were grouped by a detector_number that differs from the '
                "workflow's in size or order. The workflow attaches per-pixel "
                'coords such as positions by position, so they would land on the '
                'wrong pixels.'
            )
        if event_data.sizes != detector_number.sizes:
            return event_data.fold(dim='detector_number', sizes=detector_number.sizes)
        return event_data
    return _original_group_event_data(
        event_data=event_data, detector_number=detector_number
    )


_nexus_loader.group_event_data = _idempotent_group_event_data
