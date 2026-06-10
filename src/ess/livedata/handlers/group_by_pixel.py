# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import scipp as sc
from ess.reduce.nexus import group_event_data

# Apply monkey-patch to make group_event_data idempotent for pre-grouped data.
# See module docstring for rationale.
import ess.livedata.handlers._patch_group_event_data  # noqa: F401

from ..core.handler import Accumulator
from ..core.timestamp import Timestamp
from .to_nxevent_data import Events, ToNXevent_data


class GroupByPixel(Accumulator[Events, sc.DataArray]):
    """Accumulator that groups events by detector pixel.

    Wraps a ``ToNXevent_data`` accumulator and applies ``group_event_data``
    in ``get()``, producing events binned by ``detector_number``.

    This allows pixel grouping to happen once in the preprocessor rather than
    independently in every downstream workflow that consumes the same source.

    Parameters
    ----------
    inner:
        The underlying event accumulator.
    detector_number:
        Detector pixel numbers used for grouping.
    event_id_offset:
        Constant added to every ``event_id`` before grouping. Defaults to 0
        (no-op). This is a temporary escape hatch for instruments whose event
        producer numbers ``event_id`` in a different origin than
        ``detector_number``; it should be removed once the producer is fixed.
    """

    def __init__(
        self,
        inner: ToNXevent_data,
        detector_number: sc.Variable,
        *,
        event_id_offset: int = 0,
    ) -> None:
        self._inner = inner
        # Always use flat 1D detector_number for grouping. The downstream
        # workflow's assemble_detector_data will fold to the correct
        # multi-dimensional shape using EmptyDetector's detector_number.
        self._detector_number = detector_number.flatten(to='detector_number')
        self._event_id_offset = event_id_offset

    def add(self, timestamp: Timestamp, data: Events) -> bool:
        return self._inner.add(timestamp, data)

    def get(self) -> sc.DataArray:
        ungrouped = self._inner.get()
        # group_event_data re-bins, producing an independent copy.
        # Release the inner buffer immediately since ungrouped is consumed here.
        self._inner.release_buffers()
        if self._event_id_offset:
            eid = ungrouped.bins.coords['event_id']
            ungrouped.bins.coords['event_id'] = eid + sc.scalar(
                self._event_id_offset, dtype=eid.dtype, unit=eid.unit
            )
        return group_event_data(
            event_data=ungrouped,
            detector_number=self._detector_number,
        )

    def clear(self) -> None:
        self._inner.clear()
