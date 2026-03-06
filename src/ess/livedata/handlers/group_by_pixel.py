# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import scipp as sc
from ess.reduce.nexus import group_event_data

from ..core.handler import Accumulator
from .to_nxevent_data import Events, ToNXevent_data


class GroupByPixel(Accumulator[Events, sc.DataArray]):
    """Accumulator that groups events by detector pixel and assembles detector data.

    Wraps a ``ToNXevent_data`` accumulator and applies ``group_event_data``
    in ``get()``, producing events binned by ``detector_number`` with the
    correct multi-dimensional shape and geometry coordinates from the
    pre-computed EmptyDetector.

    This allows pixel grouping and detector assembly to happen once in the
    preprocessor rather than independently in every downstream workflow.

    Parameters
    ----------
    inner:
        The underlying event accumulator.
    empty_detector:
        Pre-computed EmptyDetector with detector_number coordinate and
        optional geometry (position, masks). Determines the output shape.
    """

    def __init__(self, inner: ToNXevent_data, empty_detector: sc.DataArray) -> None:
        self._inner = inner
        self._empty_detector = empty_detector

    def add(self, timestamp: int, data: Events) -> None:
        self._inner.add(timestamp, data)

    def get(self) -> sc.DataArray:
        ungrouped = self._inner.get()
        detector_number = self._empty_detector.coords['detector_number']
        grouped = group_event_data(
            event_data=ungrouped,
            detector_number=detector_number,
        )
        # Assign geometry coords and masks from EmptyDetector
        for name, coord in self._empty_detector.coords.items():
            if name != 'detector_number':
                grouped.coords[name] = coord
        for name, mask in self._empty_detector.masks.items():
            grouped.masks[name] = mask
        # Add variances (matching assemble_detector_data behavior)
        if grouped.bins is not None:
            content = grouped.bins.constituents['data']
            if content.data.variances is None:
                content.data = sc.values(content.data).copy()
                content.data.variances = content.data.values
        return grouped

    def clear(self) -> None:
        self._inner.clear()
