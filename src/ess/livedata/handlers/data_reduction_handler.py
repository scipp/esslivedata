# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import scipp as sc

from ..core.handler import Accumulator, JobBasedPreprocessorFactoryBase
from ..core.message import StreamId, StreamKind
from .accumulators import Cumulative, DetectorEvents
from .to_nxevent_data import ToNXevent_data
from .to_nxlog import ToNXlog


class ReductionHandlerFactory(
    JobBasedPreprocessorFactoryBase[DetectorEvents, sc.DataGroup[sc.DataArray]]
):
    """Factory for data reduction handlers."""

    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        match key.kind:
            case StreamKind.MONITOR_COUNTS:
                return Cumulative(clear_on_get=True)
            case StreamKind.LOG:
                attrs = self._instrument.f144_attribute_registry[key.name]
                return ToNXlog(attrs=attrs)
            case StreamKind.MONITOR_EVENTS | StreamKind.DETECTOR_EVENTS:
                return ToNXevent_data()
            case _:
                return None
