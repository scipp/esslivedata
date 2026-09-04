# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from ..config.instrument import Instrument
from ..core.message import StreamId, StreamKind
from ..core.preprocessor import Accumulator, JobBasedPreprocessorFactoryBase
from .accumulators import Cumulative, LatestValueAccumulator
from .detector_data import make_detector_event_preprocessor
from .to_nxevent_data import ToNXevent_data
from .to_nxlog import nxlog_for_stream


class ReductionPreprocessorFactory(JobBasedPreprocessorFactoryBase):
    """Factory for data reduction preprocessors."""

    def __init__(self, *, instrument: Instrument, group_by_pixel: bool = True) -> None:
        super().__init__(instrument=instrument)
        self._group_by_pixel = group_by_pixel

    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        match key.kind:
            case StreamKind.MONITOR_COUNTS:
                return Cumulative(clear_on_get=True)
            case StreamKind.LOG | StreamKind.DEVICE:
                return nxlog_for_stream(
                    self._instrument.streams.get(key.name), name=key.name
                )
            case StreamKind.MONITOR_EVENTS:
                return ToNXevent_data()
            case StreamKind.DETECTOR_EVENTS:
                return make_detector_event_preprocessor(
                    self._instrument, key.name, group_by_pixel=self._group_by_pixel
                )
            case StreamKind.AREA_DETECTOR:
                return Cumulative(clear_on_get=True)
            case StreamKind.LIVEDATA_CONTEXT:
                return LatestValueAccumulator()
            case _:
                return None
