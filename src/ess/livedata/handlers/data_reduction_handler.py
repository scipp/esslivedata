# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from ..config.instrument import Instrument
from ..config.stream import Device, F144Stream
from ..core.handler import Accumulator, JobBasedPreprocessorFactoryBase
from ..core.message import StreamId, StreamKind
from .accumulators import Cumulative
from .group_by_pixel import GroupByPixel
from .to_device_log import ToDeviceLog
from .to_nxevent_data import ToNXevent_data
from .to_nxlog import ToNXlog


class ReductionHandlerFactory(JobBasedPreprocessorFactoryBase):
    """Factory for data reduction handlers."""

    def __init__(self, *, instrument: Instrument, group_by_pixel: bool = True) -> None:
        super().__init__(instrument=instrument)
        self._group_by_pixel = group_by_pixel

    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        match key.kind:
            case StreamKind.MONITOR_COUNTS:
                return Cumulative(clear_on_get=True)
            case StreamKind.LOG:
                stream = self._instrument.streams.get(key.name)
                if not isinstance(stream, F144Stream):
                    return None
                return ToNXlog(attrs={'units': stream.units})
            case StreamKind.DEVICE:
                device = self._instrument.streams.get(key.name)
                if not isinstance(device, Device):
                    return None
                return ToDeviceLog(
                    units=device.units,
                    has_target=device.target is not None,
                    has_settled=device.settled is not None,
                )
            case StreamKind.MONITOR_EVENTS:
                return ToNXevent_data()
            case StreamKind.DETECTOR_EVENTS:
                if self._group_by_pixel:
                    detector_number = self._instrument.get_detector_number(key.name)
                    return GroupByPixel(ToNXevent_data(), detector_number)
                return ToNXevent_data()
            case StreamKind.AREA_DETECTOR:
                return Cumulative(clear_on_get=True)
            case _:
                return None
