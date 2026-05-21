# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import scipp as sc
import structlog

from ..core.handler import Accumulator, JobBasedPreprocessorFactoryBase
from ..core.message import StreamId, StreamKind
from .accumulators import LogData
from .to_nxlog import ToNXlog, nxlog_for_stream
from .wavelength_lut_workflow_specs import CHOPPER_CASCADE_SOURCE
from .workflow_factory import Workflow

if TYPE_CHECKING:
    from ..config.instrument import Instrument

logger = structlog.get_logger(__name__)

#: Source names for synthetic LOG streams that are not real upstream f144s and
#: are therefore not declared in :attr:`Instrument.streams`. The
#: ``chopper_cascade`` tick from :class:`ChopperSynthesizer` is the only such
#: stream today; its value is a boolean "at setpoint" indicator and carries no
#: unit (distinct from ``'dimensionless'``, which a number with cancelled units
#: would have).
_SYNTHETIC_LOG_SOURCES: frozenset[str] = frozenset({CHOPPER_CASCADE_SOURCE})


class TimeseriesStreamProcessor(Workflow):
    def __init__(self) -> None:
        self._data: sc.DataArray | None = None
        self._last_returned_index = 0

    @staticmethod
    def create_workflow() -> Workflow:
        """Factory method for creating TimeseriesStreamProcessor."""
        return TimeseriesStreamProcessor()

    def accumulate(
        self, data: dict[Hashable, sc.DataArray], *, start_time: int, end_time: int
    ) -> None:
        if len(data) != 1:
            raise ValueError("Timeseries processor expects exactly one data item.")
        # Store the full cumulative data (including history from preprocessor)
        self._data = next(iter(data.values()))

    def finalize(self) -> dict[str, sc.DataArray]:
        if self._data is None:
            raise ValueError("No data has been added")

        # Return only new data since last finalize to avoid republishing full history
        current_size = self._data.sizes['time']
        if self._last_returned_index >= current_size:
            raise ValueError("No new data since last finalize")

        result = self._data['time', self._last_returned_index :]
        self._last_returned_index = current_size

        return {'delta': result}

    def clear(self) -> None:
        self._data = None
        self._last_returned_index = 0

    @property
    def context_keys(self) -> dict[str, Any]:
        return {}


class LogdataHandlerFactory(JobBasedPreprocessorFactoryBase[LogData, sc.DataArray]):
    """
    Factory for creating handlers for log data.

    Stream metadata (units etc.) is read from :attr:`Instrument.streams`. Sources
    in :data:`_SYNTHETIC_LOG_SOURCES` are accepted as unit-less streams without a
    corresponding stream record.
    """

    def __init__(self, *, instrument: Instrument) -> None:
        self._instrument = instrument

    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        match key.kind:
            case StreamKind.DEVICE:
                return nxlog_for_stream(self._instrument.streams.get(key.name))
            case StreamKind.LOG:
                accumulator = nxlog_for_stream(self._instrument.streams.get(key.name))
                if accumulator is not None:
                    return accumulator
                if key.name in _SYNTHETIC_LOG_SOURCES:
                    return ToNXlog(attrs={})
                logger.warning(
                    "No attributes found for source name '%s'. "
                    "Messages will be dropped.",
                    key.name,
                )
                return None
            case _:
                return None
