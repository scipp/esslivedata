# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import scipp as sc
import structlog

from ..config.stream import F144Stream
from ..core.handler import JobBasedPreprocessorFactoryBase
from ..core.message import StreamId
from .accumulators import LogData
from .to_nxlog import ToNXlog
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


class LogdataHandlerFactory(JobBasedPreprocessorFactoryBase[LogData, sc.DataArray]):
    """
    Factory for creating handlers for log data.

    Stream metadata (units etc.) is read from :attr:`Instrument.streams`. Sources
    in :data:`_SYNTHETIC_LOG_SOURCES` are accepted as unit-less streams without a
    corresponding stream record.
    """

    def __init__(self, *, instrument: Instrument) -> None:
        self._instrument = instrument

    def make_preprocessor(self, key: StreamId) -> ToNXlog | None:
        source_name = key.name
        stream = self._instrument.streams.get(source_name)
        if isinstance(stream, F144Stream):
            attrs: dict[str, str | None] = {'units': stream.units}
        elif source_name in _SYNTHETIC_LOG_SOURCES:
            attrs = {}
        else:
            logger.warning(
                "No attributes found for source name '%s'. Messages will be dropped.",
                source_name,
            )
            return None

        try:
            return ToNXlog(attrs=attrs)
        except Exception:
            logger.exception(
                "Failed to create NXlog for source name '%s'. "
                "Messages will be dropped.",
                source_name,
            )
            return None
