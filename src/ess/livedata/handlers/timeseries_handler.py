# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import scipp as sc
import structlog

from ..core.handler import JobBasedPreprocessorFactoryBase
from ..core.message import StreamId
from .accumulators import LogData
from .to_nxlog import ToNXlog
from .workflow_factory import Workflow

if TYPE_CHECKING:
    from ..config.instrument import Instrument

logger = structlog.get_logger(__name__)


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

    This factory creates a handler that accumulates log data and returns it as a
    DataArray.
    """

    def __init__(
        self,
        *,
        instrument: Instrument,
        attribute_registry: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize the LogdataHandlerFactory.

        Parameters
        ----------
        instrument:
            The name of the instrument.
        attribute_registry:
            A dictionary mapping source names to attributes. This provides essential
            attributes for the values and timestamps in the log data. Log messages do
            not contain this information, so it must be provided externally.
            The keys of the dictionary are source names, and the values are dictionaries
            containing the attributes as they would be found in the fields of an NXlog
            class in a NeXus file.
        """
        self._instrument = instrument
        if attribute_registry is None:
            self._attribute_registry = instrument.f144_attribute_registry
        else:
            self._attribute_registry = attribute_registry

    def make_preprocessor(self, key: StreamId) -> ToNXlog | None:
        source_name = key.name
        attrs = self._attribute_registry.get(source_name)
        if attrs is None:
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
