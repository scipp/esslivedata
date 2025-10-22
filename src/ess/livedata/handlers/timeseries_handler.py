# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import scipp as sc

from ..core.handler import JobBasedPreprocessorFactoryBase
from ..core.message import StreamId
from .accumulators import LogData
from .to_nxlog import ToNXlog
from .workflow_factory import Workflow

if TYPE_CHECKING:
    from ..config.instrument import Instrument


class TimeseriesStreamProcessor(Workflow):
    def __init__(self) -> None:
        self._data: sc.DataArray | None = None

    @staticmethod
    def create_workflow() -> Workflow:
        """Factory method for creating TimeseriesStreamProcessor."""
        return TimeseriesStreamProcessor()

    def accumulate(self, data: dict[Hashable, sc.DataArray]) -> None:
        if len(data) != 1:
            raise ValueError("Timeseries processor expects exactly one data item.")
        # Just store the data for forwarding
        self._data = next(iter(data.values()))

    def finalize(self) -> dict[str, sc.DataArray]:
        if self._data is None:
            raise ValueError("No data has been added")
        return {'cumulative': self._data}

    def clear(self) -> None:
        self._data = None


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
        logger: logging.Logger | None = None,
        attribute_registry: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize the LogdataHandlerFactory.

        Parameters
        ----------
        instrument:
            The name of the instrument.
        logger:
            The logger to use for logging messages.
        attribute_registry:
            A dictionary mapping source names to attributes. This provides essential
            attributes for the values and timestamps in the log data. Log messages do
            not contain this information, so it must be provided externally.
            The keys of the dictionary are source names, and the values are dictionaries
            containing the attributes as they would be found in the fields of an NXlog
            class in a NeXus file.
        """

        self._logger = logger or logging.getLogger(__name__)
        self._instrument = instrument
        if attribute_registry is None:
            self._attribute_registry = instrument.f144_attribute_registry
        else:
            self._attribute_registry = attribute_registry

    def make_preprocessor(self, key: StreamId) -> ToNXlog | None:
        source_name = key.name
        attrs = self._attribute_registry.get(source_name)
        if attrs is None:
            self._logger.warning(
                "No attributes found for source name '%s'. Messages will be dropped.",
                source_name,
            )
            return None

        try:
            return ToNXlog(attrs=attrs)
        except Exception:
            self._logger.exception(
                "Failed to create NXlog for source name '%s'. "
                "Messages will be dropped.",
                source_name,
            )
            return None
