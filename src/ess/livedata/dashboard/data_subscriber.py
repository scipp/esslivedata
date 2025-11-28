# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from typing import Any, Generic, Protocol, TypeVar

from ess.livedata.config.workflow_spec import ResultKey
from ess.livedata.dashboard.data_service import DataServiceSubscriber
from ess.livedata.dashboard.extractors import UpdateExtractor


class PipeBase(Protocol):
    """
    Protocol for downstream pipes that can receive data from upstream pipes.
    """

    def send(self, data: Any) -> None:
        """
        Send data to the downstream pipe.

        Parameters
        ----------
        data:
            The data to be sent.
        """


class Pipe(PipeBase):
    """Protocol for holoviews pipes, which need to be initialized with data."""

    def __init__(self, data: Any) -> None:
        """
        Initialize the pipe with its data.

        Parameters
        ----------
        data:
            The initial data for the pipe.
        """


Key = TypeVar('Key', bound=Hashable)
P = TypeVar('P', bound=PipeBase)


class StreamAssembler(ABC, Generic[Key]):
    """
    Base class for assembling data from a data store.

    This class defines the interface for assembling data from a data store based on
    specific keys. Subclasses must implement the `assemble` method.
    """

    def __init__(self, keys: set[Key]) -> None:
        """
        Initialize the assembler with its data dependencies.

        Parameters
        ----------
        keys:
            The set of data keys this assembler depends on. This is used to determine
            when the assembler will be triggered to assemble data, i.e., updates to
            which keys in :py:class:`DataService` will trigger the assembler to run.
        """
        self._keys = keys

    @property
    def keys(self) -> set[Key]:
        """Return the set of data keys this assembler depends on."""
        return self._keys

    @abstractmethod
    def assemble(self, data: dict[Key, Any]) -> Any:
        """
        Assemble data from the provided dictionary.

        Parameters
        ----------
        data:
            A dictionary containing data keyed by ResultKey.

        Returns
        -------
        :
            The assembled data.
        """


class DataSubscriber(DataServiceSubscriber[Key], Generic[Key, P]):
    """Unified subscriber that uses a StreamAssembler to process data."""

    def __init__(
        self,
        assembler: StreamAssembler[Key],
        pipe_factory: Callable[[dict[Key, Any]], P],
        extractors: Mapping[Key, UpdateExtractor],
        on_first_data: Callable[[P], None] | None = None,
    ) -> None:
        """
        Initialize the subscriber with an assembler and pipe factory.

        Parameters
        ----------
        assembler:
            The assembler responsible for processing the data.
        pipe_factory:
            Factory function to create the pipe on first trigger.
        extractors:
            Mapping from keys to their UpdateExtractor instances.
        on_first_data:
            Optional callback invoked when first data arrives with the created pipe.
            Called after pipe creation with non-empty data.
        """
        self._assembler = assembler
        self._pipe_factory = pipe_factory
        self._pipe: P | None = None
        self._extractors = extractors
        self._on_first_data = on_first_data
        self._first_data_callback_invoked = False
        # Initialize parent class to cache keys
        super().__init__()

    @property
    def extractors(self) -> Mapping[Key, UpdateExtractor]:
        """Return extractors for obtaining data views."""
        return self._extractors

    @property
    def pipe(self) -> P:
        """Return the pipe (must be created by first trigger)."""
        if self._pipe is None:
            raise RuntimeError("Pipe not yet initialized - subscriber not triggered")
        return self._pipe

    def trigger(self, store: dict[Key, Any]) -> None:
        """
        Trigger the subscriber with the current data store.

        Parameters
        ----------
        store:
            The complete data store containing all available data.
        """
        data = {key: store[key] for key in self.keys if key in store}
        assembled_data = self._assembler.assemble(data)

        if self._pipe is None:
            # First trigger - always create pipe (even with empty data)
            self._pipe = self._pipe_factory(assembled_data)
        else:
            # Subsequent triggers - send to existing pipe
            self._pipe.send(assembled_data)

        # Invoke first-data callback when we have actual data for the first time
        if data and not self._first_data_callback_invoked and self._on_first_data:
            self._on_first_data(self._pipe)
            self._first_data_callback_invoked = True


class MergingStreamAssembler(StreamAssembler):
    """Assembler for merging data from multiple sources into a dict."""

    def assemble(self, data: dict[ResultKey, Any]) -> dict[ResultKey, Any]:
        # Sort keys to ensure deterministic ordering (important for color assignment)
        # Sort by (workflow_id, job_id, output_name) for consistent ordering
        sorted_keys = sorted(
            (key for key in self.keys if key in data),
            key=lambda k: (str(k.workflow_id), str(k.job_id), k.output_name or ''),
        )
        return {key: data[key] for key in sorted_keys}
