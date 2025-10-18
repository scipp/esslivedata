# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from typing import Any, Generic, Protocol, TypeVar

from ess.livedata.config.workflow_spec import ResultKey


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


class DataSubscriber(Generic[Key]):
    """Unified subscriber that uses a StreamAssembler to process data."""

    def __init__(self, assembler: StreamAssembler[Key], pipe: PipeBase) -> None:
        """
        Initialize the subscriber with an assembler and pipe.

        Parameters
        ----------
        assembler:
            The assembler responsible for processing the data.
        pipe:
            The pipe to send assembled data to.
        """
        self._assembler = assembler
        self._pipe = pipe

    @property
    def keys(self) -> set[Key]:
        """Return the set of data keys this subscriber depends on."""
        return self._assembler.keys

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
        self._pipe.send(assembled_data)


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


class FilteredMergingStreamAssembler(MergingStreamAssembler):
    """
    Assembler that filters data based on a predicate function before merging.

    This assembler extends MergingStreamAssembler by applying a filter predicate
    to determine which data items should be included in the assembled result.
    Useful for scenarios where data keys are subscribed to eagerly (e.g., ROI indices
    0-2), but only a subset should be displayed at any given time.
    """

    def __init__(self, keys: set[Key], filter_fn: Callable[[Key], bool]) -> None:
        """
        Initialize the filtered assembler.

        Parameters
        ----------
        keys:
            The set of data keys this assembler depends on.
        filter_fn:
            A callable that takes a key and returns True if the data should be included.
        """
        super().__init__(keys)
        self._filter_fn = filter_fn

    def assemble(self, data: dict[ResultKey, Any]) -> dict[ResultKey, Any]:
        """
        Assemble data after applying the filter predicate.

        Parameters
        ----------
        data:
            A dictionary containing data keyed by ResultKey.

        Returns
        -------
        :
            The filtered and assembled data dictionary.
        """
        # Filter data based on predicate before calling parent's assemble
        filtered_data = {k: v for k, v in data.items() if self._filter_fn(k)}
        return super().assemble(filtered_data)
