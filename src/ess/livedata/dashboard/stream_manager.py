# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Utilities for connecting subscribers to :py:class:`DataService`
"""

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from ess.livedata.config.workflow_spec import ResultKey

from .data_service import DataService
from .data_subscriber import DataSubscriber, MergingStreamAssembler, Pipe

P = TypeVar('P', bound=Pipe)


class StreamManager(Generic[P]):
    """Base class for managing data streams."""

    def __init__(
        self,
        *,
        data_service: DataService,
        pipe_factory: Callable[[dict[ResultKey, Any]], P],
    ):
        self.data_service = data_service
        self._pipe_factory = pipe_factory

    def make_merging_stream(self, items: dict[ResultKey, Any]) -> P:
        """Create a merging stream for the given set of data keys."""
        assembler = MergingStreamAssembler(set(items))
        pipe = self._pipe_factory(items)
        subscriber = DataSubscriber(assembler, pipe)
        self.data_service.register_subscriber(subscriber)
        return pipe

    def make_merging_stream_from_keys(self, keys: list[ResultKey]) -> P:
        """
        Create a merging stream for the given result keys, starting with no data.

        This is useful when you want to subscribe to keys that may not have data yet.
        The pipe is initialized with an empty dictionary, and will receive updates
        as data becomes available for the subscribed keys.

        Parameters
        ----------
        keys:
            List of result keys to subscribe to.

        Returns
        -------
        :
            A pipe that will receive merged data updates for the given keys.
        """
        assembler = MergingStreamAssembler(set(keys))
        pipe = self._pipe_factory({})
        subscriber = DataSubscriber(assembler, pipe)
        self.data_service.register_subscriber(subscriber)
        return pipe
