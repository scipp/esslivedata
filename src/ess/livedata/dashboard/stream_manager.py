# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Utilities for connecting subscribers to :py:class:`DataService`
"""

from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

from ess.livedata.config.workflow_spec import ResultKey

from .data_service import DataService
from .data_subscriber import DataSubscriber, MergingStreamAssembler, Pipe
from .extractors import UpdateExtractor

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

    def make_merging_stream(
        self,
        keys: Sequence[ResultKey] | dict[ResultKey, UpdateExtractor],
        assembler_factory: Callable[[set[ResultKey]], Any] = MergingStreamAssembler,
    ) -> P:
        """
        Create a merging stream for the given result keys.

        The pipe is created lazily on first trigger with correctly extracted data.

        Parameters
        ----------
        keys:
            Either a sequence of result keys (uses LatestValueExtractor for all)
            or a dict mapping keys to their specific UpdateExtractor instances.
        assembler_factory:
            Optional callable that creates an assembler from a set of keys.
            Use functools.partial to bind additional arguments (e.g., filter_fn).

        Returns
        -------
        :
            A pipe that will receive merged data updates for the given keys.
        """
        from .extractors import LatestValueExtractor

        if isinstance(keys, dict):
            # Dict provided: keys are dict keys, extractors are dict values
            keys_set = set(keys.keys())
            extractors = keys
        else:
            # Sequence provided: use default LatestValueExtractor for all keys
            keys_set = set(keys)
            extractors = {key: LatestValueExtractor() for key in keys_set}

        assembler = assembler_factory(keys_set)
        subscriber = DataSubscriber(assembler, self._pipe_factory, extractors)
        self.data_service.register_subscriber(subscriber)
        return subscriber.pipe

    def make_merging_stream_with_subscriber(
        self,
        keys: Sequence[ResultKey] | dict[ResultKey, UpdateExtractor],
        assembler_factory: Callable[[set[ResultKey]], Any] = MergingStreamAssembler,
    ) -> tuple[DataSubscriber, P]:
        """
        Create a merging stream and return both the subscriber and pipe.

        The pipe is created lazily on first trigger with correctly extracted data.
        Use this method when you need access to the subscriber for monitoring.

        Parameters
        ----------
        keys:
            Either a sequence of result keys (uses LatestValueExtractor for all)
            or a dict mapping keys to their specific UpdateExtractor instances.
        assembler_factory:
            Optional callable that creates an assembler from a set of keys.
            Use functools.partial to bind additional arguments (e.g., filter_fn).

        Returns
        -------
        :
            Tuple of (subscriber, pipe) where subscriber can be monitored for
            triggers and pipe receives merged data updates.
        """
        from .extractors import LatestValueExtractor

        if isinstance(keys, dict):
            # Dict provided: keys are dict keys, extractors are dict values
            keys_set = set(keys.keys())
            extractors = keys
        else:
            # Sequence provided: use default LatestValueExtractor for all keys
            keys_set = set(keys)
            extractors = {key: LatestValueExtractor() for key in keys_set}

        assembler = assembler_factory(keys_set)
        subscriber = DataSubscriber(assembler, self._pipe_factory, extractors)
        self.data_service.register_subscriber(subscriber)
        return subscriber, subscriber.pipe
