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
        on_first_data: Callable[[P], None] | None = None,
        ready_condition: Callable[[set[ResultKey]], bool] | None = None,
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
        on_first_data:
            Optional callback invoked when first data arrives with the created pipe.
            Called after pipe creation with non-empty data.
        ready_condition:
            Optional callable that determines when on_first_data should fire.
            Receives the set of keys that have data. If None, fires when any
            data is available (existing behavior). For multi-source layers,
            LayerSubscription provides a condition requiring data from each
            DataSourceConfig.

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
        subscriber = DataSubscriber(
            assembler,
            self._pipe_factory,
            extractors,
            on_first_data,
            ready_condition,
        )
        self.data_service.register_subscriber(subscriber)
        return subscriber.pipe
