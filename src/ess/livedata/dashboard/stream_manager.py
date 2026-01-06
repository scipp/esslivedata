# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Utilities for connecting subscribers to :py:class:`DataService`
"""

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from ess.livedata.config.workflow_spec import ResultKey

from .data_service import DataService
from .data_subscriber import DataSubscriber, Pipe

P = TypeVar('P', bound=Pipe)


class StreamManager(Generic[P]):
    """Manages data streams connecting DataService to plotting pipes."""

    def __init__(
        self,
        *,
        data_service: DataService,
        pipe_factory: Callable[[Any], P],
    ):
        self.data_service = data_service
        self._pipe_factory = pipe_factory

    def make_stream(
        self,
        keys_by_role: dict[str, list[ResultKey]],
        on_first_data: Callable[[P], None] | None = None,
        extractors: dict[ResultKey, Any] | None = None,
    ) -> P:
        """
        Create a stream for the given result keys organized by role.

        The pipe is created lazily on first trigger with correctly extracted data.
        Assembly format depends on role count:
        - Single role: flat dict[ResultKey, data] (standard plotters)
        - Multiple roles: dict[str, dict[ResultKey, data]] (correlation plotters)

        Parameters
        ----------
        keys_by_role
            Dict mapping role names to lists of ResultKeys. For standard plots,
            this is {"primary": [keys...]}. For correlation plots, includes
            additional roles like "x_axis", "y_axis".
        on_first_data
            Optional callback invoked when first data arrives with the created pipe.
            Called when at least one key from each role has data.
        extractors
            Optional dict mapping keys to UpdateExtractor instances. If not
            provided, uses LatestValueExtractor for all keys.

        Returns
        -------
        :
            A pipe that will receive assembled data updates.
        """
        from .extractors import LatestValueExtractor

        # Flatten keys
        all_keys = [key for keys in keys_by_role.values() for key in keys]

        # Use provided extractors or create default
        if extractors is None:
            extractors = {key: LatestValueExtractor() for key in all_keys}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            pipe_factory=self._pipe_factory,
            extractors=extractors,
            on_first_data=on_first_data,
        )
        self.data_service.register_subscriber(subscriber)
        return subscriber.pipe
