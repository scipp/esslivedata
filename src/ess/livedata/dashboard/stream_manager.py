# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Utilities for connecting subscribers to :py:class:`DataService`
"""

from collections.abc import Callable
from typing import Any

from ess.livedata.config.workflow_spec import ResultKey

from .data_service import DataService
from .data_subscriber import DataSubscriber


class StreamManager:
    """Manages data streams connecting DataService to callbacks."""

    def __init__(self, *, data_service: DataService):
        self.data_service = data_service

    def make_stream(
        self,
        keys_by_role: dict[str, list[ResultKey]],
        on_update: Callable[[], None],
        extractors: dict[ResultKey, Any] | None = None,
    ) -> DataSubscriber:
        """
        Create a data stream for the given result keys organized by role.

        Registers a subscriber whose ``on_update`` callback fires when any of
        the keys change. The consumer pulls role-grouped data via
        ``subscriber.assemble(data_service.snapshot(subscriber))``.

        Parameters
        ----------
        keys_by_role
            Dict mapping role names to lists of ResultKeys. For standard plots,
            this is {"primary": [keys...]}. For correlation plots, includes
            additional roles like "x_axis", "y_axis".
        on_update
            Callback invoked when any of the keys changed; see
            :py:class:`DataSubscriber`.
        extractors
            Optional dict mapping keys to UpdateExtractor instances. If not
            provided, uses LatestValueExtractor for all keys.

        Returns
        -------
        :
            The registered subscriber. Can be passed to
            DataService.unregister_subscriber() to stop receiving updates
            (e.g., when workflow restarts).
        """
        from .extractors import LatestValueExtractor

        # Flatten keys
        all_keys = [key for keys in keys_by_role.values() for key in keys]

        # Use provided extractors or create default
        if extractors is None:
            extractors = {key: LatestValueExtractor() for key in all_keys}

        subscriber = DataSubscriber(
            keys_by_role=keys_by_role,
            extractors=extractors,
            on_update=on_update,
        )
        self.data_service.register_subscriber(subscriber)
        return subscriber
