# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Registry for tracking backend service worker health via heartbeats."""

from __future__ import annotations

import time
from collections.abc import Callable

import structlog

from ess.livedata.core.job import ServiceState, ServiceStatus

logger = structlog.get_logger(__name__)

# Timeout threshold for considering a worker stale (30 seconds in nanoseconds)
SERVICE_HEARTBEAT_TIMEOUT_NS = 30_000_000_000


def make_worker_key(status: ServiceStatus) -> str:
    """Create a unique key for a worker from its status."""
    return f"{status.instrument}:{status.namespace}:{status.worker_id}"


class ServiceRegistry:
    """
    Registry for tracking backend service worker health.

    Stores the latest heartbeat from each worker and provides methods for
    checking staleness and notifying subscribers of updates.
    """

    def __init__(
        self,
        *,
        heartbeat_timeout_ns: int = SERVICE_HEARTBEAT_TIMEOUT_NS,
    ) -> None:
        self._worker_statuses: dict[str, ServiceStatus] = {}
        self._worker_timestamps: dict[str, int] = {}
        self._update_subscribers: list[Callable[[], None]] = []
        self._heartbeat_timeout_ns = heartbeat_timeout_ns

    @property
    def worker_statuses(self) -> dict[str, ServiceStatus]:
        """Access to all stored worker statuses keyed by worker key."""
        return self._worker_statuses

    def register_update_subscriber(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when worker status is updated."""
        self._update_subscribers.append(callback)
        # Immediately notify the new subscriber of current state
        try:
            callback()
        except Exception as e:
            logger.error("Error in service status update callback: %s", e)

    def status_updated(self, status: ServiceStatus) -> None:
        """Update the stored worker status and record timestamp."""
        worker_key = make_worker_key(status)
        logger.debug("Worker status updated: %s -> %s", worker_key, status.state)
        self._worker_statuses[worker_key] = status
        self._worker_timestamps[worker_key] = time.time_ns()
        self._notify_update()

    def _notify_update(self) -> None:
        """Notify listeners about worker status updates."""
        logger.debug(
            "Worker statuses updated: %s workers tracked",
            len(self._worker_statuses),
        )

        for callback in self._update_subscribers:
            try:
                callback()
            except Exception as e:
                logger.error("Error in service status update callback: %s", e)

    def remove_worker(self, worker_key: str) -> None:
        """Remove a worker from tracking."""
        if worker_key in self._worker_statuses:
            del self._worker_statuses[worker_key]
        if worker_key in self._worker_timestamps:
            del self._worker_timestamps[worker_key]
        self._notify_update()

    def is_status_stale(self, worker_key: str) -> bool:
        """Check if a worker's status is stale (no recent heartbeat).

        Workers in terminal states (stopped) are never considered stale since
        they are not expected to send further heartbeats.

        Parameters
        ----------
        worker_key
            The worker key (instrument:namespace:worker_id) to check.

        Returns
        -------
        :
            True if status is older than heartbeat timeout and not in a
            terminal state. False for workers in terminal states.
        """
        if worker_key not in self._worker_timestamps:
            return True

        # Workers in terminal states are never stale - they stopped intentionally
        status = self._worker_statuses.get(worker_key)
        if status is not None and status.state == ServiceState.stopped:
            return False

        last_update = self._worker_timestamps[worker_key]
        current_time = time.time_ns()
        age_ns = current_time - last_update

        return age_ns > self._heartbeat_timeout_ns

    def get_stale_workers(self) -> list[str]:
        """Get list of worker keys that have stale status."""
        return [key for key in self._worker_statuses if self.is_status_stale(key)]

    def get_worker_uptime_seconds(self, worker_key: str) -> float | None:
        """Get worker uptime in seconds.

        Returns
        -------
        :
            Uptime in seconds, or None if worker not found.
        """
        if worker_key not in self._worker_statuses:
            return None
        status = self._worker_statuses[worker_key]
        current_time_ns = time.time_ns()
        uptime_ns = current_time_ns - status.started_at
        return uptime_ns / 1_000_000_000

    def get_last_seen_seconds_ago(self, worker_key: str) -> float | None:
        """Get how long ago the worker was last seen (in seconds).

        Returns
        -------
        :
            Seconds since last heartbeat, or None if worker not found.
        """
        if worker_key not in self._worker_timestamps:
            return None
        last_update = self._worker_timestamps[worker_key]
        current_time = time.time_ns()
        return (current_time - last_update) / 1_000_000_000
