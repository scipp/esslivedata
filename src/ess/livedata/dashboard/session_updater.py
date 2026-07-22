# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
SessionUpdater - Per-session component that drives all widget updates.

This module implements the polling-based update model for multi-session support.
Each browser session has its own SessionUpdater that polls for changes from
shared services in the correct session context.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING

import panel as pn
import structlog

from .notification_queue import NotificationEvent, NotificationQueue
from .notifications import show_notification
from .session_registry import SessionId, SessionRegistry
from .widgets.heartbeat_widget import HeartbeatWidget

if TYPE_CHECKING:
    from bokeh.document import Document

logger = structlog.get_logger(__name__)


class SessionUpdater:
    """
    Per-session component that drives all widget updates.

    Each browser session creates its own SessionUpdater instance. The updater
    polls shared services (NotificationQueue) and runs custom handlers in its
    periodic callback, ensuring all session-bound components are updated
    in the correct session context.

    Plot updates are driven via custom handlers that call
    SessionLayer.update_pipe(), which uses the Presenter dirty flag
    mechanism for change detection.

    All widget updates (structural rebuilds via version-based change detection,
    status badges, timing, worker lists) use polling via custom handlers
    registered here, ensuring all updates run batched in the session context.

    Parameters
    ----------
    session_id:
        Unique identifier for this session.
    session_registry:
        Registry for session heartbeats and tracking.
    notification_queue:
        Shared queue for notifications.
    document:
        The session's Bokeh document, used to marshal document-mutating
        teardown onto the session's IOLoop when :meth:`cleanup` runs off it
        (the stale-session reaper). ``None`` in non-session contexts (tests),
        where teardown runs inline.
    """

    def __init__(
        self,
        *,
        session_id: SessionId,
        session_registry: SessionRegistry,
        notification_queue: NotificationQueue,
        username: str | None = None,
        document: Document | None = None,
    ) -> None:
        self._session_id = session_id
        self._session_registry = session_registry
        self._notification_queue = notification_queue
        self._document = document

        # Callbacks for custom updates (e.g., plot pipe polling)
        self._custom_handlers: list[Callable[[], None]] = []
        # Tier-2 teardown: sever shared state. Safe on any thread; run inline.
        self._cleanup_handlers: list[Callable[[], None]] = []
        # Tier-1 teardown: mutate Bokeh document state. Must run on the
        # session's IOLoop (marshalled via the document on the reaper path).
        self._document_teardown_handlers: list[Callable[[], None]] = []
        self._periodic_callback: pn.io.PeriodicCallback | None = None

        # Browser heartbeat mechanism using ReactiveHTML.
        # The widget's JavaScript increments a counter every 5 seconds.
        # We watch for changes to detect browser liveness.
        self._heartbeat_widget = HeartbeatWidget(interval_ms=5000)
        self._last_heartbeat_value = 0

        self._notification_queue.register_session(session_id)

        # Auto-register this session with the registry
        self._session_registry.register(session_id, self, username=username)

        logger.debug("SessionUpdater created for session %s", session_id)

    def set_periodic_callback(self, callback: pn.io.PeriodicCallback) -> None:
        """
        Store a reference to the periodic callback driving this updater.

        This allows ``cleanup`` to stop the callback when the session is
        destroyed or cleaned up as stale.

        Parameters
        ----------
        callback:
            The Panel periodic callback to stop on cleanup.
        """
        self._periodic_callback = callback

    def register_custom_handler(self, handler: Callable[[], None]) -> None:
        """
        Register a custom handler to be called during periodic updates.

        Custom handlers are called in the correct session context during
        the periodic update cycle. Use this for processing pending setups
        or other session-specific work (e.g., polling SessionLayer.update_pipe).

        Parameters
        ----------
        handler:
            Callback to invoke during periodic updates.
        """
        self._custom_handlers.append(handler)

    def unregister_custom_handler(self, handler: Callable[[], None]) -> None:
        """Unregister a custom handler."""
        if handler in self._custom_handlers:
            self._custom_handlers.remove(handler)

    def register_cleanup_handler(self, handler: Callable[[], None]) -> None:
        """
        Register a tier-2 teardown handler that severs shared state.

        Use this for releasing per-session resources held in shared state (e.g.
        a Kafka producer, orchestrator viewer/interest tokens, lifecycle
        subscriptions). Handlers run inline from :meth:`cleanup`, which fires
        both on clean browser disconnect (``on_session_destroyed``) and via the
        heartbeat-based stale-session reaper, so resources are released even
        when Panel's ``on_session_destroyed`` does not fire. The reaper runs on
        a background thread, so handlers must be safe to call there and must not
        mutate Bokeh document state (use :meth:`register_document_teardown_handler`
        for that).

        Parameters
        ----------
        handler:
            Callback to invoke on session teardown.
        """
        self._cleanup_handlers.append(handler)

    def register_document_teardown_handler(self, handler: Callable[[], None]) -> None:
        """
        Register a tier-1 teardown handler that mutates Bokeh document state.

        Use this for disposing session-bound widgets (e.g. breaking Bokeh-tool
        reference cycles). Bokeh document state may only be mutated on the
        session's IOLoop, so on the stale-session reaper path these handlers are
        scheduled onto that IOLoop via the session document rather than run on
        the calling thread. They may therefore never run if the session's
        document is already gone; tier-2 handlers must not depend on them.

        Parameters
        ----------
        handler:
            Callback to invoke on session teardown, on the session's IOLoop.
        """
        self._document_teardown_handlers.append(handler)

    def periodic_update(self) -> None:
        """
        Called from this session's periodic callback.

        Polls shared services for changes and runs custom handlers
        in a single batched UI update.

        Heartbeats are sent to the registry only when we have evidence that
        the browser is still connected (via the browser heartbeat widget).
        """
        # Check if browser has sent a heartbeat (widget value changed)
        self._check_browser_heartbeat()

        # Poll for notifications
        notifications = self._poll_notifications()

        with self._batched_update():
            self._show_notifications(notifications)

            # Run custom handlers (e.g., plot pipe polling)
            # in correct session context
            for handler in self._custom_handlers:
                try:
                    handler()
                except Exception:
                    logger.exception(
                        "Error in custom handler for session %s", self._session_id
                    )

    @contextmanager
    def _batched_update(self) -> Iterator[None]:
        """Batch UI updates using hold + freeze.

        ``pn.io.hold()`` batches document change events so they are dispatched
        to the browser in one WebSocket flush, avoiding staggered rendering.

        ``doc.models.freeze()`` batches Bokeh model-graph recomputation.
        Without it, each operation that mutates the model graph (pipe.send,
        layout child changes) triggers a full BFS traversal of every model in
        the document via ``_pop_freeze`` → ``recompute`` → ``collect_models``,
        at O(M) cost. The outer freeze keeps the counter above zero so that
        inner freeze/unfreeze cycles (e.g. HoloViews ``hold_render``) are
        no-ops, collapsing N recomputes into 1.
        """
        doc = pn.state.curdoc
        freeze = doc.models.freeze() if doc is not None else nullcontext()
        with pn.io.hold(), freeze:
            yield

    def _poll_notifications(self) -> list[NotificationEvent]:
        """Poll NotificationQueue for new events."""
        return self._notification_queue.get_new_events(self._session_id)

    def _show_notifications(self, notifications: list[NotificationEvent]) -> None:
        """Show notifications using Panel's notification system."""
        for event in notifications:
            show_notification(event)

    def cleanup(self, *, defer_document_teardown: bool = False) -> None:
        """
        Tear down the session in two tiers.

        Tier 2 (shared-state severing) runs inline and is safe on any thread:
        the notification-queue slot is released and cleanup handlers
        (:meth:`register_cleanup_handler`) run. This is the leak fix and does
        not depend on tier 1.

        Tier 1 (Bokeh document mutation) stops the periodic callback and runs
        the document-teardown handlers (:meth:`register_document_teardown_handler`).
        Document state may only be mutated on the owning session's IOLoop, so on
        the stale-session reaper path (``defer_document_teardown=True``) it is
        scheduled via the captured document's ``add_next_tick_callback`` instead
        of running on the calling thread. Scheduling into an already-destroyed
        document is tolerated: tier 2 has already severed everything.

        Parameters
        ----------
        defer_document_teardown:
            Schedule tier 1 onto the session's IOLoop instead of running it
            inline. Set by the background stale-session reaper, which is not on
            the session's IOLoop; the clean ``on_session_destroyed`` path runs
            on the IOLoop and leaves this ``False``.
        """
        # Tier 2: sever shared state. Safe on any thread.
        self._notification_queue.unregister_session(self._session_id)
        self._run_handlers(self._cleanup_handlers)
        self._cleanup_handlers.clear()

        # Tier 1: Bokeh document mutation. Must run on the session's IOLoop.
        if defer_document_teardown and self._document is not None:
            try:
                self._document.add_next_tick_callback(self._document_teardown)
            except Exception:
                logger.debug(
                    "Skipping deferred document teardown for session %s: "
                    "document already gone",
                    self._session_id,
                )
        else:
            self._document_teardown()

        logger.debug("SessionUpdater cleaned up for session %s", self._session_id)

    def _document_teardown(self) -> None:
        """Stop the periodic callback and dispose document-bound widgets.

        Mutates Bokeh document state, so it must run on the session's IOLoop.
        """
        if self._periodic_callback is not None:
            self._periodic_callback.stop()
        self._run_handlers(self._document_teardown_handlers)
        self._document_teardown_handlers.clear()
        self._custom_handlers.clear()

    def _run_handlers(self, handlers: Iterable[Callable[[], None]]) -> None:
        """Run teardown handlers, logging and swallowing any exception."""
        for handler in handlers:
            try:
                handler()
            except Exception:
                logger.exception(
                    "Error in teardown handler for session %s", self._session_id
                )

    def _check_browser_heartbeat(self) -> None:
        """
        Check if the browser has sent a heartbeat via the widget.

        The browser-side JavaScript increments the heartbeat counter value
        every 5 seconds. If the value has changed since our last check,
        the browser is alive and we send a heartbeat to the registry.
        """
        current_value = self._heartbeat_widget.counter
        if current_value != self._last_heartbeat_value:
            self._last_heartbeat_value = current_value
            self._session_registry.heartbeat(self._session_id)
            logger.debug(
                "Browser heartbeat received for session %s (counter=%d)",
                self._session_id,
                current_value,
            )

    @property
    def heartbeat_widget(self) -> HeartbeatWidget:
        """
        Get the heartbeat widget.

        This must be added to the page layout for browser heartbeats to work.
        The widget is invisible but its JavaScript runs to send periodic
        heartbeats from the browser.
        """
        return self._heartbeat_widget

    @property
    def session_id(self) -> SessionId:
        """Get this session's ID."""
        return self._session_id
