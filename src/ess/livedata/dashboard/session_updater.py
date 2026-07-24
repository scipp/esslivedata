# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
SessionUpdater - Per-session component that drives all widget updates.

This module implements the update model for multi-session support. Each browser
session has its own SessionUpdater, woken by the shared WakeupHub after commits
and backed by a slow housekeeping tick, which pulls changes from shared services
in the correct session context.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import TYPE_CHECKING

import panel as pn
import structlog

from .notification_queue import NotificationEvent, NotificationQueue
from .notifications import show_notification
from .session_registry import SessionId, SessionRegistry
from .widgets.heartbeat_widget import HeartbeatWidget

if TYPE_CHECKING:
    from bokeh.document import Document

    from .wakeup_hub import WakeupHub

logger = structlog.get_logger(__name__)

# Cadence of the unconditional full pass within housekeeping ticks. It is the
# clock for wall-clock-driven displays without a change counter (status
# staleness at a 60 s threshold, heartbeat readouts) and bounds how long a
# predicate hole can keep a widget stale. Time-based work on a finer grain
# (freshness-pill stall aging at 2 s) must be encoded in its handler's
# ``has_work`` predicate instead.
_FULL_PASS_INTERVAL_S = 5.0


class SessionUpdater:
    """
    Per-session component that drives all widget updates.

    Each browser session creates its own SessionUpdater instance. All widget
    updates (structural rebuilds via version-based change detection, plot-data
    flushes, status badges) run through custom handlers registered here, in
    the session's context and batched in a single hold+freeze pass.

    Ticks come in two flavors:

    - **Wake ticks** (:meth:`wake`), scheduled by the shared
      :class:`WakeupHub` right after data commits or shared version bumps.
      They run only handlers whose ``has_work`` predicate fires, and skip the
      hold+freeze batch entirely when none does — a wake meant for another
      session's tab costs no model-graph recompute.
    - **Housekeeping ticks** (:meth:`housekeeping_tick`), driven by a slow
      periodic callback. They are predicate-gated like wake ticks — so an
      idle tick costs no batch — except that every ``_FULL_PASS_INTERVAL_S``
      one runs every handler unconditionally: the clock for wall-clock-driven
      updates no version counter can signal, and the safety net bounding
      predicate holes (ADR 0007's degradation to the poll).

    Parameters
    ----------
    session_id:
        Unique identifier for this session.
    session_registry:
        Registry for session heartbeats and tracking.
    notification_queue:
        Shared queue for notifications.
    username:
        Authenticated username for this session, if available.
    document:
        The session's Bokeh document, used to marshal document-mutating
        teardown onto the session's IOLoop when :meth:`cleanup` runs off it
        (the stale-session reaper), and to schedule wake ticks. ``None`` in
        non-session contexts (tests), where teardown runs inline.
    wakeup_hub:
        Shared hub scheduling wake ticks across threads. When provided
        together with ``document``, this session registers for wakes and
        unregisters on cleanup. ``None`` leaves the session purely
        poll-driven.
    clock:
        Callable returning monotonic seconds, paces the full-pass interval.
        Injectable for deterministic tests.
    on_load:
        Scheduler deferring a callback until the browser session has loaded
        (websocket attached); defaults to ``pn.state.onload``, which runs the
        callback inline outside a server session. Wake registration goes
        through it — see the comment at the registration site. Injectable for
        deterministic tests.
    """

    def __init__(
        self,
        *,
        session_id: SessionId,
        session_registry: SessionRegistry,
        notification_queue: NotificationQueue,
        username: str | None = None,
        document: Document | None = None,
        wakeup_hub: WakeupHub | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_load: Callable[[Callable[[], None]], None] = pn.state.onload,
    ) -> None:
        self._session_id = session_id
        self._session_registry = session_registry
        self._notification_queue = notification_queue
        self._document = document
        self._wakeup_hub = wakeup_hub
        self._clock = clock
        self._severed = False
        # -inf so the session's first housekeeping tick is a full pass.
        self._last_full_pass = float('-inf')

        # Custom update handlers with their optional wake-tick work predicates.
        self._custom_handlers: list[
            tuple[Callable[[], None], Callable[[], bool] | None]
        ] = []
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

        # Wake delivery must not start before the browser has attached: with
        # data streaming, a wake tick would fire in the window between the
        # initial HTML render and the client's websocket sync, and any models
        # it adds (e.g. the first poll building all cell widgets) are missing
        # from the client's snapshot and never re-sent — the session renders
        # permanently empty while the server side looks healthy. Hence
        # registration is deferred until the session is loaded.
        if self._wakeup_hub is not None and self._document is not None:
            on_load(self._register_for_wakes)

        logger.debug("SessionUpdater created for session %s", session_id)

    def _register_for_wakes(self) -> None:
        """Register with the wakeup hub once the browser session has loaded.

        Skipped when cleanup already ran (session destroyed before load), so
        the deferred registration cannot resurrect a torn-down session.
        """
        if self._wakeup_hub is None:
            raise RuntimeError("Wakeup hub not initialized")
        if self._document is None:
            raise RuntimeError("Document not initialized")
        if not self._severed:
            self._wakeup_hub.register(self._session_id, self._document, self.wake)

    @property
    def severed(self) -> bool:
        """True once :meth:`cleanup` ran, i.e. the session is being torn down.

        Load-deferred setup must consult this before touching the session: the
        browser may never connect, and the stale-session reaper can tear the
        session down while its ``onload`` callbacks are still queued.
        """
        return self._severed

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

    def register_custom_handler(
        self,
        handler: Callable[[], None],
        *,
        has_work: Callable[[], bool] | None = None,
    ) -> None:
        """
        Register a custom handler to be called during update ticks.

        Custom handlers are called in the correct session context, inside the
        batched hold+freeze pass. Housekeeping ticks run every handler; wake
        ticks run only handlers whose ``has_work`` predicate returns True.

        Parameters
        ----------
        handler:
            Callback to invoke during update ticks.
        has_work:
            Cheap predicate telling a gated tick (wake or intermediate
            housekeeping) whether ``handler`` has pending visible work. Must
            be a superset signal of the handler's internal change gates for
            everything that should react faster than the full-pass cadence;
            include a time term for sub-``_FULL_PASS_INTERVAL_S`` wall-clock
            work (e.g. freshness stall aging). Anything it misses is bounded
            by the unconditional full pass. ``None`` means the handler has no
            cheap change signal and runs on full passes only.
        """
        self._custom_handlers.append((handler, has_work))

    def unregister_custom_handler(self, handler: Callable[[], None]) -> None:
        """Unregister a custom handler."""
        # Equality, not identity: ``updater.register_custom_handler(w.refresh)``
        # followed by ``updater.unregister_custom_handler(w.refresh)`` passes two
        # distinct bound-method objects that compare equal.
        self._custom_handlers = [
            (h, p) for h, p in self._custom_handlers if h != handler
        ]

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

    def housekeeping_tick(self) -> None:
        """
        Housekeeping tick, called from this session's periodic callback.

        Predicate-gated like a wake tick, except that every
        ``_FULL_PASS_INTERVAL_S`` it runs every handler unconditionally: the
        clock for wall-clock-driven updates no version counter signals, and
        the bound on how long a predicate hole can keep a widget stale.
        The gated ticks in between exist to evaluate predicate *time* terms:
        wakes fire only on events, so state defined by the absence of events
        (a stalled stream aging a freshness pill) is noticed only because
        this tick polls the predicates on a fixed clock.

        Heartbeats are sent to the registry only when we have evidence that
        the browser is still connected (via the browser heartbeat widget).
        """
        # Check if browser has sent a heartbeat (widget value changed)
        self._check_browser_heartbeat()
        # Re-arm wake delivery: a scheduled wake that never dispatched would
        # otherwise leave this session's pending flag set for good.
        if self._wakeup_hub is not None:
            self._wakeup_hub.clear_pending(self._session_id)
        now = self._clock()
        full = now - self._last_full_pass >= _FULL_PASS_INTERVAL_S
        if full:
            self._last_full_pass = now
        self._tick(full=full)

    def wake(self) -> None:
        """Wake tick, scheduled by the :class:`WakeupHub` on shared changes.

        Runs only handlers whose ``has_work`` predicate fires; when none does
        and no notifications are pending, returns without entering the
        hold+freeze batch, so a wake meant for another session's tab costs no
        model-graph recompute.
        """
        self._tick(full=False)

    def request_tick(self, *, full: bool = False) -> None:
        """Schedule a tick on this session's own IOLoop.

        For in-session events with no shared version bump (e.g. a tab switch)
        whose visible effect should not wait for the housekeeping tick.

        Parameters
        ----------
        full:
            Run every handler instead of only those whose predicate fires. Set
            by events that change *what is visible* rather than what changed:
            a handler skipped while its tab was hidden holds content frozen
            since it was last shown, and no ``has_work`` predicate can see
            that, since nothing in the shared state changed.
        """
        tick = partial(self._tick, full=full)
        if self._document is None:
            tick()
            return
        try:
            self._document.add_next_tick_callback(tick)
        except Exception:
            # Destroyed document racing teardown; nothing left to update.
            logger.debug(
                "Dropping requested tick for session %s: scheduling failed",
                self._session_id,
            )

    def _tick(self, *, full: bool) -> None:
        notifications = self._poll_notifications()
        handlers = [
            handler
            for handler, has_work in self._custom_handlers
            if full or (has_work is not None and self._check_work(has_work))
        ]
        if not notifications and not handlers:
            return

        with self._batched_update():
            self._show_notifications(notifications)
            for handler in handlers:
                try:
                    handler()
                except Exception:
                    logger.exception(
                        "Error in custom handler for session %s", self._session_id
                    )

    def _check_work(self, has_work: Callable[[], bool]) -> bool:
        """Evaluate a work predicate; a raising predicate counts as work.

        Running the handler on a broken predicate keeps updates flowing (the
        handler itself is exception-guarded) instead of silently degrading the
        widget to the full-pass cadence.
        """
        try:
            return has_work()
        except Exception:
            logger.exception(
                "Error in has_work predicate for session %s", self._session_id
            )
            return True

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
        self._severed = True
        if self._wakeup_hub is not None:
            self._wakeup_hub.unregister(self._session_id)
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
