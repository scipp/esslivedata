# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Batching of the Bokeh document mutations a single IOLoop pass makes."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import panel as pn
import structlog

if TYPE_CHECKING:
    from bokeh.document import Document

logger = structlog.get_logger(__name__)

_SLOW_UPDATE_S = 0.5
"""Duration above which one batch is reported as harmful to other sessions.

Every session on a dashboard process shares one IOLoop, so a batch this long
is time during which none of the others is served. Chosen above the steady-state
cost of a large grid, so what it reports is a grid being built or revealed rather
than the per-frame repaint.
"""

_depth = 0
"""Nesting depth, so one pass is reported once rather than once per batch."""


@contextmanager
def batched_update() -> Iterator[None]:
    """Batch the document events and model-graph recomputes of one pass.

    ``pn.io.hold()`` batches document change events so they are dispatched to
    the browser in one WebSocket flush, avoiding staggered rendering.

    The freeze batches Bokeh's model-graph recomputation. Without it, each
    operation that mutates the model graph (pipe.send, layout child changes)
    triggers a full BFS traversal of every model in the document via
    ``_pop_freeze`` -> ``recompute`` -> ``collect_models``, at O(models) cost.
    Holding the freeze counter above zero for the whole pass makes the inner
    freeze/unfreeze cycles (Panel's per-model ``freeze_doc``, HoloViews'
    ``hold_render``) no-ops, collapsing N recomputes into 1 -- or into none, see
    :func:`_frozen_models`.

    Nesting is a no-op: an inner batch is already covered by the outer one.

    Every path that blocks the loop with document work passes through here --
    the session tick and the tab reveal alike -- so this is also where that work
    is timed, and where a batch long enough to stall the other sessions on the
    process is reported.
    """
    global _depth
    doc = pn.state.curdoc
    outermost = _depth == 0
    _depth += 1
    start = time.monotonic()
    try:
        # The freeze is the inner context so that it recomputes before the hold
        # dispatches: a model added during the pass must be attached to the
        # document by the time the queued events are serialized.
        with pn.io.hold(), _frozen_models(doc):
            yield
    finally:
        _depth -= 1
        elapsed = time.monotonic() - start
        # Reported on the exception path too: a pass that blocks the loop and
        # then raises is the one most worth seeing.
        if outermost and elapsed >= _SLOW_UPDATE_S:
            logger.warning(
                'dashboard_slow_update',
                elapsed_seconds=round(elapsed, 3),
                session_id=_session_id(doc),
            )


def _session_id(doc: Document | None) -> str | None:
    """Identify the session a batch belongs to, if it has one.

    ``None`` off a server session, which is where the widget tests run.
    """
    context = getattr(doc, 'session_context', None)
    return None if context is None else context.id


@contextmanager
def _frozen_models(doc: Document | None) -> Iterator[None]:
    """Freeze the document's model graph, recomputing it only if it changed.

    ``doc.models.freeze()`` recomputes when it exits whether or not anything
    changed. A pass that mutates nothing -- the unconditional full pass, or any
    pass whose handlers find no work -- then pays an O(models) walk for nothing:
    tens of milliseconds on a document holding a plot grid, once a second or
    more, on the IOLoop every session on the server shares.

    So instead of Bokeh's ``freeze``, hold the freeze counter up ourselves and
    recompute at the end only if the pass could have changed which models are
    reachable from the roots. That is the case exactly when Bokeh called
    ``DocumentModelManager.invalidate`` -- its own signal that a property change
    touched model references, intercepted here because while frozen it does
    nothing -- or when the roots themselves changed, which Bokeh signals by
    freezing around the change rather than by invalidating.

    Data-only changes (``ColumnDataSource`` patches and streams) invalidate
    nothing by design, so a pass that merely pushes new data into existing
    glyphs now recomputes nothing either.
    """
    if doc is None:
        yield
        return
    models = doc.models
    if 'invalidate' in models.__dict__:
        # An enclosing batch already installed the hook below and holds the
        # freeze; re-installing here would unhook it again on exit.
        yield
        return
    invalidated = False

    def invalidate() -> None:
        nonlocal invalidated
        invalidated = True

    roots = doc.roots
    models.invalidate = invalidate  # type: ignore[method-assign]
    models._push_freeze()
    try:
        yield
    finally:
        del models.invalidate
        # Pop by hand: ``_pop_freeze`` would recompute unconditionally. Unlike
        # Bokeh's ``freeze`` this also runs on the exception path, so a raising
        # handler cannot leave the document frozen for good.
        models._freeze_count -= 1
        if models._freeze_count == 0 and (invalidated or doc.roots != roots):
            models.recompute()
