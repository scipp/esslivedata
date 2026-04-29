# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Synthesize a chopper-cascade trigger from chopper PV streams.

The synthesizer is a :class:`MessageSource` decorator: it wraps an
already-adapted source (i.e. domain-level :class:`Message` objects) and
injects synthetic primary ticks on the ``chopper_cascade`` logical stream.

For chopperless instruments (``v0`` scope) the synthesizer emits exactly one
vacuous "setpoints reached" tick on the first call to ``get_messages`` and
is otherwise a passthrough. Plateau detection on phase NXlogs and per-chopper
``phase_setpoint`` synthesis are deferred to v1.
"""

from __future__ import annotations

from collections.abc import Sequence

import structlog

from ..core.message import Message, MessageSource, StreamId, StreamKind
from ..core.timestamp import Timestamp
from ..handlers.accumulators import LogData
from ..handlers.lookup_table_workflow_specs import CHOPPER_CASCADE_SOURCE

logger = structlog.get_logger(__name__)

CHOPPER_CASCADE_STREAM = StreamId(kind=StreamKind.LOG, name=CHOPPER_CASCADE_SOURCE)


def _make_setpoints_reached_message() -> Message[LogData]:
    """Build a single ``LogData`` tick representing 'all setpoints reached'."""
    now = Timestamp.now()
    return Message(
        timestamp=now,
        stream=CHOPPER_CASCADE_STREAM,
        value=LogData(time=now.to_ns(), value=1),
    )


class ChopperSynthesizer(MessageSource[Message]):
    """Decorator that injects a synthetic chopper-cascade trigger.

    v0 (chopperless): emits one ``setpoints_reached`` tick on the first
    ``get_messages`` call, then forwards the wrapped source verbatim. Raw
    chopper PV streams (if any) are passed through unchanged — for
    chopperless instruments there are none, so the wrapper is a no-op
    after the initial tick.
    """

    def __init__(self, wrapped: MessageSource[Message]) -> None:
        self._wrapped = wrapped
        self._emitted_initial_tick = False

    def get_messages(self) -> Sequence[Message]:
        forwarded = list(self._wrapped.get_messages())
        if not self._emitted_initial_tick:
            self._emitted_initial_tick = True
            logger.info('chopper_cascade_initial_tick_emitted')
            return [_make_setpoints_reached_message(), *forwarded]
        return forwarded
