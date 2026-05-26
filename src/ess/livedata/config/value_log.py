# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""Per-binding Sciline keys for f144 NXlog context streams.

:class:`ValueLog` is the typed Sciline-key wrapper that
:class:`~ess.livedata.handlers.stream_processor_workflow.StreamProcessorWorkflow`
delivers raw NXlog payloads through. Each transformation context binding
declares its own subclass so multiple dynamic transforms can coexist on
one workflow without colliding on a shared Sciline parameter.

The type lives in :mod:`config` (rather than next to
``StreamProcessorWorkflow`` in :mod:`handlers`) because
:class:`~ess.livedata.config.stream.TransformationContext` references it
in its ``log_key`` field; keeping the type alongside the declaration
record avoids ``config`` depending on ``handlers``.
"""

from __future__ import annotations

from dataclasses import dataclass

import scipp as sc


@dataclass(frozen=True, slots=True)
class ValueLog:
    """Typed Sciline-key wrapper around a cumulative ``ToNXlog`` payload.

    Subclass to create a distinct Sciline parameter per stream. The class
    is the typed wrapper for an NXlog's ``value``-over-``time`` payload:
    :attr:`values` carries the cumulative timeseries (a ``DataArray`` with
    a ``time`` coord).

    :attr:`values` is ``None`` before the first ``set_context`` call —
    ``ess.reduce.streaming.StreamProcessor`` pre-sets every context key to
    ``None`` — otherwise it is the NXlog produced by ``ToNXlog``, possibly
    still empty if no f144 message has arrived yet.

    :class:`~ess.livedata.handlers.stream_processor_workflow.StreamProcessorWorkflow`
    detects subclasses of this type among its ``context_keys`` values and
    wraps the raw NXlog as ``key(values=raw)`` before delegating to
    ``set_context``.
    """

    values: sc.DataArray | None = None
