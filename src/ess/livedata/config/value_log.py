# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""Per-binding Sciline keys for f144 NXlog context streams.

:class:`ValueLog` is the typed Sciline-key wrapper that
:class:`~ess.livedata.handlers.stream_processor_workflow.StreamProcessorWorkflow`
delivers raw NXlog payloads through. Each chain-patch binding declares
its own subclass so multiple dynamic transforms can coexist on one
workflow without colliding on a shared Sciline parameter; the subclass
also carries the NeXus :attr:`transform_path` it patches.

The type lives in :mod:`config` (rather than next to
``StreamProcessorWorkflow`` in :mod:`handlers`) because
:class:`~ess.livedata.config.stream.ContextBinding` references it via its
``workflow_key`` field for chain-patching bindings; keeping the type
alongside the declaration record avoids ``config`` depending on
``handlers``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import scipp as sc


@dataclass(frozen=True, slots=True)
class ValueLog:
    """Typed Sciline-key wrapper around a cumulative ``ToNXlog`` payload.

    Subclass to create a distinct Sciline parameter per chain-patch
    binding. The class is the typed wrapper for an NXlog's
    ``value``-over-``time`` payload: :attr:`values` carries the
    cumulative timeseries (a ``DataArray`` with a ``time`` coord).

    Subclasses set :attr:`transform_path` (a class attribute) to the
    NeXus transformation chain entry whose ``value`` is patched from the
    latest sample of this log. :class:`~ess.livedata.config.stream.ContextBinding`
    entries pointing at a :class:`ValueLog` subclass as their
    ``workflow_key`` are routed via the fused per-component patched-chain
    provider instead of via direct ``set_context`` binding.

    :attr:`values` is the NXlog produced by ``ToNXlog`` — non-empty by the
    time it reaches the provider, because the JobManager context-stream
    gate (ADR 0002) blocks the workflow until the underlying f144 stream
    has produced a value.

    :class:`~ess.livedata.handlers.stream_processor_workflow.StreamProcessorWorkflow`
    detects subclasses of this type among its ``context_keys`` values and
    wraps the raw NXlog as ``key(values=raw)`` before delegating to
    ``set_context``.
    """

    #: NeXus transformation chain entry this log patches. ``None`` on the base
    #: class; chain-patching subclasses must set a concrete path.
    transform_path: ClassVar[str | None] = None

    values: sc.DataArray
