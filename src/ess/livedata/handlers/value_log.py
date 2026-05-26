# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""Per-binding Sciline keys for f144 NXlog context streams.

:class:`ValueLog` is the typed Sciline-key wrapper that
:class:`~ess.livedata.handlers.stream_processor_workflow.StreamProcessorWorkflow`
delivers raw NXlog payloads through. Each chain-patch binding declares
its own subclass so multiple dynamic transforms can coexist on one
workflow without colliding on a shared Sciline parameter.

:func:`synthesise_provider` builds a Sciline provider function with
explicit named typed positional parameters via ``exec``/``compile``.
Sciline introspects providers with ``inspect.getfullargspec`` (which
ignores ``__signature__``), so producing N named parameters requires
building a real code object — the same technique ``dataclasses``,
``attrs``, and ``namedtuple`` use to generate ``__init__``.

These primitives live in their own module to avoid the import cycle
that would arise from putting them next to ``StreamProcessorWorkflow``
(``config.workflow_spec`` needs ``ValueLog`` for pydantic schema
resolution, but ``stream_processor_workflow`` imports from
``handlers.workflow_factory`` which imports from
``config.workflow_spec``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


def synthesise_provider(
    name: str,
    impl: Callable[..., Any],
    annotations: dict[str, Any],
) -> Any:
    """Synthesise a Sciline provider with explicit named positional parameters.

    Returns a function ``name(p1, p2, ...)`` whose ``__annotations__`` come
    from ``annotations`` (with ``'return'`` consumed as the return type) and
    whose body delegates to ``impl(p1, p2, ...)``.

    Sciline introspects providers via ``inspect.getfullargspec``, which
    reads the underlying ``__code__`` and ignores ``__signature__``;
    producing N named typed positional parameters therefore requires
    building a real function via ``exec``/``compile``. Same technique
    ``dataclasses`` and ``namedtuple`` use to generate ``__init__``.
    Callers are responsible for constructing safe parameter names — no
    external string should reach the template.
    """
    params = [n for n in annotations if n != 'return']
    arg_list = ', '.join(params)
    src = f"def {name}({arg_list}):\n    return _impl({arg_list})\n"
    ns: dict[str, Any] = {'_impl': impl}
    exec(src, ns)  # noqa: S102
    fn = ns[name]
    fn.__annotations__ = dict(annotations)
    return fn
