# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Generic plumbing for f144 NXlog → Sciline parameter delivery.

The flow is:

- A live f144 stream is consumed off Kafka, adapted to a :class:`LogData`
  message, accumulated by ``ToNXlog`` into a cumulative ``DataArray`` with
  a ``time`` coord.
- :class:`~ess.livedata.handlers.stream_processor_workflow.StreamProcessorWorkflow`
  receives that cumulative payload as a ``context_keys`` entry and (if the
  Sciline key is a :class:`ValueLog` subclass) wraps it via ``key(values=raw)``
  before delegating to ``set_context``.
- Per-binding :class:`ValueLog` subclasses give the cumulative log a
  distinct Sciline node identity, so multiple bindings can coexist without
  colliding on a shared parameter key.

:class:`LogContextBinding` declares the binding at instrument level, and
:func:`compose_aux_sources` derives the spec's ``AuxSources`` from the
relevant bindings — scoped per-job by ``source_name``.

Specific binding *kinds* (e.g. :class:`DynamicTransformBinding`) live in
their own modules and patch their own providers; the routing layer here
is type-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ess.livedata.config.workflow_spec import (
    AuxSources,
    CombinedAuxSources,
    JobId,
)

from .stream_processor_workflow import ValueLog

if TYPE_CHECKING:
    from ess.livedata.config.instrument import Instrument


@dataclass(frozen=True, slots=True, kw_only=True)
class LogContextBinding:
    """Declares "f144 stream X feeds Sciline parameter K, scoped to sources S".

    Parameters
    ----------
    stream_name:
        Name of the f144 aux stream that supplies live samples. Must
        appear in :attr:`Instrument.f144_attribute_registry` (auto-derived
        from the binding when ``units`` is set).
    log_key:
        :class:`ValueLog` subclass used as the Sciline key for this
        binding. Each binding declares its own subclass so the per-binding
        log appears as a distinct, grep-able Sciline node.
    dependent_sources:
        Source names whose specs should subscribe to this stream. A spec
        whose ``source_names`` intersects this set picks up the binding's
        stream as an aux input; at render time, only jobs whose
        ``source_name`` is in this set receive it.
    units:
        Optional unit string. When provided, :class:`Instrument` auto-adds
        ``{stream_name: {'units': units}}`` to its f144 attribute registry,
        so per-instrument adoption need not splice the registry by hand.
    """

    stream_name: str
    log_key: type[ValueLog]
    dependent_sources: frozenset[str]
    units: str | None = None


class LogContextAuxSources(AuxSources):
    """Aux sources covering an instrument's log-context bindings.

    Inputs include every binding whose ``dependent_sources`` set intersects
    the spec's ``source_names``. ``render`` returns only the streams whose
    binding includes ``job_id.source_name`` in its ``dependent_sources``,
    rendered un-prefixed (these are global f144 streams shared across jobs).
    """

    def __init__(self, instrument: Instrument, source_names: list[str]) -> None:
        self._instrument = instrument
        selected = set(source_names)
        inputs: dict[str, str] = {
            b.stream_name: b.stream_name
            for b in instrument.log_context_bindings
            if b.dependent_sources & selected
        }
        super().__init__(dict(inputs))

    def render(
        self,
        job_id: JobId,
        selections: dict[str, str] | None = None,
    ) -> dict[str, str]:
        return {
            b.stream_name: b.stream_name
            for b in self._instrument.log_context_bindings
            if job_id.source_name in b.dependent_sources
        }


def compose_aux_sources(
    instrument: Instrument,
    source_names: list[str],
    caller_aux: AuxSources | None,
) -> AuxSources | None:
    """Merge caller-supplied aux sources with auto-derived log-context aux
    sources for the given source set."""
    components: list[AuxSources] = []
    if caller_aux is not None:
        components.append(caller_aux)
    if instrument.log_context_bindings:
        dyn = LogContextAuxSources(instrument, source_names)
        if dyn.inputs:
            components.append(dyn)
    if not components:
        return None
    if len(components) == 1:
        return components[0]
    return CombinedAuxSources(components)
