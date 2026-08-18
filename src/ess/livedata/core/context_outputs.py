# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Republication of designated workflow outputs as context input streams.

A workflow output designated in :attr:`WorkflowSpec.context_outputs` is
republished on a dedicated Kafka topic under a stable *stream name*, where it
becomes an ordinary context input that other workflows bind with a
``ContextBinding`` (ADR 0003). This is the feedback edge of ADR 0010: the
wavelength lookup table is computed by one job and consumed by the reduction
jobs of the same instrument.

The mirror is modelled on the NICOS derived-device path
(:mod:`ess.livedata.core.nicos_devices`, ADR 0006) and shares two of its
properties, for the same reasons:

- **A dedicated topic**, not ``livedata_data``. A backend service must not
  subscribe to every detector image in the facility in order to receive a
  lookup table.
- **Stream names carry no job identity.** A ``ContextBinding`` is declared at
  import time and cannot know a job number. Excluding it is also what lets a
  relaunched producer transparently resume feeding its consumers.

It differs in what happens downstream: a mirrored device is publish-only,
whereas a context output is consumed again by this system, so its rendered
stream name shares a namespace with device and motion stream names. Names are
therefore prefixed by convention (``wavelength_lut/…``) and checked for
collisions across the whole registry at startup.
"""

from __future__ import annotations

from collections.abc import Mapping

import scipp as sc

from ..config.workflow_spec import WorkflowId, WorkflowSpec
from .job import JobResult
from .message import Message, StreamId, StreamKind
from .timestamp import Timestamp


class ContextOutputError(ValueError):
    """Raised when ``context_outputs`` declarations cannot be resolved."""


def resolve_context_streams(
    registry: Mapping[WorkflowId, WorkflowSpec],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Resolve every ``context_outputs`` declaration in a workflow registry.

    Renders each declared stream-name template once per source name the spec
    declares, and checks that no two declarations render the same name. The
    check spans the whole registry because the rendered names share one
    namespace: two specs publishing the same name would interleave silently at
    the consumer.

    Parameters
    ----------
    registry:
        Workflow registry mapping :class:`WorkflowId` to :class:`WorkflowSpec`.
        An instrument's ``workflow_factory`` satisfies this mapping.

    Returns
    -------
    :
        Mapping from ``(workflow_id, source_name)`` to the
        ``(output_name, stream_name)`` pairs that job republishes.

    Raises
    ------
    ContextOutputError:
        If a template has an unknown placeholder, or if two declarations render
        the same stream name.
    """
    resolved: dict[tuple[str, str], list[tuple[str, str]]] = {}
    seen: dict[str, tuple[str, str, str]] = {}
    for workflow_id, spec in registry.items():
        for output_name, template in spec.context_outputs.items():
            for source_name in spec.source_names:
                try:
                    stream_name = template.format(source_name=source_name)
                except (KeyError, IndexError) as exc:
                    raise ContextOutputError(
                        f"Invalid context_outputs template {template!r} on "
                        f"{workflow_id}: unknown placeholder {exc}"
                    ) from exc
                key = (str(workflow_id), source_name, output_name)
                if (previous := seen.get(stream_name)) is not None:
                    raise ContextOutputError(
                        f"Duplicate context stream name {stream_name!r} rendered "
                        f"by {previous} and {key}"
                    )
                seen[stream_name] = key
                resolved.setdefault((str(workflow_id), source_name), []).append(
                    (output_name, stream_name)
                )
    return {key: tuple(pairs) for key, pairs in resolved.items()}


class ContextOutputExtractor:
    """Builds context-stream messages from job results.

    Resolving the registry in the constructor makes a colliding or malformed
    declaration a startup failure rather than a surprise on the first result.

    Parameters
    ----------
    registry:
        Workflow registry deciding which outputs are republished as context
        streams and under which names.
    """

    def __init__(self, *, registry: Mapping[WorkflowId, WorkflowSpec]) -> None:
        self._streams = resolve_context_streams(registry)

    def extract(self, results: list[JobResult]) -> list[Message[sc.DataArray]]:
        """Extract the designated context outputs of the given results.

        Parameters
        ----------
        results:
            Valid job results, each carrying a ``DataGroup`` of named outputs.

        Returns
        -------
        :
            One message per designated output, keyed by its rendered stream name
            on the
            :attr:`~ess.livedata.core.message.StreamKind.LIVEDATA_CONTEXT`
            stream.
        """
        messages: list[Message[sc.DataArray]] = []
        for result in results:
            if result.data is None:
                continue
            key = (str(result.workflow_id), result.job_id.source_name)
            for output_name, stream_name in self._streams.get(key, ()):
                value = result.data.get(output_name)
                if value is None:
                    continue
                messages.append(
                    Message(
                        timestamp=result.start_time or Timestamp.from_ns(0),
                        stream=StreamId(
                            kind=StreamKind.LIVEDATA_CONTEXT, name=stream_name
                        ),
                        value=value,
                    )
                )
        return messages
