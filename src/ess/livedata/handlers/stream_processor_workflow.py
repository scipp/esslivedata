# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""Adapt ess.reduce.streaming.StreamProcessor to the Workflow protocol."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import sciline
import sciline.typing
import scipp as sc
from ess.reduce import streaming

from .workflow_factory import Workflow


class StreamProcessorWorkflow(Workflow):
    """
    Wrapper around ess.reduce.streaming.StreamProcessor to match the Workflow protocol.

    This maps from stream names to sciline Keys for inputs, and from simplified
    output names to sciline Keys for targets. The simplified output names (dict keys
    in target_keys) are used as keys in the dictionary returned by finalize().
    """

    def __init__(
        self,
        base_workflow: sciline.Pipeline,
        *,
        dynamic_keys: dict[str, sciline.typing.Key],
        context_keys: dict[str, sciline.typing.Key] | None = None,
        target_keys: dict[str, sciline.typing.Key],
        window_outputs: Iterable[str] = (),
        reset_on_context_change: frozenset[str] = frozenset(),
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        base_workflow:
            The sciline Pipeline to wrap.
        dynamic_keys:
            Mapping from stream names to sciline keys for dynamic inputs.
            Dynamic inputs are accumulated across calls via
            ``StreamProcessor.accumulate()``.
        context_keys:
            Mapping from stream names to sciline keys for context inputs.
            Context inputs update pipeline parameters via
            ``StreamProcessor.set_context()``. Unlike dynamic inputs, context
            values are **stateful**: a value set in one ``accumulate()`` call
            persists into all subsequent calls until explicitly overwritten.
            If data for a context key is absent from a given batch, the key
            retains its previous value. If ``set_context`` was never called
            for a key and the underlying sciline pipeline has no default for
            it, ``finalize()`` will raise an ``UnsatisfiedGraphError``.
        target_keys:
            Mapping from output names to sciline keys for target outputs.
        window_outputs:
            Output names representing the current window (delta since last finalize).
            These receive time, start_time, end_time coords.
        reset_on_context_change:
            Set of context key names (stream names, not sciline keys) whose value
            changes should trigger an accumulator reset. When a new value for one
            of these keys differs from the previously seen value, all accumulators
            are cleared before the new data is accumulated. This is needed for
            monitor and detector view workflows where a position change invalidates
            the accumulated histogram. Reduction workflows should NOT use this,
            as they accumulate data across different positions into a shared output
            space.
        **kwargs:
            Additional arguments passed to StreamProcessor.
        """
        self._dynamic_keys = dynamic_keys
        self._context_keys = context_keys if context_keys else {}
        self._target_keys = target_keys
        self._window_outputs = set(window_outputs)
        self._reset_on_context_change = reset_on_context_change
        self._previous_context: dict[str, Any] = {}
        self._current_start_time: int | None = None
        self._current_end_time: int | None = None
        self._stream_processor = streaming.StreamProcessor(
            base_workflow,
            dynamic_keys=tuple(self._dynamic_keys.values()),
            context_keys=tuple(self._context_keys.values()),
            target_keys=tuple(self._target_keys.values()),
            **kwargs,
        )

    def accumulate(
        self, data: dict[str, Any], *, start_time: int, end_time: int
    ) -> None:
        # Track time range of data since last finalize
        if self._current_start_time is None:
            self._current_start_time = start_time
        self._current_end_time = end_time

        # Context data (e.g., positions from f144 streams) is injected via
        # set_context, which updates the sciline pipeline parameters. Only keys
        # present in this batch are updated; absent keys retain the value from
        # the most recent set_context call, or the pipeline's init-time value.
        # If a key has no init-time value and has never been set, finalize()
        # will fail. See aux_sources / render() in workflow_spec.py for how
        # the routing layer ensures only jobs that subscribed to a stream
        # receive its data.
        context = {
            sciline_key: data[key]
            for key, sciline_key in self._context_keys.items()
            if key in data
        }
        dynamic = {
            sciline_key: data[key]
            for key, sciline_key in self._dynamic_keys.items()
            if key in data
        }
        if context:
            self._maybe_reset_on_context_change(data)
            self._stream_processor.set_context(context)
        if dynamic:
            self._stream_processor.accumulate(dynamic)

    def _maybe_reset_on_context_change(self, data: dict[str, Any]) -> None:
        """Reset accumulators if a tracked context key changed value."""
        for key in self._reset_on_context_change:
            if key not in data:
                continue
            if key in self._previous_context and not _values_equal(
                self._previous_context[key], data[key]
            ):
                self._stream_processor.clear()
                self._current_start_time = None
                self._current_end_time = None
                break
        # Update tracked values for all reset-eligible keys present in this batch
        for key in self._reset_on_context_change:
            if key in data:
                self._previous_context[key] = data[key]

    def finalize(self) -> dict[str, Any]:
        targets = self._stream_processor.finalize()
        results = {name: targets[key] for name, key in self._target_keys.items()}

        # Add time coords to window outputs
        if self._window_outputs and self._current_start_time is not None:
            start_time_coord = sc.scalar(self._current_start_time, unit='ns')
            end_time_coord = sc.scalar(self._current_end_time, unit='ns')

            for name in self._window_outputs:
                if name in results:
                    results[name] = results[name].assign_coords(
                        time=start_time_coord,
                        start_time=start_time_coord,
                        end_time=end_time_coord,
                    )

        # Reset time tracking for next period
        self._current_start_time = None
        self._current_end_time = None

        return results

    def clear(self) -> None:
        self._stream_processor.clear()
        self._current_start_time = None
        self._current_end_time = None


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two values, using sc.identical for scipp types."""
    if isinstance(a, (sc.Variable, sc.DataArray)):
        return sc.identical(a, b)
    return a == b
