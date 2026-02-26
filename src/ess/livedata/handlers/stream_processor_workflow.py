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
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        base_workflow:
            The sciline Pipeline to wrap.
        dynamic_keys:
            Mapping from stream names to sciline keys for dynamic inputs.
        context_keys:
            Mapping from stream names to sciline keys for context inputs.
        target_keys:
            Mapping from output names to sciline keys for target outputs.
        window_outputs:
            Output names representing the current window (delta since last finalize).
            These receive time, start_time, end_time coords.
        **kwargs:
            Additional arguments passed to StreamProcessor.
        """
        self._dynamic_keys = dynamic_keys
        self._context_keys = context_keys if context_keys else {}
        self._target_keys = target_keys
        self._window_outputs = set(window_outputs)
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
            self._stream_processor.set_context(context)
        if dynamic:
            self._stream_processor.accumulate(dynamic)

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
