# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""Adapt ess.reduce.streaming.StreamProcessor to the Workflow protocol."""

from __future__ import annotations

from typing import Any

import sciline
import sciline.typing

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
        **kwargs: Any,
    ) -> None:
        self._dynamic_keys = dynamic_keys
        self._context_keys = context_keys if context_keys else {}
        self._target_keys = target_keys
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
        return {name: targets[key] for name, key in self._target_keys.items()}

    def clear(self) -> None:
        self._stream_processor.clear()
