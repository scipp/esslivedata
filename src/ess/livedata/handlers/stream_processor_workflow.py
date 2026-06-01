# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""Adapt ess.reduce.streaming.StreamProcessor to the Workflow protocol."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import graphviz

import sciline
import sciline.typing
from ess.reduce import streaming

from ess.livedata.config.value_log import ValueLog
from ess.livedata.core.timestamp import Timestamp

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
        aux_source_names: Mapping[str, str] | None = None,
        window_outputs: Iterable[str] = (),
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        base_workflow:
            The sciline Pipeline to wrap.
        dynamic_keys:
            Mapping from wire name to sciline key for dynamic inputs. Dynamic
            inputs are accumulated across calls via
            ``StreamProcessor.accumulate()``. Factories declare the primary
            input by its on-disk ``source_name`` and auxiliary inputs by their
            stable *role* (e.g. ``'incident_monitor'``); ``aux_source_names``
            resolves the roles so that, after construction, the stored mapping
            is uniformly keyed by on-disk stream name — the name incoming data
            arrives under.
        aux_source_names:
            Rendered ``role -> on-disk stream`` mapping for this job's aux
            sources. Normalises aux roles in ``dynamic_keys`` to on-disk stream
            names. Names absent from the map (the primary source, synthetic
            inputs) pass through unchanged.
        context_keys:
            Mapping from stream names to sciline keys for context bindings.
            Context inputs update pipeline parameters via
            ``StreamProcessor.set_context()``. Unlike dynamic inputs, context
            values are **stateful**: a value set in one ``accumulate()`` call
            persists into all subsequent calls until explicitly overwritten.
            If data for a context key is absent from a given batch, the key
            retains its previous value. If ``set_context`` was never called
            for a key and the underlying sciline pipeline has no default for
            it, ``finalize()`` will raise an ``UnsatisfiedGraphError``.
            A factory passes only its own internal context here (e.g. ROI);
            instrument- and spec-scope bindings resolved by the routing layer
            are merged in afterwards via :meth:`add_context_keys`, which is
            why construction of the wrapped ``StreamProcessor`` is deferred
            until :meth:`build` (or first use). The keys must be finalized
            before the graph is built because ``StreamProcessor`` bakes them
            into the pruned/precomputed pipeline at construction.
        target_keys:
            Mapping from output names to sciline keys for target outputs.
        window_outputs:
            Output names representing the current window (delta since last finalize).
            These receive time, start_time, end_time coords.
        **kwargs:
            Additional arguments passed to StreamProcessor.
        """
        self._base_workflow = base_workflow
        resolved = aux_source_names or {}
        self._dynamic_keys = {
            resolved.get(name, name): key for name, key in dynamic_keys.items()
        }
        self._context_keys = dict(context_keys) if context_keys else {}
        self._target_keys = target_keys
        self._window_outputs = set(window_outputs)
        self._kwargs = kwargs
        self._current_start_time: Timestamp | None = None
        self._current_end_time: Timestamp | None = None
        self._stream_processor: streaming.StreamProcessor | None = None

    def add_context_keys(self, context_keys: Mapping[str, sciline.typing.Key]) -> None:
        """Merge additional context bindings before the graph is built.

        Called by the routing layer (``WorkflowFactory.create``) to inject
        instrument- and spec-scope context bindings resolved per job, so
        factories need not thread ``context_keys`` through their signature.

        Raises if the wrapped ``StreamProcessor`` has already been built: its
        context keys are fixed at construction and cannot change afterwards.
        """
        if self._stream_processor is not None:
            raise RuntimeError(
                "Cannot add context keys after the StreamProcessor is built."
            )
        self._context_keys = {**self._context_keys, **context_keys}

    @property
    def dynamic_keys(self) -> dict[str, sciline.typing.Key]:
        """Mapping from on-disk stream name to sciline key for dynamic inputs.

        Aux roles are already resolved to on-disk names at construction, so the
        routing layer can match each wire name directly against a binding's
        ``dependent_sources`` and derive the NeXus component type (the first
        type-arg of a ``NeXusData[Component, Run]`` key) to wire f144-driven
        dynamic transforms. See
        :func:`ess.livedata.handlers.dynamic_transforms.wire_dynamic_transforms`.
        """
        return dict(self._dynamic_keys)

    @property
    def base_pipeline(self) -> sciline.Pipeline:
        """The unbuilt base pipeline, exposed for pre-build patching.

        Raises if the wrapped ``StreamProcessor`` has already been built: the
        pipeline is baked into the pruned/precomputed graph at construction, so
        patches must land before :meth:`build`.
        """
        if self._stream_processor is not None:
            raise RuntimeError(
                "Cannot patch the pipeline after the StreamProcessor is built."
            )
        return self._base_workflow

    def build(self) -> None:
        """Build the wrapped ``StreamProcessor``. Idempotent; no-op if built.

        Materializing eagerly (rather than on first ``accumulate``) keeps
        graph validation and the static-node precompute at job-creation time.
        """
        if self._stream_processor is not None:
            return
        self._stream_processor = streaming.StreamProcessor(
            self._base_workflow,
            dynamic_keys=tuple(self._dynamic_keys.values()),
            context_keys=tuple(self._context_keys.values()),
            target_keys=tuple(self._target_keys.values()),
            **self._kwargs,
        )

    @property
    def _processor(self) -> streaming.StreamProcessor:
        self.build()
        assert self._stream_processor is not None  # noqa: S101  # narrows type after build()
        return self._stream_processor

    def accumulate(
        self, data: dict[str, Any], *, start_time: Timestamp, end_time: Timestamp
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
        #
        # ValueLog subclasses are typed wrappers around an NXlog DataArray;
        # the raw payload (a DataArray) is wrapped as key(values=raw) so
        # each chain-patch binding has a distinct Sciline node identity.
        context = {
            sciline_key: (
                sciline_key(values=data[key])
                if isinstance(sciline_key, type) and issubclass(sciline_key, ValueLog)
                else data[key]
            )
            for key, sciline_key in self._context_keys.items()
            if key in data
        }
        dynamic = {
            sciline_key: data[key]
            for key, sciline_key in self._dynamic_keys.items()
            if key in data
        }
        if context:
            self._processor.set_context(context)
        if dynamic:
            self._processor.accumulate(dynamic)

    def finalize(self) -> dict[str, Any]:
        targets = self._processor.finalize()
        results = {name: targets[key] for name, key in self._target_keys.items()}

        # Add time coords to window outputs
        if self._window_outputs and self._current_start_time is not None:
            start_time_coord = self._current_start_time.to_scipp()
            end_time_coord = self._current_end_time.to_scipp()

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
        self._processor.clear()
        self._current_start_time = None
        self._current_end_time = None

    def visualize(self, **kwargs: Any) -> graphviz.Digraph:
        """Visualize the streaming workflow graph.

        See :py:meth:`ess.reduce.streaming.StreamProcessor.visualize` for parameters.
        """
        return self._processor.visualize(**kwargs)
