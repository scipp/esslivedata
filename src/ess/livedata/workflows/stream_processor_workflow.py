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
import scipp as sc
from ess.reduce import streaming

from ess.livedata.core.timestamp import Timestamp

from .dynamic_transforms import find_live_transforms
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
            Mapping from canonical stream name to sciline key for dynamic inputs.
            Dynamic inputs are accumulated across calls via
            ``StreamProcessor.accumulate()``. Factories key every entry by the
            canonical stream name incoming data arrives under: the primary input
            by its ``source_name`` and auxiliary inputs by resolving their role
            through the ``aux_source_names`` map the factory receives (e.g.
            ``aux_source_names['incident_monitor']``). The role space therefore
            never enters this mapping.
        context_keys:
            Mapping from stream names to sciline keys for context bindings.
            Context inputs update pipeline parameters via
            ``StreamProcessor.set_context()``. Unlike dynamic inputs, context
            values are **stateful**: a value set in one ``accumulate()`` call
            persists into all subsequent calls until explicitly overwritten.
            If data for a context key is absent from a given batch, the key
            retains its previous value. Every key must be set before the
            targets are computed: ``StreamProcessor`` overwrites each context
            key when it takes the pipeline over, so a value the base pipeline
            supplies for one does not survive. The ``JobManager``'s context
            gate is what guarantees this, delivering either a cached value or
            the binding's declared default before the job runs.
            A factory passes only its own internal context here (e.g. ROI);
            bindings resolved by the routing layer
            are merged in afterwards via :meth:`add_context_keys`, which is
            why construction of the wrapped ``StreamProcessor`` is deferred
            until :meth:`build`. The keys must be finalized
            before the graph is built because ``StreamProcessor`` bakes them
            into the pruned/precomputed pipeline at construction.
        target_keys:
            Mapping from output names to sciline keys for target outputs.
        window_outputs:
            Output names representing the current window (delta since last finalize).
            These receive their own ``start_time``/``time`` coords, bounding the
            window rather than the job.
        **kwargs:
            Additional arguments passed to StreamProcessor.
        """
        self._base_workflow = base_workflow
        self._dynamic_keys = dict(dynamic_keys)
        self._context_keys = dict(context_keys) if context_keys else {}
        self._target_keys = target_keys
        self._window_outputs = set(window_outputs)
        self._kwargs = kwargs
        self._current_start_time: Timestamp | None = None
        self._current_end_time: Timestamp | None = None
        self._stream_processor: streaming.StreamProcessor | None = None
        self._requested_context_streams: set[str] = set()
        # stream name -> the overrides entries its latest sample fills
        self._live_transforms: dict[str, list[tuple[sciline.typing.Key, str]]] = {}
        self._overrides: dict[sciline.typing.Key, dict[str, sc.Variable]] = {}

    def add_context_keys(self, context_keys: Mapping[str, sciline.typing.Key]) -> None:
        """Merge additional context bindings before the graph is built.

        Called by the routing layer (``WorkflowFactory.create``) to inject
        spec-scope context bindings resolved per job, so
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
    def requested_context_streams(self) -> frozenset[str]:
        """Offered context streams the built graph turned out to require.

        Empty before :meth:`build`. The routing layer reads this back off the
        workflow instead of predicting it: whether a job consumes the streamed
        lookup table or a motor readback follows from the graph its params
        produce, and only the graph knows (see :meth:`build`). Only *offered*
        streams and motor streams appear here.
        Factory-supplied context (ROI requests) reaches the gate as a
        :class:`~ess.livedata.config.stream.ContextBinding` instead, carrying
        the default that keeps a job whose producer may never publish from
        deadlocking on it.
        """
        return frozenset(self._requested_context_streams)

    def build(
        self,
        *,
        context_keys: Mapping[str, sciline.typing.Key] | None = None,
        transform_streams: Mapping[str, str] | None = None,
        offered_context_streams: Mapping[sciline.typing.Key, str] | None = None,
    ) -> None:
        """Materialize the wrapped ``StreamProcessor`` from its inputs.

        Merges the routing layer's per-job ``context_keys`` into the
        ``set_context`` parameters, then constructs the ``StreamProcessor``,
        baking them into the pruned/precomputed graph. Building eagerly (rather
        than on first ``accumulate``) keeps graph validation and the static-node
        precompute at job-creation time.

        ``transform_streams`` maps a NeXus transform path to the stream carrying
        its value, for the whole instrument. Each placeholder on a chain the
        targets read (see
        :func:`~ess.livedata.workflows.dynamic_transforms.find_live_transforms`)
        becomes a context stream of this job, filling its entry of
        ``TransformationOverrides``. A job that never computes a moving
        component's position therefore does not wait for the motor, and a
        stream can fill several component types. A placeholder with no stream
        would never get a value, so it raises here.

        ``offered_context_streams`` are the instrument's offered context
        streams (key → wire name). Unlike ``context_keys``, which the routing
        layer has already narrowed to this job, these reach every job and are
        kept only where the graph asks for the key: an entry is taken up when
        its key is an ancestor of a target key, which is precisely the
        condition under which ``finalize`` would fail without it. Insertion
        alone is not enough — a provider the targets cannot reach is pruned — so
        a factory may insert unconditionally and let the params decide, which is
        what the wavelength/time-of-arrival modes need (ADR 0010).

        Idempotent: a second call is a no-op and must not supply bindings, since
        they could no longer take effect once the graph is built.
        """
        if self._stream_processor is not None:
            if context_keys or transform_streams or offered_context_streams:
                raise RuntimeError(
                    "Cannot inject bindings: the StreamProcessor is already built."
                )
            return
        if context_keys:
            self.add_context_keys(context_keys)
        required = self._required_keys()
        offered = {
            name: key
            for key, name in (offered_context_streams or {}).items()
            if key in required
        }
        self.add_context_keys(offered)
        for key, paths in find_live_transforms(self._base_workflow, required).items():
            self._overrides[key] = {}
            for path in paths:
                stream = (transform_streams or {}).get(path)
                if stream is None:
                    raise ValueError(
                        f"Transform {path!r} on a chain read by this workflow is a "
                        "placeholder in the geometry artifact, and no stream of "
                        "the instrument carries its value."
                    )
                self._live_transforms.setdefault(stream, []).append((key, path))
        self._requested_context_streams = {*offered, *self._live_transforms}
        self._stream_processor = streaming.StreamProcessor(
            self._base_workflow,
            dynamic_keys=tuple(self._dynamic_keys.values()),
            context_keys=(*self._context_keys.values(), *self._overrides),
            target_keys=tuple(self._target_keys.values()),
            **self._kwargs,
        )

    def _required_keys(self) -> set[sciline.typing.Key]:
        """Every key a target key depends on, directly or transitively.

        Walks the pipeline's data graph backwards from the targets. Keys with no
        provider and no value are nodes like any other, so a key the graph asks
        for is found here whether or not anything can supply it -- one nothing
        supplies then fails the ``StreamProcessor`` construction below, naming
        it. Providers the targets cannot reach are excluded, matching the
        pruning ``StreamProcessor`` applies to the same targets.
        """
        graph = self._base_workflow.underlying_graph
        required: set[sciline.typing.Key] = set()
        pending = [key for key in self._target_keys.values() if key in graph]
        while pending:
            for key in graph.predecessors(pending.pop()):
                if key not in required:
                    required.add(key)
                    pending.append(key)
        return required

    @property
    def _processor(self) -> streaming.StreamProcessor:
        # No lazy build: silently materializing here would skip the routing
        # layer's bindings and move graph validation from job creation to
        # first data arrival. WorkflowFactory.create builds eagerly; any other
        # call site must do the same.
        if self._stream_processor is None:
            raise RuntimeError(
                "StreamProcessorWorkflow used before build(); "
                "WorkflowFactory.create builds it after creation."
            )
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
        # the most recent set_context call. The context gate is what ensures
        # every key has been set by the time the job first runs. The
        # routing layer (JobFactory.create) delivers a stream only to the jobs
        # whose bindings subscribe to it or whose graph requested it.
        context = {
            sciline_key: data[key]
            for key, sciline_key in self._context_keys.items()
            if key in data
        }
        for stream, entries in self._live_transforms.items():
            if stream in data:
                value = data[stream]['time', -1].data
                for key, path in entries:
                    self._overrides[key][path] = value
                    context[key] = dict(self._overrides[key])
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
            time_coord = self._current_end_time.to_scipp()

            for name in self._window_outputs:
                if name in results:
                    results[name] = results[name].assign_coords(
                        start_time=start_time_coord, time=time_coord
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
