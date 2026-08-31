# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""In-process fake backend transport for Kafka-free dashboard testing.

This transport mimics a backend worker without Kafka. Commands the dashboard
sends (``WorkflowConfig``, ``JobCommand``) are captured by an in-process
:class:`FakeBackend`, which responds exactly as a real backend would: it
acknowledges the command, reports the job as running, and emits periodic
result data on the data stream. Stopping a job yields a terminal ``stopped``
status, and :meth:`FakeBackend.fail_job` faults one into the ``error`` state,
so the dashboard's non-live presentations are reachable too.

Result data is synthesized from each workflow's output templates (the
``default_factory`` DataArrays declared on ``WorkflowSpec.outputs``). Because
the dashboard selects plotters from these same templates, generated data is
guaranteed to match what the plotter expects, so plots render correctly.

ROI requests the dashboard publishes are looped back in the same way: the
backend stores the latest request per job and geometry, echoes it as the
geometry readback, and synthesizes one spectrum per drawn ROI.

The whole flow is driven by the normal UI: start a workflow and plots come to
life. No fixtures, no external injection, no Kafka.
"""

from __future__ import annotations

import threading
import time
import zlib
from collections.abc import Mapping, Sequence
from types import TracebackType

import numpy as np
import scipp as sc
import structlog

from ..config.acknowledgement import AcknowledgementResponse, CommandAcknowledgement
from ..config.instruments import get_config
from ..config.roi_names import get_roi_mapper, roi_stream_name
from ..config.workflow_spec import (
    JobId,
    ResultKey,
    Temporality,
    WorkflowConfig,
    WorkflowId,
    WorkflowSpec,
)
from ..core.job import JobState, JobStatus
from ..core.job_manager import Command, JobAction, JobCommand
from ..core.message import (
    RESPONSES_STREAM_ID,
    STATUS_STREAM_ID,
    Message,
    StreamId,
    StreamKind,
)
from ..core.timestamp import Timestamp
from .transport import DashboardResources, Transport

logger = structlog.get_logger(__name__)

# Number of points to expand each empty template dimension to.
_DEFAULT_DIM_SIZE = 64
# Wall-clock interval between synthesized data updates per job.
_UPDATE_PERIOD_SECONDS = 1.0
# Output fields echoing ROI geometry; named after the ROI stream readback keys.
_ROI_READBACK_KEYS = frozenset(get_roi_mapper().readback_keys)


def _expand_coord(coord: sc.Variable, dim: str, size: int) -> sc.Variable:
    """Build a length-``size`` coordinate from an empty template coordinate.

    Templates carry coordinates of zero length that only declare dim, unit, and
    dtype. We synthesize a monotonically increasing coordinate spanning a unit
    range so plots have a sensible axis.
    """
    if coord.dtype == sc.DType.datetime64:
        start = np.datetime64('2025-01-01T00:00:00', 's')
        values = start + np.arange(size, dtype='timedelta64[s]')
        return sc.array(dims=[dim], values=values, unit=coord.unit)
    return sc.linspace(dim, 0.0, float(size), num=size, unit=coord.unit)


def source_variant(source_name: str) -> float:
    """Map a source name to a stable phase in ``[0, 1)``.

    Used to give each source of a multi-source workflow a distinct curve, so
    overlaid lines (e.g. per-monitor histograms) don't lie on top of each other.
    """
    return (zlib.crc32(source_name.encode()) % 1000) / 1000.0


def _synthesize_values(sizes: Sequence[int], update: int, variant: float) -> np.ndarray:
    """Generate plausible-looking data that varies over updates and sources.

    A wiggling scalar (0-D) or a drifting Gaussian bump (any higher rank) plus
    mild noise. Amplitude grows with the update count to mimic accumulating
    statistics; ``variant`` shifts peak position, amplitude, and phase so
    distinct sources look distinct. The bump is a product of per-axis
    Gaussians, each drifting on its own quarter-turn of the same slow
    oscillation, so every axis carries structure a plot can show.
    """
    rng = np.random.default_rng(seed=(update, int(variant * 1_000_000)))
    phase = 2.0 * np.pi * variant
    amplitude = 100.0 * (update + 1) * (0.6 + 0.8 * variant)
    if not sizes:
        signal = np.asarray(50.0 + 50.0 * np.sin(0.5 * update + phase))
    else:
        signal = np.full((1,) * len(sizes), amplitude)
        for axis, size in enumerate(sizes):
            shape = [1] * len(sizes)
            shape[axis] = size
            x = np.linspace(0.0, 1.0, size).reshape(shape)
            center = 0.5 + 0.25 * np.sin(0.3 * update + phase + axis * 0.5 * np.pi)
            signal = signal * np.exp(-(((x - center) / 0.15) ** 2))
    noise = rng.normal(scale=max(amplitude * 0.02, 1.0), size=tuple(sizes))
    return np.clip(signal + noise, a_min=0.0, a_max=None)


def expand_template(
    template: sc.DataArray, update: int, timestamp_ns: int, variant: float = 0.0
) -> sc.DataArray:
    """Turn an empty output template into a populated DataArray.

    Zero-length template dimensions are expanded to a default size; existing
    sized dimensions are preserved. Coordinates and synthetic values are
    generated to match the template's dims, units, and dtype.

    A scalar ``time`` coordinate (carried by series outputs, whose ``time`` is
    real per-sample data) is stamped with the current time. The backend emits a
    fresh scalar each update; the dashboard accumulates these into the series.

    Parameters
    ----------
    template:
        Empty DataArray from a workflow output field's ``default_factory``.
    update:
        Monotonic update counter; varies the data so plots appear live.
    timestamp_ns:
        Wall-clock time of this update, used for the ``time`` coordinate.
    variant:
        Per-source phase in ``[0, 1)`` distinguishing overlaid sources.
    """
    sizes = [size or _DEFAULT_DIM_SIZE for size in template.shape]
    values = _synthesize_values(sizes, update, variant)
    data = sc.array(
        dims=template.dims, values=values, unit=template.unit, dtype='float64'
    )
    coords = {
        dim: _expand_coord(template.coords[dim], dim, size)
        for dim, size in zip(template.dims, sizes, strict=True)
        if dim in template.coords
    }
    if (time := template.coords.get('time')) is not None and time.ndim == 0:
        coords['time'] = sc.scalar(timestamp_ns, unit=time.unit, dtype=time.dtype)
    return sc.DataArray(data=data, coords=coords)


def roi_variants(rois: Mapping[str, sc.DataArray]) -> dict[int, float]:
    """Map each drawn ROI's index to a stable phase in ``[0, 1)``.

    The phase is derived from the ROI's own vertex coordinates, so moving or
    resizing an ROI changes its synthesized spectrum while an untouched ROI
    keeps its curve. Indices are unique across geometries (each geometry owns a
    disjoint index range) and sorted, giving a stable row order.

    Parameters
    ----------
    rois:
        Concatenated ROI request per geometry readback key, as published by the
        dashboard.
    """
    variants: dict[int, float] = {}
    for geometry in rois.values():
        indices = geometry.coords['roi_index'].values
        for roi_index in np.unique(indices):
            vertices = np.concatenate(
                [
                    geometry.coords[dim].values[indices == roi_index]
                    for dim in ('x', 'y')
                ]
            )
            variants[int(roi_index)] = (zlib.crc32(vertices.tobytes()) % 1000) / 1000.0
    return dict(sorted(variants.items()))


def expand_roi_spectra(
    template: sc.DataArray,
    variants: Mapping[int, float],
    update: int,
    timestamp_ns: int,
) -> sc.DataArray:
    """Build one synthetic spectrum per currently drawn ROI.

    The length of the ``roi`` dimension follows the ROI set the dashboard
    published, down to zero rows while no ROI is drawn. This mirrors the real
    backend, which computes ROI spectra from the start and yields an empty
    result until an ROI request arrives.

    Parameters
    ----------
    template:
        Empty ROI spectra template, with dims ``(roi, <spectral>)``.
    variants:
        Phase per ROI index, see :func:`roi_variants`.
    update:
        Monotonic update counter; varies the data so plots appear live.
    timestamp_ns:
        Wall-clock time of this update, used for the ``time`` coordinate.
    """
    spectral_size = template.sizes[template.dims[-1]] or _DEFAULT_DIM_SIZE
    spectra = [
        _synthesize_values([spectral_size], update, variant)
        for variant in variants.values()
    ]
    values = np.stack(spectra) if spectra else np.zeros((0, spectral_size))
    coords = {'roi': sc.array(dims=['roi'], values=list(variants), dtype='int32')}
    if (time := template.coords.get('time')) is not None and time.ndim == 0:
        coords['time'] = sc.scalar(timestamp_ns, unit=time.unit, dtype=time.dtype)
    data = sc.array(dims=template.dims, values=values, unit=template.unit)
    return sc.DataArray(data=data, coords=coords)


class _Job:
    """A running fake job and its synthesized output state."""

    def __init__(self, config: WorkflowConfig, spec: WorkflowSpec) -> None:
        self.config = config
        self.spec = spec
        self.update = 0
        self.next_emit = 0.0  # monotonic deadline; 0 => emit immediately
        self.variant = source_variant(config.job_id.source_name)
        self.start_time = Timestamp.now()
        self.previous_emit = self.start_time
        self.error_message: str | None = None

    @property
    def state(self) -> JobState:
        """Wire-facing state, derived from health as the backend derives it."""
        return JobState.active if self.error_message is None else JobState.error

    def output_templates(self) -> Mapping[str, sc.DataArray]:
        """Templates for every output field that declares one."""
        return {
            name: field.default_factory()
            for name, field in self.spec.outputs.model_fields.items()
            if field.default_factory is not None
        }


class FakeBackend:
    """Captures dashboard commands and synthesizes backend responses.

    Thread-safe: commands arrive on the UI thread via :meth:`submit`; data is
    drained on the background update thread via :meth:`poll`.
    """

    def __init__(self, workflows: Mapping[WorkflowId, WorkflowSpec]) -> None:
        self._workflows = workflows
        self._jobs: dict[JobId, _Job] = {}
        # Latched ROI requests, keyed by the wire stream name they arrived on.
        # Outlives the jobs reading them, as the backend's context accumulators
        # do.
        self._rois: dict[str, sc.DataArray] = {}
        self._control: list[Message] = []
        self._lock = threading.Lock()

    def submit(self, command: Command) -> None:
        """Handle a command sent by the dashboard."""
        with self._lock:
            if isinstance(command, WorkflowConfig):
                self._start(command)
            elif isinstance(command, JobCommand):
                self._control_job(command)

    def _start(self, config: WorkflowConfig) -> None:
        spec = self._workflows.get(config.identifier)
        if spec is None:
            self._ack(config.message_id, config.job_id.source_name, error='unknown')
            return
        self._jobs[config.job_id] = _Job(config=config, spec=spec)
        self._ack(config.message_id, config.job_id.source_name)
        logger.info("fake_backend_started", job_id=str(config.job_id))

    def set_roi(self, stream_name: str, rois: sc.DataArray) -> None:
        """Store an ROI request, replacing the previous one for its geometry.

        Latest-value semantics match the backend, which accumulates ROI streams
        with a ``LatestValueAccumulator``. Requests are latched per view whether
        or not a job is running, so one published before a job starts is picked
        up when it does.

        Parameters
        ----------
        stream_name:
            ROI stream name, see
            :func:`~ess.livedata.config.roi_names.roi_stream_name`.
        rois:
            Concatenated ROI geometries for that readback key.
        """
        with self._lock:
            self._rois[stream_name] = rois

    def _job_rois(self, job: _Job) -> dict[str, sc.DataArray]:
        """The latched ROI requests a job reads, keyed by readback key."""
        workflow_id = job.config.identifier
        source_name = job.config.job_id.source_name
        found = {
            key: self._rois.get(roi_stream_name(workflow_id, source_name, key))
            for key in _ROI_READBACK_KEYS
        }
        return {key: rois for key, rois in found.items() if rois is not None}

    def fail_job(self, job_id: JobId, message: str) -> None:
        """Fault a running job, as a workflow raising in the backend does.

        The job keeps its slot and reports :attr:`JobState.error` from then on,
        but yields no more results: the backend drops results that carry an
        error message.

        Parameters
        ----------
        job_id:
            Job to fault; must be running.
        message:
            Error message reported with the job's status.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"No running job {job_id}")
            job.error_message = message

    def _control_job(self, command: JobCommand) -> None:
        if command.action is JobAction.stop and command.job_id is not None:
            job = self._jobs.pop(command.job_id, None)
            if job is not None:
                # Terminal status ahead of the job's disappearance: the
                # dashboard freezes the cell on it rather than aging it out.
                self._control.append(self._status(job, JobState.stopped))
            self._ack(command.message_id, command.job_id.source_name)

    def _ack(self, message_id: str, device: str, error: str | None = None) -> None:
        response = AcknowledgementResponse.ERR if error else AcknowledgementResponse.ACK
        ack = CommandAcknowledgement(
            message_id=message_id, device=device, response=response, message=error
        )
        self._control.append(Message(stream=RESPONSES_STREAM_ID, value=ack))

    @staticmethod
    def _status(job: _Job, state: JobState) -> Message:
        status = JobStatus(
            job_id=job.config.job_id,
            workflow_id=job.config.identifier,
            state=state,
            error_message=job.error_message,
            start_time=job.start_time,
        )
        return Message(stream=STATUS_STREAM_ID, value=status)

    def poll(self) -> Sequence[Message]:
        """Return queued control messages plus due status and data updates.

        Each due cycle re-emits the job status, acting as a heartbeat so the
        dashboard keeps the job ACTIVE rather than letting it go stale. A
        faulted job keeps heartbeating its error state but produces no data.
        """
        now = time.monotonic()
        with self._lock:
            messages = self._control
            self._control = []
            for job in self._jobs.values():
                if now >= job.next_emit:
                    data = [] if job.error_message is not None else self._emit_data(job)
                    messages = [*messages, self._status(job, job.state), *data]
                    job.next_emit = now + _UPDATE_PERIOD_SECONDS
        return messages

    def _emit_data(self, job: _Job) -> list[Message]:
        timestamp_ns = time.time_ns()
        rois = self._job_rois(job)
        variants = roi_variants(rois)
        now = Timestamp.from_ns(timestamp_ns)
        messages = []
        for output_name, template in job.output_templates().items():
            key = ResultKey(
                workflow_id=job.config.identifier,
                job_id=job.config.job_id,
                output_name=output_name,
            )
            stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name=key.model_dump_json())
            if output_name in _ROI_READBACK_KEYS:
                # The backend echoes the request as readback; the template is
                # its empty-request equivalent.
                value = rois.get(output_name, template)
            elif 'roi' in template.dims:
                value = expand_roi_spectra(template, variants, job.update, timestamp_ns)
            else:
                value = expand_template(template, job.update, timestamp_ns, job.variant)
            # Production arrives at these coords by a different route:
            # `StreamProcessorWorkflow.finalize` and `AreaDetectorView.finalize`
            # assign per-interval bounds to the outputs their factory names in
            # `window_outputs=`, then `Job._add_time_coords` stamps one (job start,
            # latest observation) pair on whatever is left carrying neither. Keying
            # off the `Temporality` declaration reproduces the composite, but more
            # forgivingly: in production only a named `window_outputs=` entry gets
            # per-interval bounds, whatever it declares.
            temporality = job.spec.outputs.temporality(output_name)
            if temporality is not Temporality.series:
                start_time = (
                    job.start_time
                    if temporality is Temporality.cumulative
                    else job.previous_emit
                )
                value = value.assign_coords(
                    start_time=start_time.to_scipp(), time=now.to_scipp()
                )
            messages.append(Message(stream=stream, value=value))
        job.update += 1
        job.previous_emit = now
        return messages


class _FakeMessageSource:
    """Drains synthesized messages from the backend."""

    def __init__(self, backend: FakeBackend) -> None:
        self._backend = backend

    def get_messages(self) -> Sequence[Message]:
        return self._backend.poll()


class _FakeCommandSink:
    """Feeds dashboard commands into the backend."""

    def __init__(self, backend: FakeBackend) -> None:
        self._backend = backend

    def publish_messages(self, messages: list[Message[Command]]) -> None:
        for message in messages:
            self._backend.submit(message.value)


class _FakeROISink:
    """Feeds dashboard ROI requests into the backend."""

    def __init__(self, backend: FakeBackend) -> None:
        self._backend = backend

    def publish_messages(self, messages: list[Message[sc.DataArray]]) -> None:
        for message in messages:
            self._backend.set_roi(message.stream.name, message.value)


class FakeBackendTransport(Transport[DashboardResources]):
    """Transport with an in-process fake backend instead of Kafka.

    Parameters
    ----------
    instrument:
        Instrument name; its workflow registry provides the output templates.
    """

    def __init__(self, *, instrument: str) -> None:
        self._instrument = instrument

    def __enter__(self) -> DashboardResources:
        # Importing the instrument module registers its workflows.
        get_config(self._instrument)
        from ..config import instrument_registry

        workflows = instrument_registry[self._instrument].workflow_factory
        self._backend = FakeBackend(workflows)
        logger.info("fake_backend_transport_initialized", instrument=self._instrument)
        return DashboardResources(
            message_source=_FakeMessageSource(self._backend),
            command_sink=_FakeCommandSink(self._backend),
            roi_sink=_FakeROISink(self._backend),
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        logger.info("fake_backend_transport_cleaned_up")

    def start(self) -> None:
        """No background tasks; data is generated on poll."""

    def stop(self) -> None:
        """No background tasks to stop."""
