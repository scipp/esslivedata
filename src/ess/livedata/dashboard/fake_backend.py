# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""In-process fake backend transport for Kafka-free dashboard testing.

This transport mimics a backend worker without Kafka. Commands the dashboard
sends (``WorkflowConfig``, ``JobCommand``) are captured by an in-process
:class:`FakeBackend`, which responds exactly as a real backend would: it
acknowledges the command, reports the job as running, and emits periodic
result data on the data stream.

Result data is synthesized from each workflow's output templates (the
``default_factory`` DataArrays declared on ``WorkflowSpec.outputs``). Because
the dashboard selects plotters from these same templates, generated data is
guaranteed to match what the plotter expects, so plots render correctly.

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
from ..config.workflow_spec import (
    JobId,
    ResultKey,
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
from .transport import DashboardResources, NullMessageSink, Transport

logger = structlog.get_logger(__name__)

# Number of points to expand each empty template dimension to.
_DEFAULT_DIM_SIZE = 64
# Wall-clock interval between synthesized data updates per job.
_UPDATE_PERIOD_SECONDS = 1.0


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

    A wiggling scalar (0-D) or a Gaussian bump (1-D and 2-D) plus mild noise.
    Amplitude grows with the update count to mimic accumulating statistics;
    ``variant`` shifts peak position, amplitude, and phase so distinct sources
    look distinct.
    """
    rng = np.random.default_rng(seed=(update, int(variant * 1_000_000)))
    phase = 2.0 * np.pi * variant
    amplitude = 100.0 * (update + 1) * (0.6 + 0.8 * variant)
    if len(sizes) == 0:
        signal = np.asarray(50.0 + 50.0 * np.sin(0.5 * update + phase))
    elif len(sizes) == 1:
        (nx,) = sizes
        x = np.linspace(0.0, 1.0, nx)
        center = 0.25 + 0.5 * variant + 0.08 * np.sin(0.3 * update)
        signal = amplitude * np.exp(-(((x - center) / 0.12) ** 2))
    elif len(sizes) == 2:
        nx, ny = sizes
        x = np.linspace(0.0, 1.0, nx)[:, None]
        y = np.linspace(0.0, 1.0, ny)[None, :]
        cx = 0.5 + 0.25 * np.sin(0.3 * update + phase)
        cy = 0.5 + 0.25 * np.cos(0.3 * update + phase)
        signal = amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 0.15**2))
    else:
        signal = np.full(sizes, amplitude, dtype=float)
    noise = rng.normal(scale=max(amplitude * 0.02, 1.0), size=tuple(sizes))
    return np.clip(signal + noise, a_min=0.0, a_max=None)


def expand_template(
    template: sc.DataArray, update: int, timestamp_ns: int, variant: float = 0.0
) -> sc.DataArray:
    """Turn an empty output template into a populated DataArray.

    Zero-length template dimensions are expanded to a default size; existing
    sized dimensions are preserved. Coordinates and synthetic values are
    generated to match the template's dims, units, and dtype.

    A scalar ``time`` coordinate (carried by per-update and timeseries outputs)
    is stamped with the current time. The backend emits a fresh scalar each
    update; the dashboard accumulates these into the time series.

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


class _Job:
    """A running fake job and its synthesized output state."""

    def __init__(self, config: WorkflowConfig, spec: WorkflowSpec) -> None:
        self.config = config
        self.spec = spec
        self.update = 0
        self.next_emit = 0.0  # monotonic deadline; 0 => emit immediately
        self.variant = source_variant(config.job_id.source_name)
        self.start_time = Timestamp.now()

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

    def _control_job(self, command: JobCommand) -> None:
        if command.action is JobAction.stop and command.job_id is not None:
            self._jobs.pop(command.job_id, None)
            self._ack(command.message_id, command.job_id.source_name)

    def _ack(self, message_id: str, device: str, error: str | None = None) -> None:
        response = AcknowledgementResponse.ERR if error else AcknowledgementResponse.ACK
        ack = CommandAcknowledgement(
            message_id=message_id, device=device, response=response, message=error
        )
        self._control.append(Message(stream=RESPONSES_STREAM_ID, value=ack))

    @staticmethod
    def _status(job: _Job) -> Message:
        status = JobStatus(
            job_id=job.config.job_id,
            workflow_id=job.config.identifier,
            state=JobState.active,
            start_time=job.start_time,
        )
        return Message(stream=STATUS_STREAM_ID, value=status)

    def poll(self) -> Sequence[Message]:
        """Return queued control messages plus due status and data updates.

        Each due cycle re-emits the job status, acting as a heartbeat so the
        dashboard keeps the job ACTIVE rather than letting it go stale.
        """
        now = time.monotonic()
        with self._lock:
            messages = self._control
            self._control = []
            for job in self._jobs.values():
                if now >= job.next_emit:
                    messages = [*messages, self._status(job), *self._emit_data(job)]
                    job.next_emit = now + _UPDATE_PERIOD_SECONDS
        return messages

    def _emit_data(self, job: _Job) -> list[Message]:
        timestamp_ns = time.time_ns()
        messages = []
        for output_name, template in job.output_templates().items():
            key = ResultKey(
                workflow_id=job.config.identifier,
                job_id=job.config.job_id,
                output_name=output_name,
            )
            stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name=key.model_dump_json())
            value = expand_template(template, job.update, timestamp_ns, job.variant)
            messages.append(Message(stream=stream, value=value))
        job.update += 1
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
            roi_sink=NullMessageSink(),
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
