# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Publisher for histogram slice range updates to Kafka."""

import scipp as sc
import structlog

from ..config.workflow_spec import JobId
from ..core.message import Message, MessageSink, StreamId, StreamKind

logger = structlog.get_logger(__name__)


class RangePublisher:
    """
    Publishes histogram slice range updates to Kafka.

    Parameters
    ----------
    sink:
        Message sink for publishing messages.
    """

    def __init__(self, sink: MessageSink):
        self._sink = sink

    def publish(self, job_id: JobId, low: float, high: float, unit: str | None) -> None:
        """
        Publish a histogram slice range to Kafka.

        Parameters
        ----------
        job_id:
            The full job identifier (source_name and job_number).
        low:
            Lower bound of the range.
        high:
            Upper bound of the range.
        unit:
            Unit of the range values.
        """
        stream_name = f"{job_id}/histogram_slice"
        stream_id = StreamId(kind=StreamKind.LIVEDATA_ROI, name=stream_name)
        data = sc.concat(
            [sc.scalar(low, unit=unit), sc.scalar(high, unit=unit)], dim='bound'
        )
        msg = Message(value=sc.DataArray(data=data), stream=stream_id)
        self._sink.publish_messages([msg])
        logger.debug(
            "Published histogram slice [%s, %s] %s for job %s",
            low,
            high,
            unit,
            job_id,
        )

    def clear(self, job_id: JobId) -> None:
        """
        Clear the histogram slice (restore full range) for a job.

        Parameters
        ----------
        job_id:
            The full job identifier.
        """
        stream_name = f"{job_id}/histogram_slice"
        stream_id = StreamId(kind=StreamKind.LIVEDATA_ROI, name=stream_name)
        msg = Message(
            value=sc.DataArray(data=sc.zeros(sizes={'bound': 0})),
            stream=stream_id,
        )
        self._sink.publish_messages([msg])
        logger.debug("Cleared histogram slice for job %s", job_id)


class FakeRangePublisher:
    """Fake range publisher for testing."""

    def __init__(self):
        self.published: list[tuple[JobId, float | None, float | None, str | None]] = []

    def publish(self, job_id: JobId, low: float, high: float, unit: str | None) -> None:
        """Record published range."""
        self.published.append((job_id, low, high, unit))

    def clear(self, job_id: JobId) -> None:
        """Record cleared range."""
        self.published.append((job_id, None, None, None))

    def reset(self) -> None:
        """Clear all recorded publishes."""
        self.published.clear()
