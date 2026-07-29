# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Publisher for ROI updates to Kafka."""

from collections.abc import Callable

import structlog

from ..config.models import PolygonROI, RectangleROI
from ..config.roi_names import ROIGeometry
from ..config.workflow_spec import JobId, JobNumber, WorkflowId
from ..core.message import Message, MessageSink, StreamId, StreamKind
from ..core.timestamp import Timestamp

logger = structlog.get_logger(__name__)

ROIs = dict[int, RectangleROI] | dict[int, PolygonROI]


class ROIPublisher:
    """
    Publishes ROI updates to Kafka.

    This class provides a simple interface for publishing ROIs to the
    LIVEDATA_ROI Kafka topic.

    Callers address ROIs by stable identity (workflow_id, source_name); the
    backend stream name needs the current job_number, which is resolved at
    publish time through a resolver injected once the JobOrchestrator exists.

    Parameters
    ----------
    sink:
        Message sink for publishing messages.
    """

    def __init__(self, sink: MessageSink):
        self._sink = sink
        self._job_number_resolver: Callable[[WorkflowId], JobNumber | None] | None = (
            None
        )

    def set_job_number_resolver(
        self, resolver: Callable[[WorkflowId], JobNumber | None]
    ) -> None:
        """Set the resolver mapping a workflow to its current job_number."""
        self._job_number_resolver = resolver

    def publish(
        self,
        workflow_id: WorkflowId,
        source_name: str,
        rois: ROIs,
        geometry: ROIGeometry,
    ) -> None:
        """
        Publish ROIs to Kafka, addressed to the workflow's current job.

        Skips (with a warning) if the workflow has no active job: there is no
        backend job the ROI selection could apply to.

        Parameters
        ----------
        workflow_id:
            The workflow whose current job the ROIs apply to.
        source_name:
            The source within the workflow.
        rois:
            Dictionary mapping ROI index to ROI. Empty dict clears all.
        geometry:
            The ROI geometry configuration.
        """
        if self._job_number_resolver is None:
            raise RuntimeError("ROIPublisher used before job-number resolver was set")
        job_number = self._job_number_resolver(workflow_id)
        if job_number is None:
            logger.warning(
                "No active job for workflow %s; skipping ROI publish", workflow_id
            )
            return
        job_id = JobId(source_name=source_name, job_number=job_number)
        stream_name = f"{job_id}/{geometry.readback_key}"
        stream_id = StreamId(kind=StreamKind.LIVEDATA_ROI, name=stream_name)

        data_array = geometry.roi_class.to_concatenated_data_array(rois)

        # ROI requests carry no meaningful event-time: the selection applies to
        # data accumulated since run start, regardless of when the request was made.
        # Stamp at epoch 0 so the event-time message batcher treats the request as
        # already-current and applies it to the next processed window, instead of
        # holding it until the data watermark catches up to wall-clock-now.
        msg = Message(
            value=data_array, stream=stream_id, timestamp=Timestamp.from_ns(0)
        )
        self._sink.publish_messages([msg])

        if rois:
            logger.debug(
                "Published %d %s ROI(s) for job %s",
                len(rois),
                geometry.geometry_type,
                job_id,
            )
        else:
            logger.debug(
                "Published empty %s ROI update (cleared all) for job %s",
                geometry.geometry_type,
                job_id,
            )


class FakeROIPublisher:
    """Fake ROI publisher for testing."""

    def __init__(self):
        self.published: list[tuple[WorkflowId, str, ROIs, ROIGeometry]] = []

    def set_job_number_resolver(
        self, resolver: Callable[[WorkflowId], JobNumber | None]
    ) -> None:
        """Accepted for interface parity; the fake records without resolving."""

    def publish(
        self,
        workflow_id: WorkflowId,
        source_name: str,
        rois: ROIs,
        geometry: ROIGeometry,
    ) -> None:
        """Record published ROIs."""
        self.published.append((workflow_id, source_name, rois, geometry))

    def reset(self) -> None:
        """Clear all recorded publishes."""
        self.published.clear()
