# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Publisher for ROI updates to Kafka."""

import logging

from ..config.models import RectangleROI
from ..config.workflow_spec import JobId
from ..core.message import Message, StreamId, StreamKind
from ..kafka.sink import KafkaSink


class ROIPublisher:
    """
    Publishes ROI updates to Kafka.

    This class provides a simple interface for publishing ROI rectangles to the
    LIVEDATA_ROI Kafka topic.

    Parameters
    ----------
    sink:
        Kafka sink for publishing messages.
    logger:
        Logger instance. If None, creates a logger using the module name.
    """

    def __init__(self, sink: KafkaSink, logger: logging.Logger | None = None):
        self._sink = sink
        self._logger = logger or logging.getLogger(__name__)

    def publish_rois(self, job_id: JobId, rois: dict[int, RectangleROI]) -> None:
        """
        Publish all ROI rectangles as single concatenated message.

        All rectangles are sent as a single DataArray with concatenated bounds
        and an roi_index coordinate identifying individual ROIs. This allows
        the backend to detect ROI deletions (missing indices).

        Parameters
        ----------
        job_id:
            The full job identifier (source_name and job_number).
        rois:
            Dictionary mapping ROI index to RectangleROI. Empty dict clears all ROIs.
        """
        # Use singular 'rectangle' to match DetectorROIAuxSources field name
        # (the concatenated DataArray is what makes it plural conceptually)
        stream_name = f"{job_id}/roi_rectangle"
        stream_id = StreamId(kind=StreamKind.LIVEDATA_ROI, name=stream_name)

        # Convert all ROIs to single concatenated DataArray
        data_array = RectangleROI.to_concatenated_data_array(rois)

        msg = Message(value=data_array, stream=stream_id)
        self._sink.publish_messages([msg])

        if rois:
            roi_summary = ", ".join(
                f"{idx}: x=[{roi.x.min}, {roi.x.max}], y=[{roi.y.min}, {roi.y.max}]"
                for idx, roi in sorted(rois.items())
            )
            self._logger.debug(
                "Published %d ROI rectangle(s) for job %s: %s",
                len(rois),
                job_id,
                roi_summary,
            )
        else:
            self._logger.debug(
                "Published empty ROI update (cleared all) for job %s", job_id
            )


class FakeROIPublisher:
    """Fake ROI publisher for testing."""

    def __init__(self):
        self.published_rois: list[tuple[JobId, dict[int, RectangleROI]]] = []

    def publish_rois(self, job_id: JobId, rois: dict[int, RectangleROI]) -> None:
        """Record published ROI collection."""
        self.published_rois.append((job_id, rois))

    def reset(self) -> None:
        """Clear all recorded publishes."""
        self.published_rois.clear()
