# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Publisher for ROI updates to Kafka."""

import logging

from ..config.models import PolygonROI, RectangleROI
from ..config.workflow_spec import JobId
from ..core.message import Message, MessageSink, StreamId, StreamKind

ROIs = dict[int, RectangleROI] | dict[int, PolygonROI]
ROIClass = type[RectangleROI] | type[PolygonROI]

# Map ROI class to stream key suffix
_READBACK_KEYS: dict[ROIClass, str] = {
    RectangleROI: "roi_rectangle",
    PolygonROI: "roi_polygon",
}


class ROIPublisher:
    """
    Publishes ROI updates to Kafka.

    This class provides a simple interface for publishing ROIs to the
    LIVEDATA_ROI Kafka topic.

    Parameters
    ----------
    sink:
        Message sink for publishing messages.
    logger:
        Logger instance. If None, creates a logger using the module name.
    """

    def __init__(self, sink: MessageSink, logger: logging.Logger | None = None):
        self._sink = sink
        self._logger = logger or logging.getLogger(__name__)

    def publish(
        self,
        job_id: JobId,
        rois: ROIs,
        roi_class: ROIClass,
    ) -> None:
        """
        Publish ROIs to Kafka.

        Parameters
        ----------
        job_id:
            The full job identifier (source_name and job_number).
        rois:
            Dictionary mapping ROI index to ROI. Empty dict clears all.
        roi_class:
            The ROI class (RectangleROI or PolygonROI).
        """
        readback_key = _READBACK_KEYS[roi_class]
        stream_name = f"{job_id}/{readback_key}"
        stream_id = StreamId(kind=StreamKind.LIVEDATA_ROI, name=stream_name)

        data_array = roi_class.to_concatenated_data_array(rois)

        msg = Message(value=data_array, stream=stream_id)
        self._sink.publish_messages([msg])


class FakeROIPublisher:
    """Fake ROI publisher for testing."""

    def __init__(self):
        self.published: list[tuple[JobId, ROIs, ROIClass]] = []

    def publish(
        self,
        job_id: JobId,
        rois: ROIs,
        roi_class: ROIClass,
    ) -> None:
        """Record published ROIs."""
        self.published.append((job_id, rois, roi_class))

    def reset(self) -> None:
        """Clear all recorded publishes."""
        self.published.clear()
