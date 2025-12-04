# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Publisher for ROI updates to Kafka."""

import logging

from ..config.models import PolygonROI, RectangleROI
from ..config.roi_names import ROIGeometry
from ..config.workflow_spec import JobId
from ..core.message import Message, MessageSink, StreamId, StreamKind

ROIs = dict[int, RectangleROI] | dict[int, PolygonROI]


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
        geometry: ROIGeometry,
    ) -> None:
        """
        Publish ROIs to Kafka.

        Parameters
        ----------
        job_id:
            The full job identifier (source_name and job_number).
        rois:
            Dictionary mapping ROI index to ROI. Empty dict clears all.
        geometry:
            The ROI geometry configuration.
        """
        stream_name = f"{job_id}/{geometry.readback_key}"
        stream_id = StreamId(kind=StreamKind.LIVEDATA_ROI, name=stream_name)

        data_array = geometry.roi_class.to_concatenated_data_array(rois)

        msg = Message(value=data_array, stream=stream_id)
        self._sink.publish_messages([msg])

        if rois:
            self._logger.debug(
                "Published %d %s ROI(s) for job %s",
                len(rois),
                geometry.geometry_type,
                job_id,
            )
        else:
            self._logger.debug(
                "Published empty %s ROI update (cleared all) for job %s",
                geometry.geometry_type,
                job_id,
            )


class FakeROIPublisher:
    """Fake ROI publisher for testing."""

    def __init__(self):
        self.published: list[tuple[JobId, ROIs, ROIGeometry]] = []

    def publish(
        self,
        job_id: JobId,
        rois: ROIs,
        geometry: ROIGeometry,
    ) -> None:
        """Record published ROIs."""
        self.published.append((job_id, rois, geometry))

    def reset(self) -> None:
        """Clear all recorded publishes."""
        self.published.clear()
