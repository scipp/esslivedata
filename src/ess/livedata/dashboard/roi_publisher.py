# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Publisher for ROI updates to Kafka."""

import logging

from ..config.models import PolygonROI, RectangleROI
from ..config.roi_names import get_roi_mapper
from ..config.workflow_spec import JobId
from ..core.message import Message, MessageSink, StreamId, StreamKind


class ROIPublisher:
    """
    Publishes ROI updates to Kafka.

    This class provides a simple interface for publishing ROI rectangles to the
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
        self._roi_mapper = get_roi_mapper()

    def publish_rois(
        self,
        job_id: JobId,
        rectangle_rois: dict[int, RectangleROI],
        polygon_rois: dict[int, PolygonROI] | None = None,
    ) -> None:
        """
        Publish all ROIs (rectangles and polygons) as concatenated messages.

        Each geometry type is sent as a separate message to its own stream.
        ROIs are sent as DataArrays with concatenated coordinates and an
        roi_index coordinate identifying individual ROIs. This allows the
        backend to detect ROI deletions (missing indices).

        Parameters
        ----------
        job_id:
            The full job identifier (source_name and job_number).
        rectangle_rois:
            Dictionary mapping ROI index to RectangleROI. Empty dict clears all.
        polygon_rois:
            Dictionary mapping ROI index to PolygonROI. None means no changes.
        """
        # Publish rectangles
        self._publish_geometry(
            job_id, "roi_rectangle", rectangle_rois, RectangleROI, "rectangle"
        )

        # Publish polygons if provided
        if polygon_rois is not None:
            self._publish_geometry(
                job_id, "roi_polygon", polygon_rois, PolygonROI, "polygon"
            )

    def _publish_geometry(
        self,
        job_id: JobId,
        readback_key: str,
        rois: dict[int, RectangleROI] | dict[int, PolygonROI],
        roi_class: type[RectangleROI] | type[PolygonROI],
        geometry_name: str,
    ) -> None:
        """Publish a single geometry type to Kafka."""
        stream_name = f"{job_id}/{readback_key}"
        stream_id = StreamId(kind=StreamKind.LIVEDATA_ROI, name=stream_name)

        # Convert all ROIs to single concatenated DataArray
        data_array = roi_class.to_concatenated_data_array(rois)

        msg = Message(value=data_array, stream=stream_id)
        self._sink.publish_messages([msg])

        if rois:
            self._logger.debug(
                "Published %d ROI %s(s) for job %s",
                len(rois),
                geometry_name,
                job_id,
            )
        else:
            self._logger.debug(
                "Published empty ROI %s update (cleared all) for job %s",
                geometry_name,
                job_id,
            )


class FakeROIPublisher:
    """Fake ROI publisher for testing."""

    def __init__(self):
        self.published_rois: list[
            tuple[JobId, dict[int, RectangleROI], dict[int, PolygonROI] | None]
        ] = []

    def publish_rois(
        self,
        job_id: JobId,
        rectangle_rois: dict[int, RectangleROI],
        polygon_rois: dict[int, PolygonROI] | None = None,
    ) -> None:
        """Record published ROI collection."""
        self.published_rois.append((job_id, rectangle_rois, polygon_rois))

    def reset(self) -> None:
        """Clear all recorded publishes."""
        self.published_rois.clear()
