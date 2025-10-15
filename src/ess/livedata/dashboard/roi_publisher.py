# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Publisher for ROI updates to Kafka."""

import logging
from typing import Any

from ..config.models import RectangleROI
from ..config.workflow_spec import JobId
from ..core.message import Message, StreamId, StreamKind
from ..handlers.detector_data_handler import DetectorROIAuxSources
from ..kafka.sink import KafkaSink


class ROIPublisher:
    """
    Publishes ROI updates to Kafka.

    This class provides a simple interface for publishing ROI rectangles to the
    LIVEDATA_ROI Kafka topic. It follows the same pattern as LogProducerWidget
    for unidirectional publishing.

    Parameters
    ----------
    sink:
        Kafka sink for publishing messages.
    logger:
        Logger instance. If None, creates a logger using the module name.
    """

    def __init__(
        self,
        sink: KafkaSink,
        logger: logging.Logger | None = None,
    ):
        self._sink = sink
        self._logger = logger or logging.getLogger(__name__)

    def publish_roi(self, job_id: JobId, roi_index: int, roi: RectangleROI) -> None:
        """
        Publish a single ROI rectangle update.

        Parameters
        ----------
        job_id:
            The full job identifier (source_name and job_number).
        roi_index:
            The index of the ROI rectangle (0-based).
        roi:
            The rectangle ROI to publish.
        """
        if roi_index != 0:
            raise NotImplementedError("Multiple ROIs are not implemented")

        # Create the aux sources model and use it to render the stream name
        aux_model = DetectorROIAuxSources(roi='rectangle')
        rendered = aux_model.render(job_id)
        stream_name = rendered['roi']

        stream_id = StreamId(kind=StreamKind.LIVEDATA_ROI, name=stream_name)

        # Convert ROI to DataArray (includes ROI type in the name field)
        data_array = roi.to_data_array()

        msg = Message(value=data_array, stream=stream_id)
        self._sink.publish_messages([msg])

        self._logger.debug(
            "Published ROI rectangle %d for job %s: x=[%s, %s], y=[%s, %s]",
            roi_index,
            job_id,
            roi.x.min,
            roi.x.max,
            roi.y.min,
            roi.y.max,
        )

    def publish_rois(self, job_id: JobId, rois: dict[int, RectangleROI]) -> None:
        """
        Publish multiple ROI rectangles.

        Parameters
        ----------
        job_id:
            The full job identifier (source_name and job_number).
        rois:
            Dictionary mapping ROI index to RectangleROI.
        """
        for roi_index, roi in rois.items():
            self.publish_roi(job_id, roi_index, roi)


class FakeROIPublisher:
    """Fake ROI publisher for testing."""

    def __init__(self):
        self.published_rois: list[tuple[JobId, int, RectangleROI]] = []

    def publish_roi(self, job_id: JobId, roi_index: int, roi: RectangleROI) -> None:
        """Record published ROI."""
        self.published_rois.append((job_id, roi_index, roi))

    def publish_rois(self, job_id: JobId, rois: dict[int, RectangleROI]) -> None:
        """Record multiple published ROIs."""
        for roi_index, roi in rois.items():
            self.publish_roi(job_id, roi_index, roi)

    def reset(self) -> None:
        """Clear all recorded publishes."""
        self.published_rois.clear()


def boxes_to_rois(
    box_data: dict[str, Any],
    x_unit: str | None = None,
    y_unit: str | None = None,
) -> dict[int, RectangleROI]:
    """
    Convert BoxEdit data dictionary to RectangleROI instances.

    BoxEdit returns data as a dictionary with keys 'x0', 'x1', 'y0', 'y1',
    where each value is a list of coordinates for all boxes.

    Parameters
    ----------
    box_data:
        Dictionary from BoxEdit stream with keys x0, x1, y0, y1.
    x_unit:
        Unit for x coordinates (from the detector data coordinates).
    y_unit:
        Unit for y coordinates (from the detector data coordinates).

    Returns
    -------
    :
        Dictionary mapping box index to RectangleROI. Empty boxes are skipped.
    """
    if not box_data or not box_data.get('x0'):
        return {}

    x0_list = box_data.get('x0', [])
    x1_list = box_data.get('x1', [])
    y0_list = box_data.get('y0', [])
    y1_list = box_data.get('y1', [])

    # Validate all lists have the same length
    lengths = {len(x0_list), len(x1_list), len(y0_list), len(y1_list)}
    if len(lengths) != 1:
        raise ValueError(
            f"BoxEdit data has inconsistent lengths: "
            f"x0={len(x0_list)}, x1={len(x1_list)}, "
            f"y0={len(y0_list)}, y1={len(y1_list)}"
        )

    rois = {}
    for i, (x0, x1, y0, y1) in enumerate(
        zip(x0_list, x1_list, y0_list, y1_list, strict=True)
    ):
        # Skip empty/invalid boxes (where corners are equal)
        if x0 == x1 or y0 == y1:
            continue

        # Ensure min < max
        x_min, x_max = (x0, x1) if x0 < x1 else (x1, x0)
        y_min, y_max = (y0, y1) if y0 < y1 else (y1, y0)

        from ..config.models import Interval

        rois[i] = RectangleROI(
            x=Interval(min=float(x_min), max=float(x_max), unit=x_unit),
            y=Interval(min=float(y_min), max=float(y_max), unit=y_unit),
        )

    return rois
