# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pytest
import scipp as sc

from ess.livedata.config.models import Interval, RectangleROI
from ess.livedata.core.message import StreamKind
from ess.livedata.dashboard.roi_publisher import (
    FakeROIPublisher,
    ROIPublisher,
    boxes_to_rois,
)
from ess.livedata.fakes import FakeMessageSink


def test_boxes_to_rois_converts_single_box():
    box_data = {'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]}

    rois = boxes_to_rois(box_data)

    assert len(rois) == 1
    assert 0 in rois
    roi = rois[0]
    assert roi.x.min == 1.0
    assert roi.x.max == 5.0
    assert roi.y.min == 2.0
    assert roi.y.max == 6.0
    assert roi.x.unit is None
    assert roi.y.unit is None


def test_boxes_to_rois_converts_multiple_boxes():
    box_data = {
        'x0': [1.0, 10.0, 20.0],
        'x1': [5.0, 15.0, 25.0],
        'y0': [2.0, 12.0, 22.0],
        'y1': [6.0, 16.0, 26.0],
    }

    rois = boxes_to_rois(box_data)

    assert len(rois) == 3
    assert rois[0].x.min == 1.0
    assert rois[1].x.min == 10.0
    assert rois[2].x.min == 20.0


def test_boxes_to_rois_handles_inverted_coordinates():
    # BoxEdit can return boxes with x0 > x1 or y0 > y1
    box_data = {'x0': [5.0], 'x1': [1.0], 'y0': [6.0], 'y1': [2.0]}

    rois = boxes_to_rois(box_data)

    roi = rois[0]
    assert roi.x.min == 1.0
    assert roi.x.max == 5.0
    assert roi.y.min == 2.0
    assert roi.y.max == 6.0


def test_boxes_to_rois_skips_degenerate_boxes():
    # Boxes with zero width or height should be skipped
    box_data = {
        'x0': [1.0, 5.0, 10.0],
        'x1': [5.0, 5.0, 15.0],  # Second box has zero width
        'y0': [2.0, 6.0, 10.0],
        'y1': [6.0, 10.0, 10.0],  # Third box has zero height
    }

    rois = boxes_to_rois(box_data)

    assert len(rois) == 1
    assert 0 in rois
    assert rois[0].x.min == 1.0


def test_boxes_to_rois_empty_data():
    assert boxes_to_rois({}) == {}
    assert boxes_to_rois({'x0': []}) == {}


def test_boxes_to_rois_raises_on_inconsistent_lengths():
    box_data = {'x0': [1.0, 2.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]}

    with pytest.raises(ValueError, match="inconsistent lengths"):
        boxes_to_rois(box_data)


def test_roi_publisher_publishes_single_roi():
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)
    job_number = uuid.uuid4()
    roi = RectangleROI(
        x=Interval(min=1.0, max=5.0, unit=None),
        y=Interval(min=2.0, max=6.0, unit=None),
    )

    publisher.publish_roi(job_number, roi_index=0, roi=roi)

    assert len(sink.messages) == 1
    msg = sink.messages[0]
    assert msg.stream.kind == StreamKind.LIVEDATA_ROI
    assert msg.stream.name == f"{job_number}/roi_rectangle_0"
    assert isinstance(msg.value, sc.DataArray)


def test_roi_publisher_publishes_multiple_rois():
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)
    job_number = uuid.uuid4()
    rois = {
        0: RectangleROI(
            x=Interval(min=1.0, max=5.0, unit=None),
            y=Interval(min=2.0, max=6.0, unit=None),
        ),
        1: RectangleROI(
            x=Interval(min=10.0, max=15.0, unit=None),
            y=Interval(min=12.0, max=16.0, unit=None),
        ),
    }

    publisher.publish_rois(job_number, rois)

    assert len(sink.messages) == 2
    stream_names = {msg.stream.name for msg in sink.messages}
    assert f"{job_number}/roi_rectangle_0" in stream_names
    assert f"{job_number}/roi_rectangle_1" in stream_names


def test_roi_publisher_serializes_to_dataarray():
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)
    job_number = uuid.uuid4()
    roi = RectangleROI(
        x=Interval(min=1.5, max=5.5, unit=None),
        y=Interval(min=2.5, max=6.5, unit=None),
    )

    publisher.publish_roi(job_number, roi_index=0, roi=roi)

    msg = sink.messages[0]
    da = msg.value
    # Check that DataArray can be converted back to RectangleROI
    recovered_roi = RectangleROI.from_data_array(da)
    assert recovered_roi == roi


def test_fake_roi_publisher_records_publishes():
    publisher = FakeROIPublisher()
    job_number = uuid.uuid4()
    roi = RectangleROI(
        x=Interval(min=1.0, max=5.0, unit=None),
        y=Interval(min=2.0, max=6.0, unit=None),
    )

    publisher.publish_roi(job_number, roi_index=0, roi=roi)

    assert len(publisher.published_rois) == 1
    assert publisher.published_rois[0] == (job_number, 0, roi)


def test_fake_roi_publisher_reset():
    publisher = FakeROIPublisher()
    job_number = uuid.uuid4()
    roi = RectangleROI(
        x=Interval(min=1.0, max=5.0, unit=None),
        y=Interval(min=2.0, max=6.0, unit=None),
    )

    publisher.publish_roi(job_number, roi_index=0, roi=roi)
    publisher.reset()

    assert len(publisher.published_rois) == 0
