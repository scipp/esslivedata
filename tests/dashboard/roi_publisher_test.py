# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import scipp as sc

from ess.livedata.config.models import Interval, RectangleROI
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.message import StreamKind
from ess.livedata.dashboard.roi_publisher import FakeROIPublisher, ROIPublisher
from ess.livedata.fakes import FakeMessageSink


def test_roi_publisher_publishes_single_roi():
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())
    roi = RectangleROI(
        x=Interval(min=1.0, max=5.0, unit=None),
        y=Interval(min=2.0, max=6.0, unit=None),
    )

    publisher.publish_rectangles(job_id, rois={0: roi})

    # Rectangle publish sends one message
    assert len(sink.messages) == 1
    msg = sink.messages[0]
    assert msg.stream.kind == StreamKind.LIVEDATA_ROI
    assert msg.stream.name == f"detector1/{job_id.job_number}/roi_rectangle"
    assert isinstance(msg.value, sc.DataArray)
    # Verify concatenated format
    assert 'roi_index' in msg.value.coords


def test_roi_publisher_publishes_multiple_rois():
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())
    rois = {
        0: RectangleROI(
            x=Interval(min=1.0, max=5.0, unit=None),
            y=Interval(min=2.0, max=6.0, unit=None),
        ),
        1: RectangleROI(
            x=Interval(min=10.0, max=15.0, unit=None),
            y=Interval(min=12.0, max=16.0, unit=None),
        ),
        2: RectangleROI(
            x=Interval(min=20.0, max=25.0, unit=None),
            y=Interval(min=22.0, max=26.0, unit=None),
        ),
    }

    publisher.publish_rectangles(job_id, rois=rois)

    assert len(sink.messages) == 1
    msg = sink.messages[0]
    assert msg.stream.kind == StreamKind.LIVEDATA_ROI
    assert msg.stream.name == f"detector1/{job_id.job_number}/roi_rectangle"

    # Verify all 3 ROIs are in concatenated DataArray
    da = msg.value
    import numpy as np

    np.testing.assert_array_equal(da.coords['roi_index'].values, [0, 0, 1, 1, 2, 2])


def test_roi_publisher_publishes_empty_to_clear():
    """Empty dict should publish empty DataArray to clear all ROIs."""
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())

    publisher.publish_rectangles(job_id, rois={})

    assert len(sink.messages) == 1
    msg = sink.messages[0]
    assert len(msg.value) == 0  # Empty DataArray


def test_roi_publisher_serializes_to_dataarray():
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())
    rois = {
        0: RectangleROI(
            x=Interval(min=1.5, max=5.5, unit='mm'),
            y=Interval(min=2.5, max=6.5, unit='mm'),
        ),
        1: RectangleROI(
            x=Interval(min=10.5, max=15.5, unit='mm'),
            y=Interval(min=12.5, max=16.5, unit='mm'),
        ),
    }

    publisher.publish_rectangles(job_id, rois=rois)

    msg = sink.messages[0]
    da = msg.value
    # Check that DataArray can be converted back to dict of ROIs
    recovered_rois = RectangleROI.from_concatenated_data_array(da)
    assert recovered_rois == rois


def test_fake_roi_publisher_records_publishes():
    publisher = FakeROIPublisher()
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())
    rois = {
        0: RectangleROI(
            x=Interval(min=1.0, max=5.0, unit=None),
            y=Interval(min=2.0, max=6.0, unit=None),
        ),
    }

    publisher.publish_rectangles(job_id, rois=rois)

    assert len(publisher.published_rectangles) == 1
    assert publisher.published_rectangles[0] == (job_id, rois)


def test_fake_roi_publisher_reset():
    publisher = FakeROIPublisher()
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())
    rois = {
        0: RectangleROI(
            x=Interval(min=1.0, max=5.0, unit=None),
            y=Interval(min=2.0, max=6.0, unit=None),
        ),
    }

    publisher.publish_rectangles(job_id, rois=rois)
    publisher.reset()

    assert len(publisher.published_rectangles) == 0


def test_roi_publisher_isolates_streams_per_detector_in_multi_detector_workflow():
    """
    Test that ROI streams are unique per detector in multi-detector workflows.

    When the same workflow runs on multiple detectors (same job_number),
    each detector must get its own unique ROI stream to prevent cross-talk.
    """
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)

    # Same job_number, different source_names (real multi-detector scenario)
    shared_job_number = uuid.uuid4()
    job_id_mantle = JobId(source_name='mantle', job_number=shared_job_number)
    job_id_high_res = JobId(source_name='high_resolution', job_number=shared_job_number)

    rois_mantle = {
        0: RectangleROI(
            x=Interval(min=1.0, max=5.0, unit=None),
            y=Interval(min=2.0, max=6.0, unit=None),
        ),
    }
    rois_high_res = {
        0: RectangleROI(
            x=Interval(min=10.0, max=20.0, unit=None),
            y=Interval(min=15.0, max=25.0, unit=None),
        ),
    }

    # Publish ROIs for both detectors
    publisher.publish_rectangles(job_id_mantle, rois=rois_mantle)
    publisher.publish_rectangles(job_id_high_res, rois=rois_high_res)

    assert len(sink.messages) == 2

    # Verify stream names are unique per detector
    mantle_msg = sink.messages[0]
    high_res_msg = sink.messages[1]

    assert mantle_msg.stream.name == f"mantle/{shared_job_number}/roi_rectangle"
    assert (
        high_res_msg.stream.name == f"high_resolution/{shared_job_number}/roi_rectangle"
    )
    assert mantle_msg.stream.name != high_res_msg.stream.name

    # Verify each message contains the correct ROIs
    assert RectangleROI.from_concatenated_data_array(mantle_msg.value) == rois_mantle
    assert (
        RectangleROI.from_concatenated_data_array(high_res_msg.value) == rois_high_res
    )
