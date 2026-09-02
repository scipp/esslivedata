# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from ess.livedata.config.models import Interval, RectangleROI
from ess.livedata.config.roi_names import ROIGeometry
from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.message import StreamKind
from ess.livedata.dashboard.roi_publisher import FakeROIPublisher, ROIPublisher
from ess.livedata.fakes import FakeMessageSink

# Geometry fixtures for tests
RECT_GEOMETRY = ROIGeometry(geometry_type="rectangle", num_rois=4, index_offset=0)

WORKFLOW_ID = WorkflowId(instrument='test', name='workflow', version=1)
RECT_STREAM = 'test/workflow/1/detector1/roi_rectangle'


def make_roi(offset: float = 0.0, unit: str | None = None) -> RectangleROI:
    return RectangleROI(
        x=Interval(min=1.0 + offset, max=5.0 + offset, unit=unit),
        y=Interval(min=2.0 + offset, max=6.0 + offset, unit=unit),
    )


def test_roi_publisher_publishes_single_roi():
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)

    publisher.publish(
        WORKFLOW_ID, 'detector1', rois={0: make_roi()}, geometry=RECT_GEOMETRY
    )

    assert len(sink.messages) == 1
    msg = sink.messages[0]
    assert msg.stream.kind == StreamKind.LIVEDATA_ROI
    assert msg.stream.name == RECT_STREAM
    assert isinstance(msg.value, sc.DataArray)
    assert 'roi_index' in msg.value.coords


def test_roi_publisher_stamps_request_with_epoch_zero():
    """ROI requests are clock-independent and carry the epoch-0 sentinel.

    This lets the event-time message batcher apply the selection to the current
    window instead of holding it until the data watermark reaches wall-clock-now.
    """
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)

    publisher.publish(
        WORKFLOW_ID, 'detector1', rois={0: make_roi()}, geometry=RECT_GEOMETRY
    )

    assert sink.messages[0].timestamp.to_ns() == 0


def test_roi_publisher_publishes_multiple_rois():
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)
    rois = {i: make_roi(offset=10.0 * i) for i in range(3)}

    publisher.publish(WORKFLOW_ID, 'detector1', rois=rois, geometry=RECT_GEOMETRY)

    assert len(sink.messages) == 1
    msg = sink.messages[0]
    assert msg.stream.kind == StreamKind.LIVEDATA_ROI
    assert msg.stream.name == RECT_STREAM

    # Verify all 3 ROIs are in concatenated DataArray
    np.testing.assert_array_equal(
        msg.value.coords['roi_index'].values, [0, 0, 1, 1, 2, 2]
    )


def test_roi_publisher_publishes_empty_to_clear():
    """Empty dict should publish empty DataArray to clear all ROIs."""
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)

    publisher.publish(WORKFLOW_ID, 'detector1', rois={}, geometry=RECT_GEOMETRY)

    assert len(sink.messages) == 1
    assert len(sink.messages[0].value) == 0  # Empty DataArray


def test_roi_publisher_serializes_to_dataarray():
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)
    rois = {0: make_roi(offset=0.5, unit='mm'), 1: make_roi(offset=10.5, unit='mm')}

    publisher.publish(WORKFLOW_ID, 'detector1', rois=rois, geometry=RECT_GEOMETRY)

    da = sink.messages[0].value
    # Check that DataArray can be converted back to dict of ROIs
    recovered_rois = RectangleROI.from_concatenated_data_array(da)
    assert recovered_rois == rois


def test_roi_publisher_publishes_without_a_running_job():
    """A selection made before the job starts must still reach the backend.

    The backend latches ROI requests per view, so there is nothing to wait for.
    """
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)

    publisher.publish(
        WORKFLOW_ID, 'detector1', rois={0: make_roi()}, geometry=RECT_GEOMETRY
    )

    assert len(sink.messages) == 1
    assert sink.messages[0].stream.name == RECT_STREAM


def test_roi_publisher_reuses_the_stream_name_across_publishes():
    """The name carries no job generation, so a restart keeps addressing it."""
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)

    publisher.publish(
        WORKFLOW_ID, 'detector1', rois={0: make_roi()}, geometry=RECT_GEOMETRY
    )
    publisher.publish(
        WORKFLOW_ID, 'detector1', rois={0: make_roi(offset=1.0)}, geometry=RECT_GEOMETRY
    )

    assert {msg.stream.name for msg in sink.messages} == {RECT_STREAM}


def test_roi_publisher_isolates_streams_per_workflow_sharing_a_source():
    """Two ROI-supporting views of one detector must not share a stream."""
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)
    other = WorkflowId(instrument='test', name='other_view', version=1)

    publisher.publish(
        WORKFLOW_ID, 'detector1', rois={0: make_roi()}, geometry=RECT_GEOMETRY
    )
    publisher.publish(other, 'detector1', rois={0: make_roi()}, geometry=RECT_GEOMETRY)

    assert sink.messages[0].stream.name != sink.messages[1].stream.name


def test_fake_roi_publisher_records_publishes():
    publisher = FakeROIPublisher()
    rois = {0: make_roi()}

    publisher.publish(WORKFLOW_ID, 'detector1', rois=rois, geometry=RECT_GEOMETRY)

    assert publisher.published == [(WORKFLOW_ID, 'detector1', rois, RECT_GEOMETRY)]


def test_fake_roi_publisher_reset():
    publisher = FakeROIPublisher()

    publisher.publish(
        WORKFLOW_ID, 'detector1', rois={0: make_roi()}, geometry=RECT_GEOMETRY
    )
    publisher.reset()

    assert len(publisher.published) == 0


def test_roi_publisher_isolates_streams_per_detector_in_multi_detector_workflow():
    """
    Test that ROI streams are unique per detector in multi-detector workflows.

    When the same workflow runs on multiple detectors, each detector must get
    its own unique ROI stream to prevent cross-talk.
    """
    sink = FakeMessageSink()
    publisher = ROIPublisher(sink=sink)

    rois_mantle = {0: make_roi()}
    rois_high_res = {0: make_roi(offset=9.0)}

    publisher.publish(WORKFLOW_ID, 'mantle', rois=rois_mantle, geometry=RECT_GEOMETRY)
    publisher.publish(
        WORKFLOW_ID, 'high_resolution', rois=rois_high_res, geometry=RECT_GEOMETRY
    )

    assert len(sink.messages) == 2

    # Verify stream names are unique per detector
    mantle_msg = sink.messages[0]
    high_res_msg = sink.messages[1]

    assert mantle_msg.stream.name == 'test/workflow/1/mantle/roi_rectangle'
    assert high_res_msg.stream.name == 'test/workflow/1/high_resolution/roi_rectangle'
    assert mantle_msg.stream.name != high_res_msg.stream.name

    # Verify each message contains the correct ROIs
    assert RectangleROI.from_concatenated_data_array(mantle_msg.value) == rois_mantle
    assert (
        RectangleROI.from_concatenated_data_array(high_res_msg.value) == rois_high_res
    )
