# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Tests for backend ROI updates (on_backend_rect_update).

These tests verify that programmatic ROI updates from the backend correctly
update both the visual representation (via Pipe) and the BoxEdit stream state
to enable proper drag operations.
"""

import logging
import uuid

import holoviews as hv
import param
import pytest

from ess.livedata.config.models import Interval, RectangleROI
from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.roi_detector_plot_factory import ROIPlotState
from ess.livedata.dashboard.roi_publisher import FakeROIPublisher


@pytest.fixture
def workflow_id():
    """Create a test WorkflowId."""
    return WorkflowId(
        instrument='test_instrument',
        namespace='test_namespace',
        name='test_workflow',
        version=1,
    )


@pytest.fixture
def job_id():
    """Create a test JobId."""
    return JobId(source_name='detector_data', job_number=uuid.uuid4())


@pytest.fixture
def result_key(workflow_id, job_id):
    """Create a test ResultKey."""
    return ResultKey(
        workflow_id=workflow_id,
        job_id=job_id,
        output_name='current',
    )


@pytest.fixture
def boxes_pipe():
    """Create a Pipe stream for testing."""
    return hv.streams.Pipe(data=[])


@pytest.fixture
def box_stream():
    """Create a BoxEdit stream for testing."""
    return hv.streams.BoxEdit()


@pytest.fixture
def fake_publisher():
    """Create a fake ROI publisher."""
    return FakeROIPublisher()


@pytest.fixture
def roi_plot_state(result_key, box_stream, boxes_pipe, fake_publisher):
    """Create a ROIPlotState for testing."""
    # Use a fixed set of colors for testing (independent of holoviews config)
    test_colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
    ]
    # boxes_pipe fixture is now used as rect_readback_pipe for backward compatibility
    # with tests that check pipe updates
    rect_request_pipe = hv.streams.Pipe(data=[])
    poly_stream = hv.streams.PolyDraw()
    poly_request_pipe = hv.streams.Pipe(data=[])
    poly_readback_pipe = hv.streams.Pipe(data=[])

    # Create ROI state stream for tracking active ROIs
    class ROIStateStream(hv.streams.Stream):
        active_rois = param.Parameter(default=set(), doc="Set of active ROI indices")

    roi_state_stream = ROIStateStream()

    return ROIPlotState(
        result_key=result_key,
        box_stream=box_stream,
        rect_request_pipe=rect_request_pipe,
        rect_readback_pipe=boxes_pipe,  # reuse boxes_pipe fixture for readback
        poly_stream=poly_stream,
        poly_request_pipe=poly_request_pipe,
        poly_readback_pipe=poly_readback_pipe,
        roi_state_stream=roi_state_stream,
        x_unit='m',
        y_unit='m',
        roi_publisher=fake_publisher,
        logger=logging.getLogger(__name__),
        colors=test_colors,
    )


class TestBackendROIUpdate:
    """Tests for on_backend_rect_update method."""

    def test_backend_update_updates_pipe(self, roi_plot_state, boxes_pipe):
        """Test that backend update sends data to pipe."""
        backend_rois = {
            0: RectangleROI(
                x=Interval(min=1.0, max=3.0, unit='m'),
                y=Interval(min=2.0, max=4.0, unit='m'),
            )
        }

        # Track pipe updates
        pipe_updates = []
        original_send = boxes_pipe.send

        def track_send(data):
            pipe_updates.append(data)
            original_send(data)

        boxes_pipe.send = track_send

        roi_plot_state.on_backend_rect_update(backend_rois)

        # Verify pipe was updated with color included
        assert len(pipe_updates) == 1
        assert len(pipe_updates[0]) == 1
        assert pipe_updates[0][0][:4] == (1.0, 2.0, 3.0, 4.0)
        # Fifth element should be the color (from colors list at index 0)
        assert isinstance(pipe_updates[0][0][4], str)  # Color is a string

    def test_backend_update_updates_box_stream(self, roi_plot_state, box_stream):
        """Test that backend update triggers BoxEdit event with dict format."""
        backend_rois = {
            0: RectangleROI(
                x=Interval(min=1.0, max=3.0, unit='m'),
                y=Interval(min=2.0, max=4.0, unit='m'),
            )
        }

        roi_plot_state.on_backend_rect_update(backend_rois)

        # Verify BoxEdit has correct dict format data
        assert box_stream.data is not None
        assert isinstance(box_stream.data, dict)
        assert 'x0' in box_stream.data
        assert box_stream.data['x0'] == [1.0]
        assert box_stream.data['y0'] == [2.0]
        assert box_stream.data['x1'] == [3.0]
        assert box_stream.data['y1'] == [4.0]

    def test_backend_update_coordinates_are_floats(self, roi_plot_state, box_stream):
        """Test that backend update ensures coordinates are floats, not ints."""
        backend_rois = {
            0: RectangleROI(
                x=Interval(min=1.0, max=3.0, unit='m'),
                y=Interval(min=2.0, max=4.0, unit='m'),
            )
        }

        roi_plot_state.on_backend_rect_update(backend_rois)

        # Verify all coordinates are float type
        assert isinstance(box_stream.data['x0'][0], float)
        assert isinstance(box_stream.data['y0'][0], float)
        assert isinstance(box_stream.data['x1'][0], float)
        assert isinstance(box_stream.data['y1'][0], float)

    def test_backend_update_with_multiple_rois(
        self, roi_plot_state, box_stream, boxes_pipe
    ):
        """Test backend update with multiple ROIs."""
        backend_rois = {
            0: RectangleROI(
                x=Interval(min=1.0, max=3.0, unit='m'),
                y=Interval(min=2.0, max=4.0, unit='m'),
            ),
            1: RectangleROI(
                x=Interval(min=5.0, max=7.0, unit='m'),
                y=Interval(min=6.0, max=8.0, unit='m'),
            ),
        }

        roi_plot_state.on_backend_rect_update(backend_rois)

        # Verify BoxEdit has all ROIs
        assert len(box_stream.data['x0']) == 2
        assert box_stream.data['x0'] == [1.0, 5.0]
        assert box_stream.data['y0'] == [2.0, 6.0]
        assert box_stream.data['x1'] == [3.0, 7.0]
        assert box_stream.data['y1'] == [4.0, 8.0]

    def test_backend_update_with_empty_rois(self, roi_plot_state, box_stream):
        """Test backend update with empty ROI dict clears existing ROIs."""
        # First, set some ROIs
        initial_rois = {
            0: RectangleROI(
                x=Interval(min=1.0, max=3.0, unit='m'),
                y=Interval(min=2.0, max=4.0, unit='m'),
            )
        }
        roi_plot_state.on_backend_rect_update(initial_rois)
        assert len(box_stream.data['x0']) == 1

        # Now clear them
        roi_plot_state.on_backend_rect_update({})

        assert box_stream.data is not None
        assert len(box_stream.data['x0']) == 0

    def test_backend_update_does_not_trigger_republish(
        self, roi_plot_state, fake_publisher
    ):
        """Test that backend updates don't cause republishing to backend."""
        backend_rois = {
            0: RectangleROI(
                x=Interval(min=1.0, max=3.0, unit='m'),
                y=Interval(min=2.0, max=4.0, unit='m'),
            )
        }

        initial_publish_count = len(fake_publisher.published)
        roi_plot_state.on_backend_rect_update(backend_rois)

        # Should not have published back to backend
        assert len(fake_publisher.published) == initial_publish_count

    def test_backend_update_no_update_if_rois_unchanged(
        self, roi_plot_state, box_stream, boxes_pipe
    ):
        """Test that backend update is skipped if ROIs haven't changed."""
        backend_rois = {
            0: RectangleROI(
                x=Interval(min=1.0, max=3.0, unit='m'),
                y=Interval(min=2.0, max=4.0, unit='m'),
            )
        }

        # First update
        roi_plot_state.on_backend_rect_update(backend_rois)

        # Track pipe updates
        pipe_updates = []
        original_send = boxes_pipe.send

        def track_send(data):
            pipe_updates.append(data)
            original_send(data)

        boxes_pipe.send = track_send

        # Second update with same data
        roi_plot_state.on_backend_rect_update(backend_rois)

        # Should not have updated pipe (no change)
        assert len(pipe_updates) == 0

    def test_backend_update_updates_active_roi_indices(self, roi_plot_state):
        """Test that backend update updates active ROI indices."""
        backend_rois = {
            0: RectangleROI(
                x=Interval(min=1.0, max=3.0, unit='m'),
                y=Interval(min=2.0, max=4.0, unit='m'),
            ),
            1: RectangleROI(
                x=Interval(min=5.0, max=7.0, unit='m'),
                y=Interval(min=6.0, max=8.0, unit='m'),
            ),
        }

        roi_plot_state.on_backend_rect_update(backend_rois)

        assert roi_plot_state._active_roi_indices == {0, 1}

    def test_backend_update_preserves_roi_order_by_index(
        self, roi_plot_state, box_stream
    ):
        """Test that ROIs are ordered by index in backend update."""
        backend_rois = {
            2: RectangleROI(
                x=Interval(min=5.0, max=7.0, unit='m'),
                y=Interval(min=6.0, max=8.0, unit='m'),
            ),
            0: RectangleROI(
                x=Interval(min=1.0, max=3.0, unit='m'),
                y=Interval(min=2.0, max=4.0, unit='m'),
            ),
            1: RectangleROI(
                x=Interval(min=3.0, max=4.0, unit='m'),
                y=Interval(min=4.0, max=5.0, unit='m'),
            ),
        }

        roi_plot_state.on_backend_rect_update(backend_rois)

        # Should be sorted by index: 0, 1, 2
        assert box_stream.data['x0'] == [1.0, 3.0, 5.0]
        assert box_stream.data['y0'] == [2.0, 4.0, 6.0]

    def test_backend_update_handles_exception_gracefully(self, roi_plot_state, caplog):
        """Test that exceptions in backend update are logged and don't crash."""
        # Pass invalid data type that will cause an error
        with caplog.at_level(logging.ERROR):
            roi_plot_state.on_backend_rect_update(None)  # type: ignore[arg-type]

        # Should have logged error
        assert "Failed to update UI from backend rectangle ROI data" in caplog.text


class TestBidirectionalSync:
    """Tests for bidirectional synchronization between UI and backend."""

    def test_user_edit_then_backend_update_different_roi(
        self, roi_plot_state, box_stream, fake_publisher
    ):
        """Test user edit followed by backend update of different ROI."""
        # User creates ROI 0
        box_stream.event(data={'x0': [1.0], 'x1': [3.0], 'y0': [2.0], 'y1': [4.0]})
        assert len(fake_publisher.published) == 1

        # Backend updates with ROI 1 (different index)
        backend_rois = {
            1: RectangleROI(
                x=Interval(min=5.0, max=7.0, unit='m'),
                y=Interval(min=6.0, max=8.0, unit='m'),
            )
        }
        roi_plot_state.on_backend_rect_update(backend_rois)

        # Should now have ROI 1 only (backend is source of truth)
        assert box_stream.data['x0'] == [5.0]

    def test_backend_update_then_user_edit(
        self, roi_plot_state, box_stream, fake_publisher
    ):
        """Test backend update followed by user edit."""
        # Backend creates ROI
        backend_rois = {
            0: RectangleROI(
                x=Interval(min=1.0, max=3.0, unit='m'),
                y=Interval(min=2.0, max=4.0, unit='m'),
            )
        }
        roi_plot_state.on_backend_rect_update(backend_rois)

        initial_publish_count = len(fake_publisher.published)

        # User modifies the ROI
        box_stream.event(
            data={'x0': [1.5], 'x1': [3.5], 'y0': [2.5], 'y1': [4.5]}  # Changed coords
        )

        # Should have published the change
        assert len(fake_publisher.published) == initial_publish_count + 1

    def test_backend_update_after_user_creates_same_roi(
        self, roi_plot_state, box_stream, fake_publisher
    ):
        """Test backend echoing back the same ROI user just created."""
        # User creates ROI
        box_stream.event(data={'x0': [1.0], 'x1': [3.0], 'y0': [2.0], 'y1': [4.0]})
        assert len(fake_publisher.published) == 1

        # Backend echoes it back (same coordinates)
        backend_rois = {
            0: RectangleROI(
                x=Interval(min=1.0, max=3.0, unit='m'),
                y=Interval(min=2.0, max=4.0, unit='m'),
            )
        }
        initial_publish_count = len(fake_publisher.published)
        roi_plot_state.on_backend_rect_update(backend_rois)

        # Should NOT republish (no change detected)
        assert len(fake_publisher.published) == initial_publish_count
