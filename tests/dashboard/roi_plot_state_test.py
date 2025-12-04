# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unit tests for ROIPlotState."""

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
    """Create a real BoxEdit stream for testing."""
    return hv.streams.BoxEdit()


@pytest.fixture
def fake_publisher():
    """Create a fake ROI publisher."""
    return FakeROIPublisher()


@pytest.fixture
def roi_plot_state(result_key, box_stream, boxes_pipe, fake_publisher):
    """Create a ROIPlotState for testing."""
    # Initialize holoviews extension to populate default color cycles
    hv.extension('bokeh')

    # Create ROI state stream for tracking active ROIs
    class ROIStateStream(hv.streams.Stream):
        active_rois = param.Parameter(default=set(), doc="Set of active ROI indices")

    roi_state_stream = ROIStateStream()

    # Create separate pipes for request and readback layers
    rect_request_pipe = hv.streams.Pipe(data=[])
    rect_readback_pipe = boxes_pipe

    # Polygon streams and pipes
    poly_stream = hv.streams.PolyDraw()
    poly_request_pipe = hv.streams.Pipe(data=[])
    poly_readback_pipe = hv.streams.Pipe(data=[])

    default_colors = hv.Cycle.default_cycles["default_colors"]
    return ROIPlotState(
        result_key=result_key,
        box_stream=box_stream,
        rect_request_pipe=rect_request_pipe,
        rect_readback_pipe=rect_readback_pipe,
        poly_stream=poly_stream,
        poly_request_pipe=poly_request_pipe,
        poly_readback_pipe=poly_readback_pipe,
        roi_state_stream=roi_state_stream,
        x_unit='m',
        y_unit='m',
        roi_publisher=fake_publisher,
        logger=logging.getLogger(__name__),
        colors=default_colors[:10],
    )


class TestROIPlotState:
    """Unit tests for ROIPlotState."""

    def test_initialization_attaches_callback_to_stream(
        self, result_key, fake_publisher
    ):
        """Test that ROIPlotState attaches callback to box_stream on init."""
        hv.extension('bokeh')

        rect_readback_pipe = hv.streams.Pipe(data=[])
        rect_request_pipe = hv.streams.Pipe(data=[])
        poly_stream = hv.streams.PolyDraw()
        poly_readback_pipe = hv.streams.Pipe(data=[])
        poly_request_pipe = hv.streams.Pipe(data=[])
        box_stream = hv.streams.BoxEdit()
        default_colors = hv.Cycle.default_cycles["default_colors"]

        # Create ROI state stream
        class ROIStateStream(hv.streams.Stream):
            active_rois = param.Parameter(
                default=set(), doc="Set of active ROI indices"
            )

        roi_state_stream = ROIStateStream()

        ROIPlotState(
            result_key=result_key,
            box_stream=box_stream,
            rect_request_pipe=rect_request_pipe,
            rect_readback_pipe=rect_readback_pipe,
            poly_stream=poly_stream,
            poly_request_pipe=poly_request_pipe,
            poly_readback_pipe=poly_readback_pipe,
            roi_state_stream=roi_state_stream,
            x_unit='m',
            y_unit='m',
            roi_publisher=fake_publisher,
            logger=logging.getLogger(__name__),
            colors=default_colors[:10],
        )

        # Trigger an event to verify callback is attached
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Verify the callback was triggered by checking published ROIs
        assert len(fake_publisher.published) == 1

    def test_publishes_valid_roi_on_box_edit(
        self, roi_plot_state, box_stream, fake_publisher
    ):
        """Test that valid ROIs are published when box is drawn."""
        # Simulate box edit event
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Assert publisher was called
        assert len(fake_publisher.published) == 1
        job_id, rect_rois_dict, _ = fake_publisher.published[0]

        # Verify job_id
        assert job_id == roi_plot_state.result_key.job_id

        # Verify ROI content
        assert len(rect_rois_dict) == 1
        assert 0 in rect_rois_dict
        roi = rect_rois_dict[0]
        assert isinstance(roi, RectangleROI)
        assert roi.x.min == 1.0
        assert roi.x.max == 5.0
        assert roi.y.min == 2.0
        assert roi.y.max == 6.0
        assert roi.x.unit == 'm'
        assert roi.y.unit == 'm'

    def test_filters_out_zero_area_rois(self, box_stream, fake_publisher):
        """Test that zero-area ROIs are filtered out."""
        # Simulate box with zero area (x0 == x1)
        box_stream.event(data={'x0': [1.0], 'x1': [1.0], 'y0': [2.0], 'y1': [6.0]})

        # Assert nothing was published (empty ROIs dict)
        assert len(fake_publisher.published) == 0

    def test_does_not_republish_unchanged_rois(
        self, roi_plot_state, box_stream, fake_publisher
    ):
        """Test that unchanged ROIs are not republished."""
        box_data = {'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]}

        # First edit
        box_stream.event(data=box_data)
        assert len(fake_publisher.published) == 1

        # Second edit with same data
        box_stream.event(data=box_data)
        # Should still be 1 (not called again)
        assert len(fake_publisher.published) == 1

    def test_publishes_when_roi_changes(
        self, roi_plot_state, box_stream, fake_publisher
    ):
        """Test that changed ROIs trigger republishing."""
        # First edit
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})
        assert len(fake_publisher.published) == 1

        # Second edit with different data
        box_stream.event(data={'x0': [2.0], 'x1': [6.0], 'y0': [3.0], 'y1': [7.0]})
        # Should be called again
        assert len(fake_publisher.published) == 2

    def test_handles_multiple_rois(self, roi_plot_state, box_stream, fake_publisher):
        """Test publishing multiple ROIs."""
        box_stream.event(
            data={
                'x0': [1.0, 10.0],
                'x1': [5.0, 15.0],
                'y0': [2.0, 20.0],
                'y1': [6.0, 25.0],
            }
        )

        # Assert publisher was called
        assert len(fake_publisher.published) == 1
        _, rect_rois_dict, _ = fake_publisher.published[0]

        # Verify both ROIs
        assert len(rect_rois_dict) == 2
        assert 0 in rect_rois_dict
        assert 1 in rect_rois_dict

        # Verify first ROI
        roi0 = rect_rois_dict[0]
        assert roi0.x.min == 1.0
        assert roi0.x.max == 5.0

        # Verify second ROI
        roi1 = rect_rois_dict[1]
        assert roi1.x.min == 10.0
        assert roi1.x.max == 15.0

    def test_handles_empty_box_data(self, roi_plot_state, box_stream, fake_publisher):
        """Test handling of empty box data."""
        # First create some ROIs
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})
        assert len(fake_publisher.published) == 1

        # Now send empty data (clearing ROIs)
        box_stream.event(data={})

        # Should publish empty ROI dict (because it changed from non-empty to empty)
        assert len(fake_publisher.published) == 2
        _, rect_rois_dict, _ = fake_publisher.published[1]
        assert len(rect_rois_dict) == 0

    def test_handles_none_box_data(self, roi_plot_state, box_stream, fake_publisher):
        """Test handling of None box data."""
        # First create some ROIs
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})
        assert len(fake_publisher.published) == 1

        # Now send None data (clearing ROIs)
        box_stream.event(data=None)

        # Should publish empty ROI dict (because it changed from non-empty to empty)
        assert len(fake_publisher.published) == 2
        _, rect_rois_dict, _ = fake_publisher.published[1]
        assert len(rect_rois_dict) == 0

    def test_is_roi_active(self, roi_plot_state, box_stream, result_key):
        """Test is_roi_active method tracks backend ROI state correctly."""
        # Initially no ROIs are active
        roi_key_0 = result_key.model_copy(update={'output_name': 'roi_current_0'})
        roi_key_1 = result_key.model_copy(update={'output_name': 'roi_current_1'})
        assert roi_plot_state.is_roi_active(roi_key_0) is False
        assert roi_plot_state.is_roi_active(roi_key_1) is False

        # Add ROI at index 0 via user edit
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})
        # Simulate backend readback
        roi_plot_state.on_backend_rect_update(
            {
                0: RectangleROI(
                    x=Interval(min=1.0, max=5.0, unit='m'),
                    y=Interval(min=2.0, max=6.0, unit='m'),
                )
            }
        )

        # ROI 0 should be active, ROI 1 should not
        assert roi_plot_state.is_roi_active(roi_key_0) is True
        assert roi_plot_state.is_roi_active(roi_key_1) is False

        # Add second ROI
        box_stream.event(
            data={
                'x0': [1.0, 10.0],
                'x1': [5.0, 15.0],
                'y0': [2.0, 20.0],
                'y1': [6.0, 25.0],
            }
        )
        roi_plot_state.on_backend_rect_update(
            {
                0: RectangleROI(
                    x=Interval(min=1.0, max=5.0, unit='m'),
                    y=Interval(min=2.0, max=6.0, unit='m'),
                ),
                1: RectangleROI(
                    x=Interval(min=10.0, max=15.0, unit='m'),
                    y=Interval(min=20.0, max=25.0, unit='m'),
                ),
            }
        )

        # Both ROIs should be active
        assert roi_plot_state.is_roi_active(roi_key_0) is True
        assert roi_plot_state.is_roi_active(roi_key_1) is True

        # Remove all ROIs
        box_stream.event(data={})
        roi_plot_state.on_backend_rect_update({})

        # No ROIs should be active
        assert roi_plot_state.is_roi_active(roi_key_0) is False
        assert roi_plot_state.is_roi_active(roi_key_1) is False

        # Non-ROI key should never be considered active
        non_roi_key = result_key.model_copy(update={'output_name': 'current'})
        assert roi_plot_state.is_roi_active(non_roi_key) is False

    def test_handles_event_with_new_attribute(
        self, roi_plot_state, box_stream, fake_publisher
    ):
        """Test handling of event object with 'new' attribute."""
        # The HoloViews stream.event() method creates proper event objects internally
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Should publish ROI
        assert len(fake_publisher.published) == 1

    def test_no_publishing_when_publisher_is_none(self, result_key):
        """Test that ROI state works correctly when publisher is None."""
        hv.extension('bokeh')

        rect_readback_pipe = hv.streams.Pipe(data=[])
        rect_request_pipe = hv.streams.Pipe(data=[])
        poly_stream = hv.streams.PolyDraw()
        poly_readback_pipe = hv.streams.Pipe(data=[])
        poly_request_pipe = hv.streams.Pipe(data=[])
        box_stream = hv.streams.BoxEdit()
        default_colors = hv.Cycle.default_cycles["default_colors"]

        # Create ROI state stream
        class ROIStateStream(hv.streams.Stream):
            active_rois = param.Parameter(
                default=set(), doc="Set of active ROI indices"
            )

        roi_state_stream = ROIStateStream()

        state = ROIPlotState(
            result_key=result_key,
            box_stream=box_stream,
            rect_request_pipe=rect_request_pipe,
            rect_readback_pipe=rect_readback_pipe,
            poly_stream=poly_stream,
            poly_request_pipe=poly_request_pipe,
            poly_readback_pipe=poly_readback_pipe,
            roi_state_stream=roi_state_stream,
            x_unit='m',
            y_unit='m',
            roi_publisher=None,  # No publisher
            logger=logging.getLogger(__name__),
            colors=default_colors[:10],
        )

        # Simulate box edit - should not crash
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Simulate backend readback - should not crash
        state.on_backend_rect_update(
            {
                0: RectangleROI(
                    x=Interval(min=1.0, max=5.0, unit='m'),
                    y=Interval(min=2.0, max=6.0, unit='m'),
                )
            }
        )

        # Verify ROI is tracked via public API
        roi_key = result_key.model_copy(update={'output_name': 'roi_current_0'})
        assert state.is_roi_active(roi_key) is True

    def test_logs_error_on_publishing_failure(self, result_key, caplog):
        """Test that errors during publishing are logged."""
        hv.extension('bokeh')

        class FailingPublisher:
            """Publisher that raises an error."""

            def publish(self, job_id, rois, geometry):
                raise RuntimeError("Test error")

        rect_readback_pipe = hv.streams.Pipe(data=[])
        rect_request_pipe = hv.streams.Pipe(data=[])
        poly_stream = hv.streams.PolyDraw()
        poly_readback_pipe = hv.streams.Pipe(data=[])
        poly_request_pipe = hv.streams.Pipe(data=[])
        box_stream = hv.streams.BoxEdit()
        failing_publisher = FailingPublisher()
        default_colors = hv.Cycle.default_cycles["default_colors"]

        # Create ROI state stream
        class ROIStateStream(hv.streams.Stream):
            active_rois = param.Parameter(
                default=set(), doc="Set of active ROI indices"
            )

        roi_state_stream = ROIStateStream()

        ROIPlotState(
            result_key=result_key,
            box_stream=box_stream,
            rect_request_pipe=rect_request_pipe,
            rect_readback_pipe=rect_readback_pipe,
            poly_stream=poly_stream,
            poly_request_pipe=poly_request_pipe,
            poly_readback_pipe=poly_readback_pipe,
            roi_state_stream=roi_state_stream,
            x_unit='m',
            y_unit='m',
            roi_publisher=failing_publisher,
            logger=logging.getLogger(__name__),
            colors=default_colors[:10],
        )

        with caplog.at_level(logging.ERROR):
            box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Check error was logged
        assert len(caplog.records) == 1
        assert "Failed to publish" in caplog.records[0].message
        assert "ROI update" in caplog.records[0].message
        assert "Test error" in caplog.records[0].message

    def test_logs_info_on_successful_publish(
        self, roi_plot_state, box_stream, fake_publisher, caplog
    ):
        """Test that successful publishing is logged."""
        with caplog.at_level(logging.INFO):
            box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Check info was logged
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) == 1
        assert "Published 1 rectangle ROI(s)" in info_records[0].message
