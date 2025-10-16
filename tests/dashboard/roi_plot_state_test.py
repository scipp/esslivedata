# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unit tests for ROIPlotState."""

import logging
import uuid

import holoviews as hv
import pytest

from ess.livedata.config.models import RectangleROI
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
def box_stream():
    """Create a real BoxEdit stream for testing."""
    return hv.streams.BoxEdit()


@pytest.fixture
def fake_publisher():
    """Create a fake ROI publisher."""
    return FakeROIPublisher()


@pytest.fixture
def roi_plot_state(result_key, box_stream, fake_publisher):
    """Create a ROIPlotState for testing."""
    return ROIPlotState(
        result_key=result_key,
        box_stream=box_stream,
        x_unit='m',
        y_unit='m',
        roi_publisher=fake_publisher,
        logger=logging.getLogger(__name__),
    )


class TestROIPlotState:
    """Unit tests for ROIPlotState."""

    def test_initialization_attaches_callback_to_stream(
        self, result_key, fake_publisher
    ):
        """Test that ROIPlotState attaches callback to box_stream on init."""
        box_stream = hv.streams.BoxEdit()
        ROIPlotState(
            result_key=result_key,
            box_stream=box_stream,
            x_unit='m',
            y_unit='m',
            roi_publisher=fake_publisher,
            logger=logging.getLogger(__name__),
        )

        # Trigger an event to verify callback is attached
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Verify the callback was triggered by checking published ROIs
        assert len(fake_publisher.published_rois) == 1

    def test_publishes_valid_roi_on_box_edit(
        self, roi_plot_state, box_stream, fake_publisher
    ):
        """Test that valid ROIs are published when box is drawn."""
        # Simulate box edit event
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Assert publisher was called
        assert len(fake_publisher.published_rois) == 1
        job_id, rois_dict = fake_publisher.published_rois[0]

        # Verify job_id
        assert job_id == roi_plot_state.result_key.job_id

        # Verify ROI content
        assert len(rois_dict) == 1
        assert 0 in rois_dict
        roi = rois_dict[0]
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
        assert len(fake_publisher.published_rois) == 0

    def test_does_not_republish_unchanged_rois(
        self, roi_plot_state, box_stream, fake_publisher
    ):
        """Test that unchanged ROIs are not republished."""
        box_data = {'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]}

        # First edit
        box_stream.event(data=box_data)
        assert len(fake_publisher.published_rois) == 1

        # Second edit with same data
        box_stream.event(data=box_data)
        # Should still be 1 (not called again)
        assert len(fake_publisher.published_rois) == 1

    def test_publishes_when_roi_changes(
        self, roi_plot_state, box_stream, fake_publisher
    ):
        """Test that changed ROIs trigger republishing."""
        # First edit
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})
        assert len(fake_publisher.published_rois) == 1

        # Second edit with different data
        box_stream.event(data={'x0': [2.0], 'x1': [6.0], 'y0': [3.0], 'y1': [7.0]})
        # Should be called again
        assert len(fake_publisher.published_rois) == 2

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
        assert len(fake_publisher.published_rois) == 1
        _, rois_dict = fake_publisher.published_rois[0]

        # Verify both ROIs
        assert len(rois_dict) == 2
        assert 0 in rois_dict
        assert 1 in rois_dict

        # Verify first ROI
        roi0 = rois_dict[0]
        assert roi0.x.min == 1.0
        assert roi0.x.max == 5.0

        # Verify second ROI
        roi1 = rois_dict[1]
        assert roi1.x.min == 10.0
        assert roi1.x.max == 15.0

    def test_handles_empty_box_data(self, roi_plot_state, box_stream, fake_publisher):
        """Test handling of empty box data."""
        # First create some ROIs
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})
        assert len(fake_publisher.published_rois) == 1

        # Now send empty data (clearing ROIs)
        box_stream.event(data={})

        # Should publish empty ROI dict (because it changed from non-empty to empty)
        assert len(fake_publisher.published_rois) == 2
        _, rois_dict = fake_publisher.published_rois[1]
        assert len(rois_dict) == 0

    def test_handles_none_box_data(self, roi_plot_state, box_stream, fake_publisher):
        """Test handling of None box data."""
        # First create some ROIs
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})
        assert len(fake_publisher.published_rois) == 1

        # Now send None data (clearing ROIs)
        box_stream.event(data=None)

        # Should publish empty ROI dict (because it changed from non-empty to empty)
        assert len(fake_publisher.published_rois) == 2
        _, rois_dict = fake_publisher.published_rois[1]
        assert len(rois_dict) == 0

    def test_updates_active_roi_indices(self, roi_plot_state, box_stream):
        """Test that active ROI indices are tracked correctly."""
        # Initially no active ROIs
        assert len(roi_plot_state._active_roi_indices) == 0

        # Add one ROI
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})
        assert roi_plot_state._active_roi_indices == {0}

        # Add second ROI
        box_stream.event(
            data={
                'x0': [1.0, 10.0],
                'x1': [5.0, 15.0],
                'y0': [2.0, 20.0],
                'y1': [6.0, 25.0],
            }
        )
        assert roi_plot_state._active_roi_indices == {0, 1}

        # Remove all ROIs
        box_stream.event(data={})
        assert roi_plot_state._active_roi_indices == set()

    def test_is_roi_active(self, roi_plot_state, box_stream, result_key):
        """Test is_roi_active method."""
        # Add ROI at index 0
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Create key for ROI 0
        roi_key_0 = result_key.model_copy(update={'output_name': 'roi_current_0'})
        assert roi_plot_state.is_roi_active(roi_key_0) is True

        # Create key for ROI 1 (not active)
        roi_key_1 = result_key.model_copy(update={'output_name': 'roi_current_1'})
        assert roi_plot_state.is_roi_active(roi_key_1) is False

        # Non-ROI key
        non_roi_key = result_key.model_copy(update={'output_name': 'current'})
        assert roi_plot_state.is_roi_active(non_roi_key) is False

    def test_handles_event_with_new_attribute(
        self, roi_plot_state, box_stream, fake_publisher
    ):
        """Test handling of event object with 'new' attribute."""
        # The HoloViews stream.event() method creates proper event objects internally
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Should publish ROI
        assert len(fake_publisher.published_rois) == 1

    def test_no_publishing_when_publisher_is_none(self, result_key):
        """Test that no publishing happens when publisher is None."""
        box_stream = hv.streams.BoxEdit()
        state = ROIPlotState(
            result_key=result_key,
            box_stream=box_stream,
            x_unit='m',
            y_unit='m',
            roi_publisher=None,  # No publisher
            logger=logging.getLogger(__name__),
        )

        # Simulate box edit
        box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Active ROI indices should still be tracked
        assert state._active_roi_indices == {0}

    def test_logs_error_on_publishing_failure(self, result_key, caplog):
        """Test that errors during publishing are logged."""

        class FailingPublisher:
            """Publisher that raises an error."""

            def publish_rois(self, job_id, rois):
                raise RuntimeError("Test error")

        box_stream = hv.streams.BoxEdit()
        failing_publisher = FailingPublisher()
        ROIPlotState(
            result_key=result_key,
            box_stream=box_stream,
            x_unit='m',
            y_unit='m',
            roi_publisher=failing_publisher,
            logger=logging.getLogger(__name__),
        )

        with caplog.at_level(logging.ERROR):
            box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

        # Check error was logged
        assert len(caplog.records) == 1
        assert "Failed to publish ROI update" in caplog.records[0].message
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
        assert "Published 1 ROI(s)" in info_records[0].message
