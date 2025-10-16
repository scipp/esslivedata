# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import holoviews as hv
import pytest
import scipp as sc

from ess.livedata.config.models import Interval, RectangleROI
from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_params import (
    PlotParamsROIDetector,
    PlotScaleParams2d,
)
from ess.livedata.dashboard.roi_detector_plot_factory import (
    ROIDetectorPlotFactory,
    boxes_to_rois,
    rois_to_rectangles,
)
from ess.livedata.dashboard.stream_manager import StreamManager

hv.extension('bokeh')


def make_job_number() -> uuid.UUID:
    """Generate a random UUID for job number."""
    return uuid.uuid4()


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
def job_number():
    """Create a test job number."""
    return make_job_number()


@pytest.fixture
def job_id(job_number):
    """Create a test JobId."""
    return JobId(source_name='detector_data', job_number=job_number)


@pytest.fixture
def data_service():
    """Create a DataService for testing."""
    return DataService()


@pytest.fixture
def job_service(data_service):
    """Create a JobService for testing."""
    return JobService(data_service=data_service)


@pytest.fixture
def stream_manager(data_service):
    """Create a StreamManager for testing."""
    return StreamManager(data_service=data_service, pipe_factory=hv.streams.Pipe)


@pytest.fixture
def roi_plot_factory(stream_manager):
    """Create a ROIDetectorPlotFactory for testing."""
    return ROIDetectorPlotFactory(
        stream_manager=stream_manager,
    )


@pytest.fixture
def detector_data():
    """Create 2D detector data."""
    x = sc.arange('x', 10, dtype='float64')
    y = sc.arange('y', 8, dtype='float64')
    data = sc.arange('y', 0, 80, dtype='float64').fold(dim='y', sizes={'y': 8, 'x': 10})
    return sc.DataArray(data, coords={'x': x, 'y': y})


@pytest.fixture
def spectrum_data():
    """Create 1D ROI spectrum data."""
    tof = sc.linspace('tof', 0.0, 100.0, num=50, unit='us')
    return sc.DataArray(
        sc.arange('tof', 50, dtype='float64', unit='counts'), coords={'tof': tof}
    )


def get_detector_pipe(
    data_service: DataService, detector_key: ResultKey
) -> hv.streams.Pipe:
    """
    Get detector pipe from data_service for testing.

    Parameters
    ----------
    data_service:
        The data service containing the data.
    detector_key:
        ResultKey for the detector output.

    Returns
    -------
    :
        Pipe stream with detector data.
    """
    if detector_key in data_service:
        return data_service[detector_key]
    raise KeyError(f"Key {detector_key} not found in data service")


class TestROIDetectorPlotFactory:
    def test_create_roi_detector_plot_components_returns_detector_and_spectrum(
        self,
        roi_plot_factory,
        data_service,
        workflow_id,
        job_number,
        detector_data,
        spectrum_data,
    ):
        """Test create_roi_detector_plot_components returns detector and spectrum."""
        # Create result keys for detector and spectrum
        # Using 'current' as the output_name, which will look for 'roi_current'
        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='current',
        )
        spectrum_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='roi_current_0',
        )

        # Add data to data service
        data_service[detector_key] = detector_data
        data_service[spectrum_key] = spectrum_data

        # Create plot params
        params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())

        # Create ROI detector plot components
        detector_with_boxes, roi_spectrum, plot_state = (
            roi_plot_factory.create_roi_detector_plot_components(
                detector_key=detector_key,
                detector_data=detector_data,
                params=params,
            )
        )

        # Should return detector and spectrum components
        assert isinstance(detector_with_boxes, hv.Overlay | hv.DynamicMap)
        assert isinstance(roi_spectrum, hv.DynamicMap)
        assert plot_state is not None

    def test_create_roi_detector_plot_components_with_only_detector(
        self,
        roi_plot_factory,
        data_service,
        workflow_id,
        job_number,
        detector_data,
    ):
        """Test ROI detector plot components with only detector data (no spectrum)."""
        # Create result key for detector only (spectrum doesn't exist yet)
        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='current',
        )

        # Add data to data service
        data_service[detector_key] = detector_data

        # Create plot params
        params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())

        # Create ROI detector plot components
        detector_with_boxes, roi_spectrum, plot_state = (
            roi_plot_factory.create_roi_detector_plot_components(
                detector_key=detector_key,
                detector_data=detector_data,
                params=params,
            )
        )

        # Should create components even without spectrum data
        assert isinstance(detector_with_boxes, hv.Overlay | hv.DynamicMap)
        assert isinstance(roi_spectrum, hv.DynamicMap)
        assert plot_state is not None

    def test_create_roi_detector_plot_components_returns_valid_components(
        self,
        roi_plot_factory,
        data_service,
        workflow_id,
        job_number,
        detector_data,
        spectrum_data,
    ):
        """Test that create_roi_detector_plot_components returns valid components."""
        # Create result keys
        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='current',
        )
        spectrum_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='roi_current_0',
        )

        # Add data to data service
        data_service[detector_key] = detector_data
        data_service[spectrum_key] = spectrum_data

        # Create plot params
        params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())

        # Create components using public API
        detector_dmap, roi_dmap, plot_state = (
            roi_plot_factory.create_roi_detector_plot_components(
                detector_key=detector_key,
                detector_data=detector_data,
                params=params,
            )
        )

        # Verify components are returned correctly
        assert isinstance(detector_dmap, hv.Overlay | hv.DynamicMap)
        assert isinstance(roi_dmap, hv.DynamicMap)
        assert plot_state is not None
        assert isinstance(plot_state.box_stream, hv.streams.BoxEdit)


def test_roi_detector_plot_publishes_roi_on_box_edit(
    roi_plot_factory, data_service, workflow_id, job_number
):
    """Integration test: BoxEdit changes trigger ROI publishing."""
    from ess.livedata.dashboard.roi_publisher import FakeROIPublisher

    # Set up fake publisher
    fake_publisher = FakeROIPublisher()
    roi_plot_factory._roi_publisher = fake_publisher

    # Add detector data output
    detector_data = sc.DataArray(
        sc.array(dims=['y', 'x'], values=[[1, 2], [3, 4]]),
        coords={'x': sc.arange('x', 2), 'y': sc.arange('y', 2)},
    )
    detector_key = ResultKey(
        workflow_id=workflow_id,
        job_id=JobId(source_name='detector_data', job_number=job_number),
        output_name='current',
    )
    data_service[detector_key] = detector_data

    # Create plot params
    params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())

    # Create ROI detector plot components using public API
    _detector_dmap, _roi_dmap, plot_state = (
        roi_plot_factory.create_roi_detector_plot_components(
            detector_key=detector_key,
            detector_data=detector_data,
            params=params,
        )
    )

    # Simulate user drawing a box via the box stream
    box_stream = plot_state.box_stream
    box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})

    # Check that ROI was published
    assert len(fake_publisher.published_rois) == 1
    published_job_id, rois_dict = fake_publisher.published_rois[0]
    assert published_job_id.job_number == job_number
    assert published_job_id.source_name == 'detector_data'
    assert len(rois_dict) == 1
    assert 0 in rois_dict
    roi = rois_dict[0]
    assert roi.x.min == 1.0
    assert roi.x.max == 5.0
    assert roi.y.min == 2.0
    assert roi.y.max == 6.0


def test_roi_detector_plot_only_publishes_changed_rois(
    roi_plot_factory, data_service, workflow_id, job_number
):
    """Integration test: ROI publishing only happens when ROI changes."""
    from ess.livedata.dashboard.roi_publisher import FakeROIPublisher

    # Set up fake publisher
    fake_publisher = FakeROIPublisher()
    roi_plot_factory._roi_publisher = fake_publisher

    # Add detector data
    detector_data = sc.DataArray(
        sc.array(dims=['y', 'x'], values=[[1, 2], [3, 4]]),
        coords={'x': sc.arange('x', 2), 'y': sc.arange('y', 2)},
    )
    detector_key = ResultKey(
        workflow_id=workflow_id,
        job_id=JobId(source_name='detector_data', job_number=job_number),
        output_name='current',
    )
    data_service[detector_key] = detector_data

    # Create plot components using public API
    params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())
    _detector_dmap, _roi_dmap, plot_state = (
        roi_plot_factory.create_roi_detector_plot_components(
            detector_key=detector_key,
            detector_data=detector_data,
            params=params,
        )
    )

    # Get the box stream
    box_stream = plot_state.box_stream

    # First box edit
    box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})
    assert len(fake_publisher.published_rois) == 1

    # Trigger same box again - should not publish duplicate
    box_stream.event(data={'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]})
    assert len(fake_publisher.published_rois) == 1  # Still 1

    # Change the box - should publish
    box_stream.event(data={'x0': [2.0], 'x1': [6.0], 'y0': [3.0], 'y1': [7.0]})
    assert len(fake_publisher.published_rois) == 2


def test_roi_detector_plot_without_publisher_does_not_crash(
    roi_plot_factory, data_service, workflow_id, job_number
):
    """Test that ROI plot works without a publisher configured."""
    # Ensure no publisher is set
    roi_plot_factory._roi_publisher = None

    # Add detector data
    detector_data = sc.DataArray(
        sc.array(dims=['y', 'x'], values=[[1, 2], [3, 4]]),
        coords={'x': sc.arange('x', 2), 'y': sc.arange('y', 2)},
    )
    detector_key = ResultKey(
        workflow_id=workflow_id,
        job_id=JobId(source_name='detector_data', job_number=job_number),
        output_name='current',
    )
    data_service[detector_key] = detector_data

    # Create plot components - should not crash
    params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())
    detector_with_boxes, roi_spectrum, plot_state = (
        roi_plot_factory.create_roi_detector_plot_components(
            detector_key=detector_key,
            detector_data=detector_data,
            params=params,
        )
    )

    # Should create components successfully
    assert isinstance(detector_with_boxes, hv.Overlay | hv.DynamicMap)
    assert isinstance(roi_spectrum, hv.DynamicMap)
    assert plot_state is not None


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

    with pytest.raises(
        ValueError, match="zip\\(\\) argument .* is (shorter|longer) than argument"
    ):
        boxes_to_rois(box_data)


def test_boxes_to_rois_with_units():
    box_data = {'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]}

    rois = boxes_to_rois(box_data, x_unit='m', y_unit='mm')

    assert len(rois) == 1
    roi = rois[0]
    assert roi.x.min == 1.0
    assert roi.x.max == 5.0
    assert roi.y.min == 2.0
    assert roi.y.max == 6.0
    assert roi.x.unit == 'm'
    assert roi.y.unit == 'mm'


def test_boxes_to_rois_preserves_units_across_multiple_boxes():
    box_data = {
        'x0': [1.0, 10.0],
        'x1': [5.0, 15.0],
        'y0': [2.0, 12.0],
        'y1': [6.0, 16.0],
    }

    rois = boxes_to_rois(box_data, x_unit='angstrom', y_unit='angstrom')

    assert len(rois) == 2
    assert rois[0].x.unit == 'angstrom'
    assert rois[0].y.unit == 'angstrom'
    assert rois[1].x.unit == 'angstrom'
    assert rois[1].y.unit == 'angstrom'


def test_rois_to_rectangles_converts_single_roi():
    rois = {
        0: RectangleROI(
            x=Interval(min=1.0, max=5.0, unit='m'),
            y=Interval(min=2.0, max=6.0, unit='mm'),
        )
    }

    rectangles = rois_to_rectangles(rois)

    assert len(rectangles) == 1
    assert rectangles[0] == (1.0, 2.0, 5.0, 6.0)


def test_rois_to_rectangles_converts_multiple_rois():
    rois = {
        0: RectangleROI(x=Interval(min=1.0, max=5.0), y=Interval(min=2.0, max=6.0)),
        1: RectangleROI(x=Interval(min=10.0, max=15.0), y=Interval(min=12.0, max=16.0)),
        2: RectangleROI(x=Interval(min=20.0, max=25.0), y=Interval(min=22.0, max=26.0)),
    }

    rectangles = rois_to_rectangles(rois)

    assert len(rectangles) == 3
    assert rectangles[0] == (1.0, 2.0, 5.0, 6.0)
    assert rectangles[1] == (10.0, 12.0, 15.0, 16.0)
    assert rectangles[2] == (20.0, 22.0, 25.0, 26.0)


def test_rois_to_rectangles_empty():
    assert rois_to_rectangles({}) == []


def test_rois_to_rectangles_sorts_by_index():
    rois = {
        2: RectangleROI(x=Interval(min=20.0, max=25.0), y=Interval(min=22.0, max=26.0)),
        0: RectangleROI(x=Interval(min=1.0, max=5.0), y=Interval(min=2.0, max=6.0)),
        1: RectangleROI(x=Interval(min=10.0, max=15.0), y=Interval(min=12.0, max=16.0)),
    }

    rectangles = rois_to_rectangles(rois)

    # Should be in sorted order by index (0, 1, 2)
    assert rectangles[0] == (1.0, 2.0, 5.0, 6.0)
    assert rectangles[1] == (10.0, 12.0, 15.0, 16.0)
    assert rectangles[2] == (20.0, 22.0, 25.0, 26.0)


def test_create_roi_plot_with_initial_rois(
    roi_plot_factory, data_service, workflow_id, job_number, detector_data
):
    """Test that ROI plot can be initialized with existing ROI configurations."""
    from ess.livedata.config.models import ROI

    detector_key = ResultKey(
        workflow_id=workflow_id,
        job_id=JobId(source_name='detector_data', job_number=job_number),
        output_name='current',
    )

    initial_rois = {
        0: RectangleROI(x=Interval(min=2.0, max=6.0), y=Interval(min=3.0, max=5.0)),
        1: RectangleROI(x=Interval(min=7.0, max=9.0), y=Interval(min=4.0, max=6.0)),
    }

    # Inject ROI readback data into DataService - this simulates backend publishing ROIs
    roi_readback_key = detector_key.model_copy(update={"output_name": "roi_rectangle"})
    roi_readback_data = ROI.to_concatenated_data_array(initial_rois)
    data_service[roi_readback_key] = roi_readback_data

    params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())

    _detector_with_boxes, _roi_dmap, plot_state = (
        roi_plot_factory.create_roi_detector_plot_components(
            detector_key=detector_key,
            detector_data=detector_data,
            params=params,
        )
    )

    # Verify the plot state has the correct active ROI indices
    assert plot_state._active_roi_indices == {0, 1}

    # Verify the BoxEdit stream was initialized with the rectangles
    box_data = plot_state.box_stream.data
    assert len(box_data['x0']) == 2
    assert box_data['x0'][0] == 2.0
    assert box_data['x1'][0] == 6.0
    assert box_data['y0'][0] == 3.0
    assert box_data['y1'][0] == 5.0


def test_custom_max_roi_count(roi_plot_factory, detector_data, workflow_id, job_number):
    """Test that max_roi_count parameter is correctly applied to BoxEdit."""
    detector_key = ResultKey(
        workflow_id=workflow_id,
        job_id=JobId(source_name='detector_data', job_number=job_number),
        output_name='current',
    )

    # Create params with custom max_roi_count
    params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())
    params.roi_options.max_roi_count = 5

    _detector_with_boxes, _roi_dmap, plot_state = (
        roi_plot_factory.create_roi_detector_plot_components(
            detector_key=detector_key,
            detector_data=detector_data,
            params=params,
        )
    )

    # Verify the BoxEdit was configured with the custom max_roi_count
    box_stream = plot_state.box_stream
    assert box_stream.num_objects == 5
    assert len(box_stream.styles['fill_color']) == 5
