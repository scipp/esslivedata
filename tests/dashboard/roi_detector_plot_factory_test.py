# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import holoviews as hv
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_params import PlotParams2d, PlotScaleParams2d
from ess.livedata.dashboard.roi_detector_plot_factory import ROIDetectorPlotFactory
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
    def test_create_roi_detector_plot_returns_layout(
        self,
        roi_plot_factory,
        data_service,
        workflow_id,
        job_number,
        detector_data,
        spectrum_data,
    ):
        """Test that create_roi_detector_plot returns a Layout."""
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
        params = PlotParams2d(plot_scale=PlotScaleParams2d())

        # Create ROI detector plot
        result = roi_plot_factory.create_roi_detector_plot(
            detector_key=detector_key,
            detector_data=detector_data,
            params=params,
        )

        # Should return a Layout
        assert isinstance(result, hv.Layout)
        # Should have 2 elements: detector image + spectrum
        assert len(result) == 2

    def test_create_roi_detector_plot_with_only_detector(
        self,
        roi_plot_factory,
        data_service,
        workflow_id,
        job_number,
        detector_data,
    ):
        """Test ROI detector plot with only detector data (no spectrum)."""
        # Create result key for detector only (spectrum doesn't exist yet)
        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='current',
        )

        # Add data to data service
        data_service[detector_key] = detector_data

        # Create plot params
        params = PlotParams2d(plot_scale=PlotScaleParams2d())

        # Create ROI detector plot
        result = roi_plot_factory.create_roi_detector_plot(
            detector_key=detector_key,
            detector_data=detector_data,
            params=params,
        )

        # Should return a Layout with 2 elements (detector + empty spectrum placeholder)
        assert isinstance(result, hv.Layout)
        assert len(result) == 2

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
        params = PlotParams2d(plot_scale=PlotScaleParams2d())

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

    def test_create_roi_detector_plot_separates_detector_and_spectrum(
        self,
        roi_plot_factory,
        data_service,
        workflow_id,
        job_number,
        detector_data,
        spectrum_data,
    ):
        """Test that detector and spectrum data are properly separated."""
        # Create result keys
        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='cumulative',
        )
        spectrum_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='roi_cumulative_0',
        )

        # Add data to data service
        data_service[detector_key] = detector_data
        data_service[spectrum_key] = spectrum_data

        # Create plot params
        params = PlotParams2d(plot_scale=PlotScaleParams2d())

        # Create ROI detector plot
        result = roi_plot_factory.create_roi_detector_plot(
            detector_key=detector_key,
            detector_data=detector_data,
            params=params,
        )

        # Layout should have 2 elements
        assert len(result) == 2

        # First element should be detector (DynamicMap overlaid with Rectangles)
        # Second element should be spectrum (DynamicMap)
        # Both should be wrapped in overlays or be DynamicMaps
        assert isinstance(result[0], hv.Overlay | hv.DynamicMap)
        assert isinstance(result[1], hv.DynamicMap)


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
    params = PlotParams2d(plot_scale=PlotScaleParams2d())

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
    params = PlotParams2d(plot_scale=PlotScaleParams2d())
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

    # Create plot - should not crash
    params = PlotParams2d(plot_scale=PlotScaleParams2d())
    result = roi_plot_factory.create_roi_detector_plot(
        detector_key=detector_key,
        detector_data=detector_data,
        params=params,
    )

    # Should create plot successfully
    assert isinstance(result, hv.Layout)
