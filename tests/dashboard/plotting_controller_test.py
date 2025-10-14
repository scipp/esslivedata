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
from ess.livedata.dashboard.plotting_controller import PlottingController
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
def plotting_controller(job_service, stream_manager):
    """Create a PlottingController for testing."""
    return PlottingController(
        job_service=job_service,
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


class TestPlottingControllerROIDetector:
    def test_create_roi_detector_plot_returns_layout(
        self,
        plotting_controller,
        job_service,
        data_service,
        workflow_id,
        job_number,
        detector_data,
        spectrum_data,
    ):
        """Test that _create_roi_detector_plot returns a Layout."""
        # Create result keys for detector and spectrum
        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='detector_image',
        )
        spectrum_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='roi_spectrum',
        )

        # Add data to data service (job service will pick it up automatically)
        data_service[detector_key] = detector_data
        data_service[spectrum_key] = spectrum_data

        # Create plot params
        params = PlotParams2d(plot_scale=PlotScaleParams2d())

        # Create ROI detector plot
        result = plotting_controller._create_roi_detector_plot(
            job_number=job_number,
            source_names=['detector_data'],
            params=params,
        )

        # Should return a Layout
        assert isinstance(result, hv.Layout)
        # Should have 2 elements: detector image + spectrum
        assert len(result) == 2

    def test_create_roi_detector_plot_with_only_detector(
        self,
        plotting_controller,
        job_service,
        data_service,
        workflow_id,
        job_number,
        detector_data,
    ):
        """Test ROI detector plot with only detector data (no spectrum)."""
        # Create result key for detector only
        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='detector_image',
        )

        # Add data to data service
        data_service[detector_key] = detector_data

        # Create plot params
        params = PlotParams2d(plot_scale=PlotScaleParams2d())

        # Create ROI detector plot
        result = plotting_controller._create_roi_detector_plot(
            job_number=job_number,
            source_names=['detector_data'],
            params=params,
        )

        # Should return a Layout with 1 element (detector only)
        assert isinstance(result, hv.Layout)
        assert len(result) == 1

    def test_create_roi_detector_plot_stores_box_stream(
        self,
        plotting_controller,
        job_service,
        data_service,
        workflow_id,
        job_number,
        detector_data,
        spectrum_data,
    ):
        """Test that BoxEdit stream is stored in controller."""
        # Create result keys
        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='detector_image',
        )
        spectrum_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='roi_spectrum',
        )

        # Add data to data service
        data_service[detector_key] = detector_data
        data_service[spectrum_key] = spectrum_data

        # Create plot params
        params = PlotParams2d(plot_scale=PlotScaleParams2d())

        # Verify no box streams initially
        assert len(plotting_controller._box_streams) == 0

        # Create ROI detector plot
        plotting_controller._create_roi_detector_plot(
            job_number=job_number,
            source_names=['detector_data'],
            params=params,
        )

        # Should have stored box stream
        assert len(plotting_controller._box_streams) == 1
        assert detector_key in plotting_controller._box_streams
        assert isinstance(
            plotting_controller._box_streams[detector_key], hv.streams.BoxEdit
        )

    def test_create_plot_with_roi_detector_plot_name(
        self,
        plotting_controller,
        job_service,
        data_service,
        workflow_id,
        job_number,
        detector_data,
        spectrum_data,
    ):
        """Test create_plot with roi_detector plot_name."""
        # Create result keys
        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='detector_image',
        )
        spectrum_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='roi_spectrum',
        )

        # Add data to data service
        data_service[detector_key] = detector_data
        data_service[spectrum_key] = spectrum_data

        # Create plot params
        params = PlotParams2d(plot_scale=PlotScaleParams2d())

        # Create plot using create_plot method with roi_detector plot_name
        result = plotting_controller.create_plot(
            job_number=job_number,
            source_names=['detector_data'],
            output_name=None,
            plot_name='roi_detector',
            params=params,
        )

        # Should return a Layout (not DynamicMap)
        assert isinstance(result, hv.Layout)
        assert len(result) == 2

    def test_create_plot_roi_detector_validates_params_type(
        self,
        plotting_controller,
        job_service,
        data_service,
        workflow_id,
        job_number,
        detector_data,
    ):
        """Test that roi_detector validates params is PlotParams2d."""
        # Add detector data
        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='detector_image',
        )
        data_service[detector_key] = detector_data

        # Try to create plot with wrong params type (using a simple pydantic model)
        import pydantic

        class WrongParams(pydantic.BaseModel):
            value: int = 42

        wrong_params = WrongParams()

        # Should raise TypeError
        with pytest.raises(TypeError, match="roi_detector requires PlotParams2d"):
            plotting_controller.create_plot(
                job_number=job_number,
                source_names=['detector_data'],
                output_name=None,
                plot_name='roi_detector',
                params=wrong_params,
            )

    def test_create_roi_detector_plot_separates_detector_and_spectrum(
        self,
        plotting_controller,
        job_service,
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
            output_name='detector_image',
        )
        spectrum_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_data', job_number=job_number),
            output_name='roi_spectrum',
        )

        # Add data to data service
        data_service[detector_key] = detector_data
        data_service[spectrum_key] = spectrum_data

        # Create plot params
        params = PlotParams2d(plot_scale=PlotScaleParams2d())

        # Create ROI detector plot
        result = plotting_controller._create_roi_detector_plot(
            job_number=job_number,
            source_names=['detector_data'],
            params=params,
        )

        # Layout should have 2 elements
        assert len(result) == 2

        # First element should be detector (DynamicMap overlaid with Rectangles)
        # Second element should be spectrum (DynamicMap)
        # Both should be wrapped in overlays or be DynamicMaps
        assert isinstance(result[0], hv.Overlay | hv.DynamicMap)
        assert isinstance(result[1], hv.DynamicMap)

    def test_create_roi_detector_plot_with_empty_data(
        self,
        plotting_controller,
        job_service,
        data_service,
        workflow_id,
        job_number,
    ):
        """Test ROI detector plot with no data returns placeholder."""
        # Don't add any data - just create empty job data structure
        # We need to ensure the job_number is in job_data
        job_service._job_data[job_number] = {'detector_data': {}}

        # Create plot params
        params = PlotParams2d(plot_scale=PlotScaleParams2d())

        # Create ROI detector plot with no data
        result = plotting_controller._create_roi_detector_plot(
            job_number=job_number,
            source_names=['detector_data'],
            params=params,
        )

        # Should return a Layout with "No data" text
        assert isinstance(result, hv.Layout)
        assert len(result) == 1
        # The element should be a Text element with "No data"
        assert isinstance(result[0], hv.Text)
