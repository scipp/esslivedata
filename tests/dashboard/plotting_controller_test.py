# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import holoviews as hv
import pydantic
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import (
    JobId,
    ResultKey,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_params import (
    PlotParamsROIDetector,
    PlotScaleParams2d,
)
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
def job_service():
    """Create a JobService for testing."""
    return JobService()


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


class TestROIDetectorTwoPhaseCreation:
    """Tests for two-phase ROI plot creation."""

    def test_setup_pipeline_single_detector_invokes_callback_with_dict(
        self,
        plotting_controller,
        data_service,
        workflow_id,
        job_number,
        detector_data,
    ):
        """setup_pipeline for ROI invokes callback with dict of pipes."""
        from ess.livedata.dashboard.data_roles import PRIMARY

        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_1', job_number=job_number),
            output_name='current',
        )
        data_service[detector_key] = detector_data

        params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())
        callback_results = []

        def on_first_data(pipes):
            callback_results.append(pipes)

        plotting_controller.setup_pipeline(
            keys_by_role={PRIMARY: [detector_key]},
            plot_name='roi_detector',
            params=params,
            on_first_data=on_first_data,
        )

        # Callback should have been invoked with a dict
        assert len(callback_results) == 1
        pipes = callback_results[0]
        assert isinstance(pipes, dict)
        assert len(pipes) == 1
        assert detector_key in pipes

    def test_setup_pipeline_multiple_detectors_waits_for_all(
        self,
        plotting_controller,
        data_service,
        workflow_id,
        job_number,
        detector_data,
    ):
        """setup_pipeline for ROI waits for all detectors before callback."""
        from ess.livedata.dashboard.data_roles import PRIMARY

        detector_key_1 = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_1', job_number=job_number),
            output_name='current',
        )
        detector_key_2 = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_2', job_number=job_number),
            output_name='current',
        )

        # Only add data for detector_1 initially
        data_service[detector_key_1] = detector_data

        params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())
        callback_results = []

        def on_first_data(pipes):
            callback_results.append(pipes)

        plotting_controller.setup_pipeline(
            keys_by_role={PRIMARY: [detector_key_1, detector_key_2]},
            plot_name='roi_detector',
            params=params,
            on_first_data=on_first_data,
        )

        # Callback should NOT have been invoked yet (only 1 of 2 detectors ready)
        assert len(callback_results) == 0

        # Add data for detector_2
        data_service[detector_key_2] = detector_data

        # Now callback should be invoked with both pipes
        assert len(callback_results) == 1
        pipes = callback_results[0]
        assert isinstance(pipes, dict)
        assert len(pipes) == 2
        assert detector_key_1 in pipes
        assert detector_key_2 in pipes

    def test_create_plot_from_pipeline_roi_returns_layout(
        self,
        plotting_controller,
        data_service,
        workflow_id,
        job_number,
        detector_data,
        spectrum_data,
    ):
        """create_plot_from_pipeline for ROI returns Layout from dict of pipes."""
        from ess.livedata.dashboard.data_roles import PRIMARY

        detector_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_1', job_number=job_number),
            output_name='current',
        )
        spectrum_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='detector_1', job_number=job_number),
            output_name='roi_current_0',
        )
        data_service[detector_key] = detector_data
        data_service[spectrum_key] = spectrum_data

        params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())
        captured_pipes = []

        def on_first_data(pipes):
            captured_pipes.append(pipes)

        # Phase 1: setup pipeline
        plotting_controller.setup_pipeline(
            keys_by_role={PRIMARY: [detector_key]},
            plot_name='roi_detector',
            params=params,
            on_first_data=on_first_data,
        )

        # Phase 2: create plot from pipeline
        assert len(captured_pipes) == 1
        result = plotting_controller.create_plot_from_pipeline(
            plot_name='roi_detector',
            params=params,
            pipe=captured_pipes[0],
        )

        assert isinstance(result, hv.Layout)
        assert len(result) == 1  # detector only (spectrum created separately)

    def test_create_plot_from_pipeline_roi_multiple_detectors(
        self,
        plotting_controller,
        data_service,
        workflow_id,
        job_number,
        detector_data,
        spectrum_data,
    ):
        """create_plot_from_pipeline for ROI handles multiple detectors."""
        from ess.livedata.dashboard.data_roles import PRIMARY

        # Set up two detectors
        keys = []
        for source_name in ['detector_1', 'detector_2']:
            detector_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name='current',
            )
            spectrum_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name='roi_current_0',
            )
            data_service[detector_key] = detector_data
            data_service[spectrum_key] = spectrum_data
            keys.append(detector_key)

        params = PlotParamsROIDetector(plot_scale=PlotScaleParams2d())
        captured_pipes = []

        def on_first_data(pipes):
            captured_pipes.append(pipes)

        # Phase 1: setup pipeline
        plotting_controller.setup_pipeline(
            keys_by_role={PRIMARY: keys},
            plot_name='roi_detector',
            params=params,
            on_first_data=on_first_data,
        )

        # Phase 2: create plot from pipeline
        assert len(captured_pipes) == 1
        result = plotting_controller.create_plot_from_pipeline(
            plot_name='roi_detector',
            params=params,
            pipe=captured_pipes[0],
        )

        assert isinstance(result, hv.Layout)
        # 2 detectors (spectrum created separately)
        assert len(result) == 2

    def test_create_plot_from_pipeline_roi_validates_params_type(
        self,
        plotting_controller,
        workflow_id,
        job_number,
    ):
        """create_plot_from_pipeline for ROI validates params type."""

        class WrongParams(pydantic.BaseModel):
            value: int = 42

        # Create a fake pipes dict (structure doesn't matter for validation)
        fake_pipes = {
            ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name='detector_1', job_number=job_number),
                output_name='current',
            ): hv.streams.Pipe(data={})
        }

        with pytest.raises(
            TypeError, match="roi_detector requires PlotParamsROIDetector"
        ):
            plotting_controller.create_plot_from_pipeline(
                plot_name='roi_detector',
                params=WrongParams(),
                pipe=fake_pipes,
            )


class TestGetAvailablePlottersFromSpec:
    """Tests for PlottingController.get_available_plotters_from_spec()."""

    def test_returns_compatible_plotters_for_1d_template(self, plotting_controller):
        """Test that 1D output template returns compatible plotters."""

        class TestOutputs(WorkflowOutputsBase):
            i_of_q: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['Q'], shape=[0], unit='counts'),
                    coords={'Q': sc.arange('Q', 0, unit='1/angstrom')},
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
        )

        plotters, has_template = plotting_controller.get_available_plotters_from_spec(
            workflow_spec=spec, output_name='i_of_q'
        )

        assert has_template is True
        assert 'lines' in plotters
        assert 'image' not in plotters

    def test_returns_compatible_plotters_for_2d_template(self, plotting_controller):
        """Test that 2D output template returns compatible plotters."""

        class TestOutputs(WorkflowOutputsBase):
            detector: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x', 'y'], shape=[0, 0], unit='counts'),
                    coords={
                        'x': sc.arange('x', 0, unit='m'),
                        'y': sc.arange('y', 0, unit='m'),
                    },
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
        )

        plotters, has_template = plotting_controller.get_available_plotters_from_spec(
            workflow_spec=spec, output_name='detector'
        )

        assert has_template is True
        assert 'image' in plotters
        # roi_detector requires aux_sources with ROI support, not included here
        assert 'roi_detector' not in plotters
        assert 'lines' not in plotters

    def test_returns_roi_detector_for_2d_template_with_roi_aux_sources(
        self, plotting_controller
    ):
        """Test that roi_detector is included when aux_sources supports ROI."""
        from ess.livedata.handlers.detector_view_specs import DetectorROIAuxSources

        class TestOutputs(WorkflowOutputsBase):
            detector: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x', 'y'], shape=[0, 0], unit='counts'),
                    coords={
                        'x': sc.arange('x', 0, unit='m'),
                        'y': sc.arange('y', 0, unit='m'),
                    },
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
            aux_sources=DetectorROIAuxSources,
        )

        plotters, has_template = plotting_controller.get_available_plotters_from_spec(
            workflow_spec=spec, output_name='detector'
        )

        assert has_template is True
        assert 'image' in plotters
        assert 'roi_detector' in plotters
        assert 'lines' not in plotters

    def test_returns_all_plotters_when_no_template(self, plotting_controller):
        """Test that all plotters are returned as fallback when no template exists."""

        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = pydantic.Field(title='Result')

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
        )

        plotters, has_template = plotting_controller.get_available_plotters_from_spec(
            workflow_spec=spec, output_name='result'
        )

        assert has_template is False
        assert len(plotters) > 0
        assert 'lines' in plotters
        assert 'image' in plotters


class TestSpecRequirements:
    """Tests for SpecRequirements validation logic."""

    def test_empty_requirements_always_validates(self) -> None:
        """Test that empty requires_aux_sources always returns True."""
        from ess.livedata.dashboard.plotting import SpecRequirements

        spec_req = SpecRequirements(requires_aux_sources=[])

        # Should return True regardless of aux_sources_type
        assert spec_req.validate_spec(None) is True
        assert spec_req.validate_spec(object) is True

    def test_returns_false_when_aux_sources_is_none_but_required(self) -> None:
        """Test that validation fails when aux_sources is None but required."""
        from ess.livedata.dashboard.plotting import SpecRequirements
        from ess.livedata.handlers.detector_view_specs import DetectorROIAuxSources

        spec_req = SpecRequirements(requires_aux_sources=[DetectorROIAuxSources])

        assert spec_req.validate_spec(None) is False

    def test_returns_true_when_aux_sources_matches_required(self) -> None:
        """Test that validation passes when aux_sources matches requirement."""
        from ess.livedata.dashboard.plotting import SpecRequirements
        from ess.livedata.handlers.detector_view_specs import DetectorROIAuxSources

        spec_req = SpecRequirements(requires_aux_sources=[DetectorROIAuxSources])

        assert spec_req.validate_spec(DetectorROIAuxSources) is True

    def test_returns_true_for_subclass_of_required(self) -> None:
        """Test that validation passes for subclasses of required type."""
        from ess.livedata.dashboard.plotting import SpecRequirements
        from ess.livedata.handlers.detector_view_specs import DetectorROIAuxSources

        # Create a subclass of DetectorROIAuxSources
        class CustomROIAuxSources(DetectorROIAuxSources):
            pass

        spec_req = SpecRequirements(requires_aux_sources=[DetectorROIAuxSources])

        assert spec_req.validate_spec(CustomROIAuxSources) is True

    def test_returns_false_when_aux_sources_does_not_match(self) -> None:
        """Test that validation fails when aux_sources doesn't match requirement."""
        from ess.livedata.config.workflow_spec import AuxSourcesBase
        from ess.livedata.dashboard.plotting import SpecRequirements
        from ess.livedata.handlers.detector_view_specs import DetectorROIAuxSources

        # Create a different aux_sources type
        class OtherAuxSources(AuxSourcesBase):
            pass

        spec_req = SpecRequirements(requires_aux_sources=[DetectorROIAuxSources])

        assert spec_req.validate_spec(OtherAuxSources) is False

    def test_default_spec_requirements_has_no_requirements(self) -> None:
        """Test that default SpecRequirements has no aux_sources requirements."""
        from ess.livedata.dashboard.plotting import SpecRequirements

        spec_req = SpecRequirements()

        assert spec_req.requires_aux_sources == []
        assert spec_req.validate_spec(None) is True
