# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import holoviews as hv
import pydantic
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import (
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.stream_manager import StreamManager

hv.extension('bokeh')


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
