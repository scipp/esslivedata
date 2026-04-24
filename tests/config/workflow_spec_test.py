# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from pydantic import Field

from ess.livedata.config.workflow_spec import (
    REDUCTION,
    TIMESERIES,
    AuxInput,
    AuxSources,
    JobId,
    WorkflowConfig,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)


class SimpleTestOutputs(WorkflowOutputsBase):
    """Simple outputs model for testing."""

    result: sc.DataArray = Field(title='Result')


@pytest.fixture
def sample_workflow_id() -> WorkflowId:
    return WorkflowId(
        instrument="INSTRUMENT",
        namespace="NAMESPACE",
        name="NAME",
        version=1,
    )


@pytest.fixture
def sample_workflow_config(sample_workflow_id: WorkflowId) -> WorkflowConfig:
    return WorkflowConfig(
        identifier=sample_workflow_id,
        params={"param1": 10, "param2": "value"},
    )


class TestWorkflowSpecAuxSources:
    """Tests for WorkflowSpec.aux_sources field."""

    def test_workflow_spec_without_aux_sources(self) -> None:
        """Test that WorkflowSpec can be created without aux_sources."""
        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=SimpleTestOutputs,
            group=REDUCTION,
        )
        assert spec.aux_sources is None

    def test_workflow_spec_with_aux_sources_model(self) -> None:
        """Test WorkflowSpec with aux_sources as an AuxSources instance."""
        aux = AuxSources(
            {
                'monitor': AuxInput(
                    choices=('monitor1', 'monitor2'),
                    default='monitor1',
                    title='Monitor',
                    description='Select which monitor to use',
                )
            }
        )

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            aux_sources=aux,
            outputs=SimpleTestOutputs,
            group=REDUCTION,
        )
        assert spec.aux_sources is not None
        assert isinstance(spec.aux_sources, AuxSources)


class TestWorkflowSpecOutputs:
    """Tests for WorkflowSpec.outputs field and get_output_template()."""

    def test_workflow_spec_with_outputs_model(self) -> None:
        """Test WorkflowSpec with outputs as a Pydantic model."""

        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x'], shape=[0], unit='counts'),
                    coords={'x': sc.arange('x', 0, unit='m')},
                ),
                title='Result',
                description='Test output',
            )

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=TestOutputs,
            group=REDUCTION,
        )
        assert spec.outputs is TestOutputs

    def test_get_output_template_returns_dataarray(self) -> None:
        """Test that get_output_template returns a DataArray from default_factory."""

        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x', 'y'], shape=[0, 0], unit='counts'),
                    coords={
                        'x': sc.arange('x', 0, unit='m'),
                        'y': sc.arange('y', 0, unit='m'),
                    },
                ),
                title='Result',
            )

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=TestOutputs,
            group=REDUCTION,
        )

        template = spec.get_output_template('result')

        assert template is not None
        assert isinstance(template, sc.DataArray)
        assert template.dims == ('x', 'y')
        assert set(template.coords.keys()) == {'x', 'y'}
        assert template.unit == 'counts'

    def test_get_output_template_returns_none_for_missing_output(self) -> None:
        """Test that get_output_template returns None for non-existent output."""

        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x'], shape=[0]),
                    coords={'x': sc.arange('x', 0)},
                )
            )

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=TestOutputs,
            group=REDUCTION,
        )

        template = spec.get_output_template('nonexistent')
        assert template is None

    def test_get_output_template_returns_none_without_default_factory(
        self,
    ) -> None:
        """Test that get_output_template returns None without default_factory."""

        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = Field(title='Result')

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=TestOutputs,
            group=REDUCTION,
        )

        template = spec.get_output_template('result')
        assert template is None

    def test_get_output_template_creates_fresh_instances(self) -> None:
        """Test that get_output_template creates a new instance each time."""

        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x'], shape=[0]),
                    coords={'x': sc.arange('x', 0)},
                )
            )

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=TestOutputs,
            group=REDUCTION,
        )

        template1 = spec.get_output_template('result')
        template2 = spec.get_output_template('result')

        # Should be different instances (not the same mutable object)
        assert template1 is not template2


class TestGetOutputTitle:
    """Tests for WorkflowSpec.get_output_title()."""

    def test_returns_title_when_defined(self) -> None:
        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = Field(title='I(d)')

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=TestOutputs,
            group=REDUCTION,
        )
        assert spec.get_output_title('result') == 'I(d)'

    def test_falls_back_to_field_name_when_no_title(self) -> None:
        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = Field(description='No title set')

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=TestOutputs,
            group=REDUCTION,
        )
        assert spec.get_output_title('result') == 'result'

    def test_falls_back_to_output_name_for_missing_field(self) -> None:
        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = Field(title='Result')

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=TestOutputs,
            group=REDUCTION,
        )
        assert spec.get_output_title('nonexistent') == 'nonexistent'


class TestGetOutputDescription:
    """Tests for WorkflowSpec.get_output_description()."""

    def test_returns_description_when_defined(self) -> None:
        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = Field(description='A detailed description.')

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=TestOutputs,
            group=REDUCTION,
        )
        assert spec.get_output_description('result') == 'A detailed description.'

    def test_returns_none_when_no_description(self) -> None:
        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = Field(title='Result')

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=TestOutputs,
            group=REDUCTION,
        )
        assert spec.get_output_description('result') is None

    def test_returns_none_for_missing_field(self) -> None:
        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = Field(description='Exists')

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            outputs=TestOutputs,
            group=REDUCTION,
        )
        assert spec.get_output_description('nonexistent') is None


class TestWorkflowConfigAuxSourceNames:
    """Tests for WorkflowConfig.aux_source_names field."""

    def test_workflow_config_default_aux_source_names(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test that WorkflowConfig has empty aux_source_names by default."""
        config = WorkflowConfig(identifier=sample_workflow_id)
        assert config.aux_source_names == {}

    def test_workflow_config_with_aux_source_names(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test that WorkflowConfig can store aux_source_names as dict."""
        config = WorkflowConfig(
            identifier=sample_workflow_id,
            aux_source_names={"monitor": "monitor1", "rotation": "rotation_a"},
        )
        assert config.aux_source_names == {
            "monitor": "monitor1",
            "rotation": "rotation_a",
        }

    def test_workflow_config_serialization_with_aux_source_names(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test that WorkflowConfig with aux_source_names serializes correctly."""
        config = WorkflowConfig(
            identifier=sample_workflow_id,
            aux_source_names={"monitor": "monitor2"},
            params={"param1": 10},
        )

        dumped = config.model_dump()
        loaded = WorkflowConfig.model_validate(dumped)

        assert loaded.aux_source_names == {"monitor": "monitor2"}
        assert loaded.params == {"param1": 10}


class TestWorkflowConfigFromParams:
    """Tests for WorkflowConfig.from_params() helper method."""

    def test_from_params_with_all_arguments(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test from_params with params and aux_source_names."""
        params = {"param1": 20, "param2": "custom"}
        aux_sources = {"monitor": "monitor1"}

        config = WorkflowConfig.from_params(
            workflow_id=sample_workflow_id,
            params=params,
            aux_source_names=aux_sources,
        )

        assert config.identifier == sample_workflow_id
        assert config.params == {"param1": 20, "param2": "custom"}
        assert config.aux_source_names == {"monitor": "monitor1"}
        assert config.job_number is not None  # Should be auto-generated

    def test_from_params_with_none_params(self, sample_workflow_id: WorkflowId) -> None:
        """Test from_params with no params (None)."""
        config = WorkflowConfig.from_params(
            workflow_id=sample_workflow_id,
            params=None,
            aux_source_names=None,
        )

        assert config.identifier == sample_workflow_id
        assert config.params == {}
        assert config.aux_source_names == {}
        assert config.job_number is not None

    def test_from_params_with_custom_job_number(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test from_params with explicit job_number."""
        import uuid

        custom_job_number = uuid.uuid4()

        config = WorkflowConfig.from_params(
            workflow_id=sample_workflow_id,
            params=None,
            job_number=custom_job_number,
        )

        assert config.job_number == custom_job_number

    def test_from_params_with_complex_params(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test from_params with complex nested dict params."""
        params = {"nested_value": 10, "string_list": ["x", "y", "z"]}

        config = WorkflowConfig.from_params(
            workflow_id=sample_workflow_id,
            params=params,
        )

        assert isinstance(config.params, dict)
        assert config.params == {"nested_value": 10, "string_list": ["x", "y", "z"]}


class TestAuxSourcesConstruction:
    """Tests for AuxSources construction and get_defaults()."""

    def test_string_shorthand_creates_single_fixed_choice(self) -> None:
        aux = AuxSources({'rotation': 'det_rotation'})
        inp = aux.inputs['rotation']
        assert inp.choices == ('det_rotation',)
        assert inp.default == 'det_rotation'

    def test_string_shorthand_leaves_title_empty(self) -> None:
        aux = AuxSources({'rotation': 'det_rotation'})
        assert aux.inputs['rotation'].title == ''

    def test_aux_input_preserves_metadata(self) -> None:
        aux = AuxSources(
            {
                'monitor': AuxInput(
                    choices=('mon1', 'mon2'),
                    default='mon1',
                    title='Monitor',
                    description='Select monitor',
                ),
            }
        )
        inp = aux.inputs['monitor']
        assert inp.choices == ('mon1', 'mon2')
        assert inp.default == 'mon1'
        assert inp.title == 'Monitor'
        assert inp.description == 'Select monitor'

    def test_get_defaults_single_input(self) -> None:
        aux = AuxSources({'monitor': 'mon1'})
        assert aux.get_defaults() == {'monitor': 'mon1'}

    def test_get_defaults_multiple_inputs(self) -> None:
        aux = AuxSources(
            {
                'incident': AuxInput(choices=('mon1', 'mon2'), default='mon1'),
                'transmission': AuxInput(choices=('mon3', 'mon4'), default='mon4'),
            }
        )
        assert aux.get_defaults() == {'incident': 'mon1', 'transmission': 'mon4'}

    def test_inputs_returns_copy(self) -> None:
        aux = AuxSources({'a': 'x'})
        inputs1 = aux.inputs
        inputs2 = aux.inputs
        assert inputs1 is not inputs2
        assert inputs1 == inputs2


class TestAuxSourcesRender:
    """Tests for AuxSources render() method."""

    def test_default_render_returns_fixed_values(self) -> None:
        aux_sources = AuxSources({'monitor': 'monitor1'})
        job_id = JobId(source_name='detector1', job_number='test-uuid-123')

        rendered = aux_sources.render(job_id)

        assert rendered == {'monitor': 'monitor1'}

    def test_render_with_selections_overrides_defaults(self) -> None:
        aux_sources = AuxSources(
            {
                'monitor': AuxInput(choices=('mon1', 'mon2'), default='mon1'),
            }
        )
        job_id = JobId(source_name='det1', job_number='id-1')

        rendered = aux_sources.render(job_id, selections={'monitor': 'mon2'})

        assert rendered == {'monitor': 'mon2'}

    def test_render_with_partial_selections(self) -> None:
        aux_sources = AuxSources(
            {
                'a': AuxInput(choices=('x', 'y'), default='x'),
                'b': AuxInput(choices=('p', 'q'), default='p'),
            }
        )
        job_id = JobId(source_name='det1', job_number='id-1')

        rendered = aux_sources.render(job_id, selections={'a': 'y'})

        assert rendered == {'a': 'y', 'b': 'p'}

    def test_render_ignores_unknown_selection_keys(self) -> None:
        aux_sources = AuxSources({'monitor': 'mon1'})
        job_id = JobId(source_name='det1', job_number='id-1')

        rendered = aux_sources.render(
            job_id, selections={'monitor': 'mon1', 'unknown': 'ignored'}
        )

        assert rendered == {'monitor': 'mon1'}

    def test_render_with_none_selections_uses_defaults(self) -> None:
        aux_sources = AuxSources({'monitor': 'mon1'})
        job_id = JobId(source_name='det1', job_number='id-1')

        rendered = aux_sources.render(job_id, selections=None)

        assert rendered == {'monitor': 'mon1'}

    def test_custom_render_transforms_stream_names(self) -> None:
        class RoiAuxSources(AuxSources):
            def render(
                self, job_id: JobId, selections: dict[str, str] | None = None
            ) -> dict[str, str]:
                base = super().render(job_id, selections)
                return {
                    field: f"{job_id.job_number}/{stream}"
                    for field, stream in base.items()
                }

        aux_sources = RoiAuxSources(
            {
                'roi': AuxInput(
                    choices=('roi_rectangle', 'roi_polygon'), default='roi_rectangle'
                )
            }
        )
        job_id = JobId(source_name='detector1', job_number='abc-123')

        rendered = aux_sources.render(job_id, selections={'roi': 'roi_polygon'})

        assert rendered == {'roi': 'abc-123/roi_polygon'}

    def test_custom_render_with_source_name_prefix(self) -> None:
        class SourcePrefixedAuxSources(AuxSources):
            def render(
                self, job_id: JobId, selections: dict[str, str] | None = None
            ) -> dict[str, str]:
                base = super().render(job_id, selections)
                return {
                    field: f"{job_id.source_name}/{stream}"
                    for field, stream in base.items()
                }

        aux_sources = SourcePrefixedAuxSources(
            {'monitor': AuxInput(choices=('monitor1', 'monitor2'), default='monitor1')}
        )
        job_id = JobId(source_name='detector1', job_number='uuid-456')

        rendered = aux_sources.render(job_id, selections={'monitor': 'monitor2'})

        assert rendered == {'monitor': 'detector1/monitor2'}

    def test_render_with_multiple_fields(self) -> None:
        aux_sources = AuxSources(
            {
                'incident_monitor': 'monitor1',
                'transmission_monitor': 'monitor2',
            }
        )
        job_id = JobId(source_name='detector1', job_number='test-id')

        rendered = aux_sources.render(job_id)

        assert rendered == {
            'incident_monitor': 'monitor1',
            'transmission_monitor': 'monitor2',
        }


class TestJobId:
    """Test cases for JobId."""

    def test_str_format(self) -> None:
        """Test that __str__ returns source_name/job_number format."""
        import uuid

        job_number = uuid.uuid4()
        job_id = JobId(source_name='detector', job_number=job_number)

        result = str(job_id)

        assert result == f'detector/{job_number}'

    def test_str_with_different_source_names(self) -> None:
        """Test __str__ with various source_name values."""
        import uuid

        test_cases = ['detector', 'monitor', 'mantle', 'source_1']

        for source_name in test_cases:
            job_number = uuid.uuid4()
            job_id = JobId(source_name=source_name, job_number=job_number)

            result = str(job_id)

            assert result == f'{source_name}/{job_number}'
            assert '/' in result
            assert result.startswith(source_name + '/')

    def test_str_used_in_stream_names(self) -> None:
        """Test that __str__ is suitable for use in stream names."""
        import uuid

        job_number = uuid.uuid4()
        job_id = JobId(source_name='detector', job_number=job_number)

        # Simulate stream name construction
        stream_name = f'{job_id}/roi_rectangle'

        expected = f'detector/{job_number}/roi_rectangle'
        assert stream_name == expected

    def test_str_ensures_uniqueness_across_detectors(self) -> None:
        """Test that __str__ provides unique identifiers for different detectors."""
        import uuid

        job_number = uuid.uuid4()  # Same job number
        job_id_1 = JobId(source_name='detector_1', job_number=job_number)
        job_id_2 = JobId(source_name='detector_2', job_number=job_number)

        # Should be different due to different source names
        assert str(job_id_1) != str(job_id_2)
        assert str(job_id_1) == f'detector_1/{job_number}'
        assert str(job_id_2) == f'detector_2/{job_number}'


class TestFindTimeseriesOutputs:
    """Tests for find_timeseries_outputs() helper function."""

    def test_finds_timeseries_output_with_time_coord(self) -> None:
        """Test that outputs with 0-D data and time coord are found."""
        from ess.livedata.config.workflow_spec import find_timeseries_outputs

        class TimeseriesOutputs(WorkflowOutputsBase):
            delta: sc.DataArray = Field(
                default_factory=lambda: sc.DataArray(
                    sc.scalar(0.0),
                    coords={'time': sc.scalar(0, unit='ns')},
                ),
            )

        workflow_id = WorkflowId(
            instrument='test', namespace='timeseries', name='test', version=1
        )
        spec = WorkflowSpec(
            instrument='test',
            namespace='timeseries',
            name='test',
            version=1,
            title='Test',
            description='Test',
            params=None,
            outputs=TimeseriesOutputs,
            source_names=['source1', 'source2'],
            group=TIMESERIES,
        )

        results = find_timeseries_outputs({workflow_id: spec})

        # Should find 2 entries (one per source_name)
        assert len(results) == 2
        assert (workflow_id, 'source1', 'delta') in results
        assert (workflow_id, 'source2', 'delta') in results

    def test_ignores_multidimensional_outputs(self) -> None:
        """Test that outputs with ndim > 0 are not identified as timeseries."""
        from ess.livedata.config.workflow_spec import find_timeseries_outputs

        class NonTimeseriesOutputs(WorkflowOutputsBase):
            histogram: sc.DataArray = Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x'], shape=[10]),
                    coords={'time': sc.scalar(0, unit='ns')},
                ),
            )

        workflow_id = WorkflowId(
            instrument='test', namespace='other', name='test', version=1
        )
        spec = WorkflowSpec(
            instrument='test',
            namespace='other',
            name='test',
            version=1,
            title='Test',
            description='Test',
            params=None,
            outputs=NonTimeseriesOutputs,
            source_names=['source1'],
            group=REDUCTION,
        )

        results = find_timeseries_outputs({workflow_id: spec})

        assert len(results) == 0

    def test_ignores_outputs_without_time_coord(self) -> None:
        """Test that 0-D outputs without time coord are not identified as timeseries."""
        from ess.livedata.config.workflow_spec import find_timeseries_outputs

        class NoTimeCoordOutputs(WorkflowOutputsBase):
            value: sc.DataArray = Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0.0)),
            )

        workflow_id = WorkflowId(
            instrument='test', namespace='other', name='test', version=1
        )
        spec = WorkflowSpec(
            instrument='test',
            namespace='other',
            name='test',
            version=1,
            title='Test',
            description='Test',
            params=None,
            outputs=NoTimeCoordOutputs,
            source_names=['source1'],
            group=REDUCTION,
        )

        results = find_timeseries_outputs({workflow_id: spec})

        assert len(results) == 0

    def test_ignores_outputs_without_default_factory(self) -> None:
        """Test that outputs without default_factory are skipped."""
        from ess.livedata.config.workflow_spec import find_timeseries_outputs

        class NoFactoryOutputs(WorkflowOutputsBase):
            delta: sc.DataArray = Field(title='Delta')

        workflow_id = WorkflowId(
            instrument='test', namespace='other', name='test', version=1
        )
        spec = WorkflowSpec(
            instrument='test',
            namespace='other',
            name='test',
            version=1,
            title='Test',
            description='Test',
            params=None,
            outputs=NoFactoryOutputs,
            source_names=['source1'],
            group=REDUCTION,
        )

        results = find_timeseries_outputs({workflow_id: spec})

        assert len(results) == 0

    def test_empty_registry_returns_empty_list(self) -> None:
        """Test that empty registry returns empty list."""
        from ess.livedata.config.workflow_spec import find_timeseries_outputs

        results = find_timeseries_outputs({})

        assert results == []

    def test_multiple_workflows_combined(self) -> None:
        """Test that timeseries from multiple workflows are combined."""
        from ess.livedata.config.workflow_spec import find_timeseries_outputs

        class TimeseriesOutputs(WorkflowOutputsBase):
            delta: sc.DataArray = Field(
                default_factory=lambda: sc.DataArray(
                    sc.scalar(0.0),
                    coords={'time': sc.scalar(0, unit='ns')},
                ),
            )

        class NonTimeseriesOutputs(WorkflowOutputsBase):
            histogram: sc.DataArray = Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x'], shape=[10]),
                ),
            )

        workflow_id_1 = WorkflowId(
            instrument='test', namespace='timeseries', name='ts1', version=1
        )
        workflow_id_2 = WorkflowId(
            instrument='test', namespace='detector', name='det1', version=1
        )

        spec1 = WorkflowSpec(
            instrument='test',
            namespace='timeseries',
            name='ts1',
            version=1,
            title='TS 1',
            description='Test',
            params=None,
            outputs=TimeseriesOutputs,
            source_names=['src1'],
            group=TIMESERIES,
        )
        spec2 = WorkflowSpec(
            instrument='test',
            namespace='detector',
            name='det1',
            version=1,
            title='Det 1',
            description='Test',
            params=None,
            outputs=NonTimeseriesOutputs,
            source_names=['src2'],
            group=REDUCTION,
        )

        results = find_timeseries_outputs({workflow_id_1: spec1, workflow_id_2: spec2})

        # Only workflow_id_1 has timeseries outputs
        assert len(results) == 1
        assert (workflow_id_1, 'src1', 'delta') in results
