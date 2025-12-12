# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for WorkflowTemplate protocol and TemplateInstance."""

import uuid

import pytest

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.config.workflow_template import TemplateInstance, WorkflowTemplate
from ess.livedata.dashboard.correlation_histogram import (
    CorrelationHistogram1dTemplate,
    CorrelationHistogram2dTemplate,
)


class FakeCorrelationHistogramController:
    """Fake controller for testing templates."""

    def __init__(self, timeseries_keys: list[ResultKey]) -> None:
        self._timeseries_keys = timeseries_keys

    def get_timeseries(self) -> list[ResultKey]:
        return self._timeseries_keys


def make_result_key(source_name: str, output_name: str = 'value') -> ResultKey:
    """Helper to create ResultKey for testing."""
    return ResultKey(
        workflow_id=WorkflowId(
            instrument='test', namespace='timeseries', name='test', version=1
        ),
        job_id=JobId(source_name=source_name, job_number=uuid.uuid4()),
        output_name=output_name,
    )


class TestTemplateInstance:
    def test_template_instance_serialization(self):
        instance = TemplateInstance(
            template_name='correlation_histogram_1d',
            config={'x_axis': 'temperature'},
        )
        serialized = instance.model_dump(mode='json')

        assert serialized['template_name'] == 'correlation_histogram_1d'
        assert serialized['config'] == {'x_axis': 'temperature'}

    def test_template_instance_roundtrip(self):
        original = TemplateInstance(
            template_name='test_template',
            config={'key': 'value', 'nested': {'a': 1}},
        )
        serialized = original.model_dump(mode='json')
        restored = TemplateInstance.model_validate(serialized)

        assert restored == original


class TestCorrelationHistogram1dTemplate:
    @pytest.fixture
    def timeseries_keys(self):
        return [
            make_result_key('temperature'),
            make_result_key('pressure'),
            make_result_key('flow_rate'),
        ]

    @pytest.fixture
    def template(self, timeseries_keys):
        controller = FakeCorrelationHistogramController(timeseries_keys)
        return CorrelationHistogram1dTemplate(controller=controller)

    def test_implements_workflow_template_protocol(self, template):
        assert isinstance(template, WorkflowTemplate)

    def test_name_property(self, template):
        assert template.name == 'correlation_histogram_1d'

    def test_title_property(self, template):
        assert template.title == '1D Correlation Histogram'

    def test_ndim_property(self, template):
        assert template.ndim == 1

    def test_get_configuration_model_returns_model_with_axis_options(self, template):
        config_model = template.get_configuration_model()

        assert config_model is not None
        assert 'x_axis' in config_model.model_fields

    def test_get_configuration_model_returns_none_when_no_timeseries(self):
        controller = FakeCorrelationHistogramController([])
        template = CorrelationHistogram1dTemplate(controller=controller)
        assert template.get_configuration_model() is None

    def test_make_instance_id(self, template):
        config_model = template.get_configuration_model()
        # Use the formatted name from the template's source mapping
        source_names = list(template.get_source_name_to_key().keys())
        config = config_model(x_axis=source_names[0])  # e.g., 'temperature: value'

        workflow_id = template.make_instance_id(config)

        assert workflow_id.instrument == 'frontend'
        assert workflow_id.namespace == 'correlation'
        assert 'histogram_1d' in workflow_id.name
        assert workflow_id.version == 1

    def test_make_instance_title(self, template):
        config_model = template.get_configuration_model()
        source_names = list(template.get_source_name_to_key().keys())
        config = config_model(x_axis=source_names[0])

        title = template.make_instance_title(config)

        assert 'Correlation Histogram' in title

    def test_create_workflow_spec(self, template, timeseries_keys):
        config_model = template.get_configuration_model()
        source_names = list(template.get_source_name_to_key().keys())
        config = config_model(x_axis=source_names[0])

        spec = template.create_workflow_spec(config)

        assert spec.instrument == 'frontend'
        assert spec.namespace == 'correlation'
        assert spec.aux_sources is None  # Axis is baked into identity
        # Source names are empty - determined dynamically at job start time
        assert spec.source_names == []

    def test_get_axis_refs(self, template, timeseries_keys):
        config_model = template.get_configuration_model()
        source_names = list(template.get_source_name_to_key().keys())
        config = config_model(x_axis=source_names[0])

        axis_refs = template.get_axis_refs(config)

        assert len(axis_refs) == 1
        # The axis ref should match one of the original timeseries_keys
        ref = axis_refs[0]
        matching_keys = [
            k
            for k in timeseries_keys
            if k.workflow_id == ref.workflow_id
            and k.job_id.source_name == ref.source_name
            and (k.output_name or '') == ref.output_name
        ]
        assert len(matching_keys) == 1


class TestCorrelationHistogram2dTemplate:
    @pytest.fixture
    def timeseries_keys(self):
        return [
            make_result_key('temperature'),
            make_result_key('pressure'),
            make_result_key('flow_rate'),
        ]

    @pytest.fixture
    def template(self, timeseries_keys):
        controller = FakeCorrelationHistogramController(timeseries_keys)
        return CorrelationHistogram2dTemplate(controller=controller)

    def test_implements_workflow_template_protocol(self, template):
        assert isinstance(template, WorkflowTemplate)

    def test_name_property(self, template):
        assert template.name == 'correlation_histogram_2d'

    def test_title_property(self, template):
        assert template.title == '2D Correlation Histogram'

    def test_ndim_property(self, template):
        assert template.ndim == 2

    def test_get_configuration_model_returns_model_with_both_axes(self, template):
        config_model = template.get_configuration_model()

        assert config_model is not None
        assert 'x_axis' in config_model.model_fields
        assert 'y_axis' in config_model.model_fields

    def test_get_configuration_model_returns_none_when_insufficient_timeseries(self):
        # Need at least 2 timeseries for 2D
        controller = FakeCorrelationHistogramController(
            [make_result_key('temperature')]
        )
        template = CorrelationHistogram2dTemplate(controller=controller)
        assert template.get_configuration_model() is None

    def test_make_instance_id_includes_both_axes(self, template):
        config_model = template.get_configuration_model()
        source_names = list(template.get_source_name_to_key().keys())
        config = config_model(x_axis=source_names[0], y_axis=source_names[1])

        workflow_id = template.make_instance_id(config)

        assert 'histogram_2d' in workflow_id.name

    def test_make_instance_title_includes_both_axes(self, template):
        config_model = template.get_configuration_model()
        source_names = list(template.get_source_name_to_key().keys())
        config = config_model(x_axis=source_names[0], y_axis=source_names[1])

        title = template.make_instance_title(config)

        assert 'vs' in title
        assert 'Correlation Histogram' in title

    def test_create_workflow_spec(self, template, timeseries_keys):
        config_model = template.get_configuration_model()
        source_names = list(template.get_source_name_to_key().keys())
        config = config_model(x_axis=source_names[0], y_axis=source_names[1])

        spec = template.create_workflow_spec(config)

        assert spec.instrument == 'frontend'
        assert spec.namespace == 'correlation'
        assert spec.aux_sources is None  # Axes are baked into identity
        # Source names are empty - determined dynamically at job start time
        assert spec.source_names == []

    def test_get_axis_refs_returns_both_axes(self, template, timeseries_keys):
        config_model = template.get_configuration_model()
        source_names = list(template.get_source_name_to_key().keys())
        config = config_model(x_axis=source_names[0], y_axis=source_names[1])

        axis_refs = template.get_axis_refs(config)

        assert len(axis_refs) == 2
        # Both axis refs should match original timeseries_keys
        for ref in axis_refs:
            matching_keys = [
                k
                for k in timeseries_keys
                if k.workflow_id == ref.workflow_id
                and k.job_id.source_name == ref.source_name
                and (k.output_name or '') == ref.output_name
            ]
            assert len(matching_keys) == 1
