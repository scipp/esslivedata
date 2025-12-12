# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for WorkflowRegistryManager."""

import logging
import uuid

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
from ess.livedata.dashboard.config_store import InMemoryConfigStore
from ess.livedata.dashboard.correlation_histogram import (
    CorrelationHistogram1dTemplate,
)
from ess.livedata.dashboard.job_orchestrator import WorkflowRegistryManager


class SimpleOutputs(WorkflowOutputsBase):
    result: sc.DataArray = pydantic.Field(title='Result')


def make_workflow_spec(name: str) -> WorkflowSpec:
    """Helper to create a simple WorkflowSpec for testing."""
    return WorkflowSpec(
        instrument='test',
        namespace='testing',
        name=name,
        version=1,
        title=f'Test {name}',
        description=f'Test workflow {name}',
        source_names=['source_1'],
        params=None,
        outputs=SimpleOutputs,
    )


def make_result_key(source_name: str) -> ResultKey:
    """Helper to create ResultKey for testing."""
    return ResultKey(
        workflow_id=WorkflowId(
            instrument='test', namespace='timeseries', name='test', version=1
        ),
        job_id=JobId(source_name=source_name, job_number=uuid.uuid4()),
        output_name='value',
    )


@pytest.fixture
def static_registry():
    spec1 = make_workflow_spec('workflow_1')
    spec2 = make_workflow_spec('workflow_2')
    return {spec1.get_id(): spec1, spec2.get_id(): spec2}


class FakeCorrelationHistogramController:
    """Fake controller for testing templates."""

    def __init__(self, timeseries_keys: list[ResultKey]) -> None:
        self._timeseries_keys = timeseries_keys

    def get_timeseries(self) -> list[ResultKey]:
        return self._timeseries_keys


@pytest.fixture
def timeseries_keys():
    return [
        make_result_key('temperature'),
        make_result_key('pressure'),
    ]


@pytest.fixture
def template(timeseries_keys):
    controller = FakeCorrelationHistogramController(timeseries_keys)
    return CorrelationHistogram1dTemplate(controller=controller)


@pytest.fixture
def logger():
    return logging.getLogger('test')


class TestWorkflowRegistryManagerBasics:
    def test_get_registry_returns_static_workflows(self, static_registry, logger):
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[],
            logger=logger,
        )

        registry = manager.get_registry()

        assert len(registry) == 2
        for wid in static_registry:
            assert wid in registry

    def test_get_templates_returns_empty_when_no_templates(
        self, static_registry, logger
    ):
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[],
            logger=logger,
        )

        assert manager.get_templates() == {}

    def test_get_templates_returns_templates_by_name(
        self, static_registry, template, logger
    ):
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[template],
            logger=logger,
        )

        templates = manager.get_templates()
        assert template.name in templates
        assert templates[template.name] is template

    def test_is_template_instance_false_for_static_workflows(
        self, static_registry, logger
    ):
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[],
            logger=logger,
        )

        for wid in static_registry:
            assert not manager.is_template_instance(wid)


class TestWorkflowRegistryManagerTemplateRegistration:
    def test_register_from_template_creates_workflow(
        self, static_registry, template, logger
    ):
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[template],
            logger=logger,
        )
        # Get the formatted source name from the template
        source_names = list(template.get_source_name_to_key().keys())

        workflow_id = manager.register_from_template(
            template.name, {'x_param': source_names[0]}
        )

        assert workflow_id is not None
        assert workflow_id in manager.get_registry()
        assert manager.is_template_instance(workflow_id)

    def test_register_from_template_returns_none_for_unknown_template(
        self, static_registry, logger
    ):
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[],
            logger=logger,
        )

        workflow_id = manager.register_from_template(
            'nonexistent_template', {'x_param': 'temperature'}
        )

        assert workflow_id is None

    def test_register_from_template_returns_none_for_missing_fields(
        self, static_registry, template, logger
    ):
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[template],
            logger=logger,
        )

        # Invalid config - missing required 'x_param' field
        workflow_id = manager.register_from_template(template.name, {})

        assert workflow_id is None

    def test_register_from_template_accepts_any_axis_name(
        self, static_registry, template, logger
    ):
        """Registration should work with any axis name, even if not in current data."""
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[template],
            logger=logger,
        )

        # Should succeed - enum validation is for UI, not registration
        workflow_id = manager.register_from_template(
            template.name, {'x_param': 'future_timeseries'}
        )

        assert workflow_id is not None
        assert workflow_id in manager.get_registry()

    def test_register_same_config_twice_returns_existing_id(
        self, static_registry, template, logger
    ):
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[template],
            logger=logger,
        )
        source_names = list(template.get_source_name_to_key().keys())

        workflow_id1 = manager.register_from_template(
            template.name, {'x_param': source_names[0]}
        )
        workflow_id2 = manager.register_from_template(
            template.name, {'x_param': source_names[0]}
        )

        assert workflow_id1 == workflow_id2
        # Should only have one dynamic workflow
        registry = manager.get_registry()
        assert len(registry) == len(static_registry) + 1


class TestWorkflowRegistryManagerUnregistration:
    def test_unregister_removes_template_instance(
        self, static_registry, template, logger
    ):
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[template],
            logger=logger,
        )
        source_names = list(template.get_source_name_to_key().keys())
        workflow_id = manager.register_from_template(
            template.name, {'x_param': source_names[0]}
        )

        result = manager.unregister(workflow_id)

        assert result is True
        assert workflow_id not in manager.get_registry()
        assert not manager.is_template_instance(workflow_id)

    def test_unregister_returns_false_for_static_workflow(
        self, static_registry, logger
    ):
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[],
            logger=logger,
        )
        static_wid = next(iter(static_registry.keys()))

        result = manager.unregister(static_wid)

        assert result is False
        # Static workflow should still be present
        assert static_wid in manager.get_registry()

    def test_unregister_returns_false_for_unknown_workflow(
        self, static_registry, logger
    ):
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=None,
            templates=[],
            logger=logger,
        )
        unknown_wid = WorkflowId(
            instrument='unknown', namespace='unknown', name='unknown', version=1
        )

        result = manager.unregister(unknown_wid)

        assert result is False


class TestWorkflowRegistryManagerPersistence:
    def test_persists_template_instances_on_register(
        self, static_registry, template, logger
    ):
        config_store = InMemoryConfigStore()
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=config_store,
            templates=[template],
            logger=logger,
        )
        source_names = list(template.get_source_name_to_key().keys())

        manager.register_from_template(template.name, {'x_param': source_names[0]})

        # Check that template instances were persisted
        instances_data = config_store.get('_template_instances')
        assert instances_data is not None
        assert len(instances_data) == 1

    def test_removes_from_persistence_on_unregister(
        self, static_registry, template, logger
    ):
        config_store = InMemoryConfigStore()
        manager = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=config_store,
            templates=[template],
            logger=logger,
        )
        source_names = list(template.get_source_name_to_key().keys())
        workflow_id = manager.register_from_template(
            template.name, {'x_param': source_names[0]}
        )

        manager.unregister(workflow_id)

        instances_data = config_store.get('_template_instances')
        assert instances_data == {}

    def test_restores_template_instances_on_init(
        self, static_registry, template, logger
    ):
        config_store = InMemoryConfigStore()

        # First manager registers a workflow
        manager1 = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=config_store,
            templates=[template],
            logger=logger,
        )
        source_names = list(template.get_source_name_to_key().keys())
        workflow_id = manager1.register_from_template(
            template.name, {'x_param': source_names[0]}
        )

        # Second manager should restore the workflow on init
        manager2 = WorkflowRegistryManager(
            static_registry=static_registry,
            config_store=config_store,
            templates=[template],
            logger=logger,
        )

        assert workflow_id in manager2.get_registry()
        assert manager2.is_template_instance(workflow_id)
