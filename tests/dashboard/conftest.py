# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Shared test fixtures for dashboard tests."""

import pydantic
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import (
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.job_orchestrator import JobOrchestrator
from ess.livedata.dashboard.workflow_config_service import WorkflowConfigService
from ess.livedata.fakes import FakeMessageSink


class TestWorkflowParams(pydantic.BaseModel):
    """Simple params model for testing workflows."""

    threshold: float = 100.0


class SimpleTestOutputs(WorkflowOutputsBase):
    """Simple outputs model for testing."""

    result: sc.DataArray = pydantic.Field(title='Result')


class FakeWorkflowConfigService(WorkflowConfigService):
    """Minimal fake for WorkflowConfigService."""

    def subscribe_to_workflow_status(self, source_name, callback):
        pass


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
def workflow_id_2():
    """Create a second test WorkflowId."""
    return WorkflowId(
        instrument='test_instrument',
        namespace='test_namespace',
        name='test_workflow_2',
        version=1,
    )


@pytest.fixture
def workflow_spec(workflow_id):
    """Create a WorkflowSpec with params for testing.

    Having params ensures JobOrchestrator initializes staged configs,
    allowing tests to use public API (stage_config, commit_workflow).
    """
    return WorkflowSpec(
        instrument=workflow_id.instrument,
        namespace=workflow_id.namespace,
        name=workflow_id.name,
        version=workflow_id.version,
        title='Test Workflow',
        description='A test workflow',
        source_names=['source1', 'source2'],
        params=TestWorkflowParams,
        aux_sources=None,
        outputs=SimpleTestOutputs,
    )


@pytest.fixture
def workflow_spec_2(workflow_id_2):
    """Create a second WorkflowSpec with params for testing."""
    return WorkflowSpec(
        instrument=workflow_id_2.instrument,
        namespace=workflow_id_2.namespace,
        name=workflow_id_2.name,
        version=workflow_id_2.version,
        title='Test Workflow 2',
        description='A second test workflow',
        source_names=['source_a', 'source_b'],
        params=TestWorkflowParams,
        aux_sources=None,
        outputs=SimpleTestOutputs,
    )


@pytest.fixture
def workflow_registry(workflow_id, workflow_spec, workflow_id_2, workflow_spec_2):
    """Create a workflow registry with two workflows for testing."""
    return {workflow_id: workflow_spec, workflow_id_2: workflow_spec_2}


@pytest.fixture
def fake_workflow_config_service():
    """Create a fake workflow config service for tests."""
    return FakeWorkflowConfigService()


@pytest.fixture
def fake_message_sink():
    """Create a FakeMessageSink for testing."""
    return FakeMessageSink()


@pytest.fixture
def command_service(fake_message_sink):
    """Create a CommandService with FakeMessageSink."""
    return CommandService(sink=fake_message_sink)


@pytest.fixture
def job_orchestrator(command_service, fake_workflow_config_service, workflow_registry):
    """Create a JobOrchestrator with fakes for testing."""
    return JobOrchestrator(
        command_service=command_service,
        workflow_config_service=fake_workflow_config_service,
        workflow_registry=workflow_registry,
        config_store=None,
    )
