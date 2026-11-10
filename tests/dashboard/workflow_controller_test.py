# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NamedTuple

import pydantic
import pytest

from ess.livedata.config.workflow_spec import (
    REDUCTION,
    WorkflowId,
    WorkflowSpec,
)
from ess.livedata.core.desired_state import (
    DesiredTable,
    InMemoryDesiredStateLog,
    fold_updates,
)
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.workflow_controller import WorkflowController
from ess.livedata.fakes import FakeMessageSink


class WorkflowControllerFixture(NamedTuple):
    """Container for workflow controller fixture components."""

    controller: WorkflowController
    desired_state: InMemoryDesiredStateLog
    config_store: dict[WorkflowId, dict]


class SomeWorkflowParams(pydantic.BaseModel):
    """Test Pydantic model for workflow parameters."""

    threshold: float = 100.0
    mode: str = "fast"


def desired_table(log: InMemoryDesiredStateLog) -> DesiredTable:
    """The desired table the log holds."""
    return fold_updates({}, log.read(0))


@pytest.fixture
def source_names() -> list[str]:
    """Test source names."""
    return ["detector_1", "detector_2"]


@pytest.fixture
def workflow_id() -> WorkflowId:
    """Test workflow ID."""
    return WorkflowId(instrument='test_instrument', name="test_workflow", version=1)


@pytest.fixture
def workflow_spec(workflow_id: WorkflowId) -> WorkflowSpec:
    """Test workflow specification."""
    return WorkflowSpec(
        instrument=workflow_id.instrument,
        name=workflow_id.name,
        version=workflow_id.version,
        title="Test Workflow",
        description="A test workflow for unit testing",
        source_names=["detector_1", "detector_2"],
        params=SomeWorkflowParams,
        group=REDUCTION,
    )


@pytest.fixture
def workflow_registry(
    workflow_id: WorkflowId, workflow_spec: WorkflowSpec
) -> dict[WorkflowId, WorkflowSpec]:
    """Test workflow registry."""
    return {workflow_id: workflow_spec}


@pytest.fixture
def fake_message_sink() -> FakeMessageSink:
    """Fake message sink for testing."""
    return FakeMessageSink()


@pytest.fixture
def command_service(fake_message_sink: FakeMessageSink) -> CommandService:
    """Create a command service with fake sink."""
    return CommandService(sink=fake_message_sink)


@pytest.fixture
def fake_config_store() -> dict[str, dict]:
    """Plain dict for config store testing."""
    return {}


@pytest.fixture
def workflow_controller(
    command_service: CommandService,
    fake_config_store: dict[str, dict],
    source_names: list[str],
    workflow_registry: dict[WorkflowId, WorkflowSpec],
) -> WorkflowControllerFixture:
    """Workflow controller instance for testing."""
    # Create JobOrchestrator with shared services
    from ess.livedata.dashboard.active_job_registry import ActiveJobRegistry
    from ess.livedata.dashboard.data_service import DataService
    from ess.livedata.dashboard.job_orchestrator import JobOrchestrator
    from ess.livedata.dashboard.job_service import JobService

    ds = DataService()
    js = JobService()
    desired_state = InMemoryDesiredStateLog()
    job_orchestrator = JobOrchestrator(
        desired_state=desired_state,
        command_service=command_service,
        workflow_registry=workflow_registry,
        active_job_registry=ActiveJobRegistry(data_service=ds, job_service=js),
        config_store=fake_config_store,
    )

    controller = WorkflowController(
        job_orchestrator=job_orchestrator,
        workflow_registry=workflow_registry,
    )
    return WorkflowControllerFixture(
        controller=controller,
        desired_state=desired_state,
        config_store=fake_config_store,
    )


class TestWorkflowController:
    def test_start_workflow_sends_config_to_sources(
        self,
        workflow_controller: WorkflowControllerFixture,
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        """Test that start_workflow sends configuration to all specified sources."""
        config = SomeWorkflowParams(threshold=150.0, mode="accurate")

        # Act
        workflow_controller.controller.start_workflow(workflow_id, source_names, config)

        # Assert
        desired = desired_table(workflow_controller.desired_state)[workflow_id]
        assert list(desired.jobs) == source_names
        for job in desired.jobs.values():
            assert job.params == {"threshold": 150.0, "mode": "accurate"}

    def test_start_workflow_saves_persistent_config(
        self,
        workflow_controller: WorkflowControllerFixture,
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        """Test that start_workflow saves persistent configuration."""
        config = SomeWorkflowParams(threshold=200.0, mode="fast")

        # Act
        workflow_controller.controller.start_workflow(workflow_id, source_names, config)

        # Assert - check ConfigStore instead of service
        persistent_config_data = workflow_controller.config_store.get(str(workflow_id))
        assert persistent_config_data is not None
        assert set(persistent_config_data['jobs'].keys()) == set(source_names)
        # With per-source config, each source has its own params
        for source in source_names:
            assert persistent_config_data['jobs'][source]['params'] == {
                "threshold": 200.0,
                "mode": "fast",
            }

    def test_start_workflow_with_empty_config(
        self,
        workflow_controller: WorkflowControllerFixture,
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        """Test that start_workflow works with empty configuration."""
        config = SomeWorkflowParams()  # Use defaults

        # Act
        workflow_controller.controller.start_workflow(workflow_id, source_names, config)

        # Assert
        desired = desired_table(workflow_controller.desired_state)[workflow_id]
        for job in desired.jobs.values():
            assert job.params == {"threshold": 100.0, "mode": "fast"}

    def test_start_workflow_raises_for_nonexistent_workflow(
        self,
        workflow_controller: WorkflowControllerFixture,
        source_names: list[str],
    ):
        """Test that start_workflow raises ValueError for non-existent workflow."""
        nonexistent_workflow_id = "nonexistent_workflow"
        config = SomeWorkflowParams(threshold=100.0)

        # Act & Assert
        with pytest.raises(ValueError, match=r"Workflow spec for .* not found"):
            workflow_controller.controller.start_workflow(
                nonexistent_workflow_id, source_names, config
            )

        assert desired_table(workflow_controller.desired_state) == {}

    def test_persistent_config_stores_multiple_workflows(
        self,
        command_service: CommandService,
        fake_config_store: dict[str, dict],
        source_names: list[str],
    ):
        """Test that multiple workflow configurations can be stored persistently."""
        config_store = fake_config_store

        config_1 = SomeWorkflowParams(threshold=100.0, mode="fast")
        config_2 = SomeWorkflowParams(threshold=200.0, mode="accurate")
        sources_1 = ["detector_1"]
        sources_2 = ["detector_2"]

        # Add workflow specs to the controller's registry
        workflow_spec_1 = WorkflowSpec(
            instrument="test_instrument",
            name="Workflow 1",
            version=1,
            title="Workflow 1",
            description="First workflow",
            source_names=sources_1,
            params=SomeWorkflowParams,
            group=REDUCTION,
        )
        workflow_spec_2 = WorkflowSpec(
            instrument="test_instrument",
            name="Workflow 2",
            version=1,
            title="Workflow 2",
            description="Second workflow",
            source_names=sources_2,
            params=SomeWorkflowParams,
            group=REDUCTION,
        )
        workflow_id_1 = workflow_spec_1.get_id()
        workflow_id_2 = workflow_spec_2.get_id()

        registry = {workflow_id_1: workflow_spec_1, workflow_id_2: workflow_spec_2}

        # Create JobOrchestrator
        from ess.livedata.dashboard.active_job_registry import ActiveJobRegistry
        from ess.livedata.dashboard.data_service import DataService
        from ess.livedata.dashboard.job_orchestrator import JobOrchestrator
        from ess.livedata.dashboard.job_service import JobService

        ds = DataService()
        js = JobService()
        job_orchestrator = JobOrchestrator(
            desired_state=InMemoryDesiredStateLog(),
            command_service=command_service,
            workflow_registry=registry,
            active_job_registry=ActiveJobRegistry(data_service=ds, job_service=js),
            config_store=config_store,
        )

        controller = WorkflowController(
            job_orchestrator=job_orchestrator,
            workflow_registry=registry,
        )

        # Start both workflows
        controller.start_workflow(workflow_id_1, sources_1, config_1)
        controller.start_workflow(workflow_id_2, sources_2, config_2)

        # Assert - check ConfigStore instead of service
        config_1_data = config_store.get(str(workflow_id_1))
        assert config_1_data is not None
        assert set(config_1_data['jobs'].keys()) == set(sources_1)
        for source in sources_1:
            assert config_1_data['jobs'][source]['params'] == {
                "threshold": 100.0,
                "mode": "fast",
            }

        config_2_data = config_store.get(str(workflow_id_2))
        assert config_2_data is not None
        assert set(config_2_data['jobs'].keys()) == set(sources_2)
        for source in sources_2:
            assert config_2_data['jobs'][source]['params'] == {
                "threshold": 200.0,
                "mode": "accurate",
            }

    def test_persistent_config_replaces_existing_workflow(
        self,
        workflow_controller: WorkflowControllerFixture,
        workflow_id: WorkflowId,
    ):
        """Test that starting a workflow replaces existing persistent configuration."""
        # Start workflow with initial config
        initial_config = SomeWorkflowParams(threshold=100.0, mode="fast")
        initial_sources = ["detector_1"]
        workflow_controller.controller.start_workflow(
            workflow_id, initial_sources, initial_config
        )

        # Start same workflow with different config
        updated_config = SomeWorkflowParams(threshold=300.0, mode="accurate")
        updated_sources = ["detector_1", "detector_2"]
        workflow_controller.controller.start_workflow(
            workflow_id, updated_sources, updated_config
        )

        # Assert - check ConfigStore instead of service
        workflow_config_data = workflow_controller.config_store.get(str(workflow_id))
        assert workflow_config_data is not None

        # Should have the updated values
        assert set(workflow_config_data['jobs'].keys()) == set(updated_sources)
        for source in updated_sources:
            assert workflow_config_data['jobs'][source]['params'] == {
                "threshold": 300.0,
                "mode": "accurate",
            }

    def test_get_workflow_spec_returns_correct_spec(
        self,
        workflow_controller: WorkflowControllerFixture,
        workflow_id: WorkflowId,
        workflow_spec: WorkflowSpec,
    ):
        """Test that get_workflow_spec returns the correct specification."""
        # Act
        result = workflow_controller.controller.get_workflow_spec(workflow_id)

        # Assert
        assert result == workflow_spec
        assert result.title == "Test Workflow"
        assert result.description == "A test workflow for unit testing"

    def test_get_workflow_spec_returns_none_for_nonexistent(
        self,
        workflow_controller: WorkflowControllerFixture,
    ):
        """Test that get_workflow_spec returns None for non-existent workflow."""
        # Act
        result = workflow_controller.controller.get_workflow_spec(
            "nonexistent_workflow"
        )

        # Assert
        assert result is None

    def test_get_workflow_config_returns_persistent_config(
        self,
        workflow_controller: WorkflowControllerFixture,
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        """Test that get_workflow_config returns saved persistent configuration."""
        config = SomeWorkflowParams(threshold=150.0, mode="accurate")

        # Start workflow to create persistent config
        workflow_controller.controller.start_workflow(workflow_id, source_names, config)

        # Act - get_workflow_config returns reference config (single source)
        result = workflow_controller.controller.get_workflow_config(workflow_id)

        # Assert - returns reference config from first staged source
        assert result is not None
        assert result.params == {
            "threshold": 150.0,
            "mode": "accurate",
        }

    def test_get_workflow_config_returns_none_for_nonexistent(
        self,
        workflow_controller: WorkflowControllerFixture,
    ):
        """Test that get_workflow_config returns None for non-existent workflow."""
        nonexistent_id = WorkflowId(instrument='test', name='nonexistent', version=1)

        # Act
        result = workflow_controller.controller.get_workflow_config(nonexistent_id)

        # Assert
        assert result is None

    def test_start_workflow_with_empty_source_names_list(
        self,
        workflow_controller: WorkflowControllerFixture,
        workflow_id: WorkflowId,
    ):
        """Test that start_workflow returns empty list with empty source names."""
        config = SomeWorkflowParams(threshold=100.0)

        # Act
        result = workflow_controller.controller.start_workflow(workflow_id, [], config)

        # Assert
        assert result == []
        assert desired_table(workflow_controller.desired_state) == {}

    def test_start_workflow_writes_one_record(
        self,
        workflow_controller: WorkflowControllerFixture,
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        config = SomeWorkflowParams(threshold=150.0, mode="accurate")

        workflow_controller.controller.start_workflow(workflow_id, source_names, config)

        assert len(workflow_controller.desired_state.read(0)) == 1

    def test_restart_replaces_the_record(
        self,
        workflow_controller: WorkflowControllerFixture,
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        controller = workflow_controller.controller
        first = controller.start_workflow(
            workflow_id, source_names, SomeWorkflowParams(threshold=150.0)
        )
        second = controller.start_workflow(
            workflow_id, source_names, SomeWorkflowParams(threshold=300.0)
        )

        desired = desired_table(workflow_controller.desired_state)[workflow_id]
        assert desired.job_number == second[0].job_number != first[0].job_number
        assert desired.jobs['detector_1'].params['threshold'] == 300.0
