# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable

import pydantic
import pytest

from ess.livedata.config.workflow_spec import (
    WorkflowConfig,
    WorkflowId,
    WorkflowSpec,
    WorkflowStatus,
    WorkflowStatusType,
)
from ess.livedata.dashboard.configuration_adapter import ConfigurationState
from ess.livedata.dashboard.workflow_config_service import WorkflowConfigService
from ess.livedata.dashboard.workflow_controller import WorkflowController


class SomeWorkflowParams(pydantic.BaseModel):
    """Test Pydantic model for workflow parameters."""

    threshold: float = 100.0
    mode: str = "fast"


class FakeWorkflowConfigService(WorkflowConfigService):
    """Fake service for testing WorkflowController."""

    def __init__(self):
        self._sent_configs: list[tuple[str, WorkflowConfig]] = []
        self._status_callbacks: dict[str, list[Callable[[WorkflowStatus], None]]] = {}

    def send_workflow_config(self, source_name: str, config: WorkflowConfig) -> None:
        self._sent_configs.append((source_name, config))

    def subscribe_to_workflow_status(
        self, source_name: str, callback: Callable[[WorkflowStatus], None]
    ) -> None:
        if source_name not in self._status_callbacks:
            self._status_callbacks[source_name] = []
        self._status_callbacks[source_name].append(callback)

    def simulate_status_update(self, status: WorkflowStatus) -> None:
        """Test helper to simulate status updates."""
        for callback in self._status_callbacks.get(status.source_name, []):
            callback(status)

    def get_sent_configs(self) -> list[tuple[str, WorkflowConfig]]:
        """Test helper to get sent configs."""
        return self._sent_configs.copy()

    def clear_sent_configs(self) -> None:
        """Test helper to clear sent configs."""
        self._sent_configs.clear()


@pytest.fixture
def source_names() -> list[str]:
    """Test source names."""
    return ["detector_1", "detector_2"]


@pytest.fixture
def workflow_id() -> WorkflowId:
    """Test workflow ID."""
    return WorkflowId(
        instrument='test_instrument', namespace='abc', name="test_workflow", version=1
    )


@pytest.fixture
def workflow_spec(workflow_id: WorkflowId) -> WorkflowSpec:
    """Test workflow specification."""
    return WorkflowSpec(
        instrument=workflow_id.instrument,
        namespace=workflow_id.namespace,
        name=workflow_id.name,
        version=workflow_id.version,
        title="Test Workflow",
        description="A test workflow for unit testing",
        source_names=["detector_1", "detector_2"],
        params=SomeWorkflowParams,
    )


@pytest.fixture
def workflow_registry(
    workflow_id: WorkflowId, workflow_spec: WorkflowSpec
) -> dict[WorkflowId, WorkflowSpec]:
    """Test workflow registry."""
    return {workflow_id: workflow_spec}


@pytest.fixture
def fake_service() -> FakeWorkflowConfigService:
    """Fake service for testing."""
    return FakeWorkflowConfigService()


@pytest.fixture
def fake_config_store() -> dict[WorkflowId, dict]:
    """Plain dict for config store testing."""
    return {}


@pytest.fixture
def workflow_controller(
    fake_service: FakeWorkflowConfigService,
    fake_config_store: dict[WorkflowId, dict],
    source_names: list[str],
    workflow_registry: dict[WorkflowId, WorkflowSpec],
) -> tuple[WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]]:
    """Workflow controller instance for testing."""
    controller = WorkflowController(
        service=fake_service,
        source_names=source_names,
        workflow_registry=workflow_registry,
        config_store=fake_config_store,
    )
    return controller, fake_service, fake_config_store


class TestWorkflowController:
    def test_start_workflow_sends_config_to_sources(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        """Test that start_workflow sends configuration to all specified sources."""
        controller, service, _config_store = workflow_controller
        config = SomeWorkflowParams(threshold=150.0, mode="accurate")

        # Act
        controller.start_workflow(workflow_id, source_names, config)

        # Assert
        sent_configs = service.get_sent_configs()
        assert len(sent_configs) == len(source_names)

        for source_name in source_names:
            # Find the config for this source
            source_config = next(
                (sc for sc in sent_configs if sc[0] == source_name), None
            )
            assert source_config is not None
            assert source_config[1].identifier == workflow_id
            assert source_config[1].params == {"threshold": 150.0, "mode": "accurate"}

    def test_start_workflow_saves_persistent_config(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        """Test that start_workflow saves persistent configuration."""
        controller, _service, config_store = workflow_controller
        config = SomeWorkflowParams(threshold=200.0, mode="fast")

        # Act
        controller.start_workflow(workflow_id, source_names, config)

        # Assert - check ConfigStore instead of service
        persistent_config_data = config_store.get(workflow_id)
        assert persistent_config_data is not None
        persistent_config = ConfigurationState.model_validate(persistent_config_data)
        assert persistent_config.source_names == source_names
        assert persistent_config.params == {"threshold": 200.0, "mode": "fast"}

    def test_start_workflow_updates_status_to_starting(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        """Test that start_workflow immediately updates status to STARTING."""
        controller, service, _config_store = workflow_controller
        config = SomeWorkflowParams(threshold=75.0)

        # Set up callback to capture status
        captured_status = {}

        def capture_status(all_status):
            captured_status.update(all_status)

        controller.subscribe_to_workflow_status_updates(capture_status)
        captured_status.clear()  # Clear initial callback

        # Act
        controller.start_workflow(workflow_id, source_names, config)

        # Assert
        for source_name in source_names:
            status = captured_status[source_name]
            assert status.source_name == source_name
            assert status.workflow_id == workflow_id
            assert status.status == WorkflowStatusType.STARTING

    def test_start_workflow_with_empty_config(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        """Test that start_workflow works with empty configuration."""
        controller, service, _config_store = workflow_controller
        config = SomeWorkflowParams()  # Use defaults

        # Act
        controller.start_workflow(workflow_id, source_names, config)

        # Assert
        sent_configs = service.get_sent_configs()
        for _, workflow_config in sent_configs:
            assert workflow_config.identifier == workflow_id
            assert workflow_config.params == {"threshold": 100.0, "mode": "fast"}

    def test_start_workflow_with_single_source(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
    ):
        """Test that start_workflow works with a single source."""
        controller, service, _config_store = workflow_controller
        single_source = ["detector_1"]
        config = SomeWorkflowParams(threshold=300.0)

        # Set up callback to capture status
        captured_status = {}

        def capture_status(all_status):
            captured_status.update(all_status)

        controller.subscribe_to_workflow_status_updates(capture_status)
        captured_status.clear()  # Clear initial callback

        # Act
        controller.start_workflow(workflow_id, single_source, config)

        # Assert
        sent_configs = service.get_sent_configs()
        assert len(sent_configs) == 1
        assert sent_configs[0][0] == "detector_1"

        # Check status
        assert captured_status["detector_1"].status == WorkflowStatusType.STARTING
        assert captured_status["detector_1"].workflow_id == workflow_id

    def test_start_workflow_raises_for_nonexistent_workflow(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        source_names: list[str],
    ):
        """Test that start_workflow raises ValueError for non-existent workflow."""
        controller, service, _config_store = workflow_controller
        nonexistent_workflow_id = "nonexistent_workflow"
        config = SomeWorkflowParams(threshold=100.0)

        # Act & Assert
        with pytest.raises(ValueError, match="Workflow spec for .* not found"):
            controller.start_workflow(nonexistent_workflow_id, source_names, config)

        # Should not have sent any configs
        sent_configs = service.get_sent_configs()
        assert len(sent_configs) == 0

    def test_persistent_config_stores_multiple_workflows(
        self,
        fake_service: FakeWorkflowConfigService,
        fake_config_store: dict[WorkflowId, dict],
        source_names: list[str],
    ):
        """Test that multiple workflow configurations can be stored persistently."""
        service = fake_service
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
        )
        workflow_spec_2 = WorkflowSpec(
            instrument="test_instrument",
            name="Workflow 2",
            version=1,
            title="Workflow 2",
            description="Second workflow",
            source_names=sources_2,
            params=SomeWorkflowParams,
        )
        workflow_id_1 = workflow_spec_1.get_id()
        workflow_id_2 = workflow_spec_2.get_id()

        registry = {workflow_id_1: workflow_spec_1, workflow_id_2: workflow_spec_2}
        controller = WorkflowController(
            service=service,
            source_names=source_names,
            workflow_registry=registry,
            config_store=config_store,
        )

        # Start both workflows
        controller.start_workflow(workflow_id_1, sources_1, config_1)
        controller.start_workflow(workflow_id_2, sources_2, config_2)

        # Assert - check ConfigStore instead of service
        config_1_data = config_store.get(workflow_id_1)
        assert config_1_data is not None
        config_1 = ConfigurationState.model_validate(config_1_data)
        assert config_1.source_names == sources_1
        assert config_1.params == {"threshold": 100.0, "mode": "fast"}

        config_2_data = config_store.get(workflow_id_2)
        assert config_2_data is not None
        config_2 = ConfigurationState.model_validate(config_2_data)
        assert config_2.source_names == sources_2
        assert config_2.params == {"threshold": 200.0, "mode": "accurate"}

    def test_persistent_config_replaces_existing_workflow(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
    ):
        """Test that starting a workflow replaces existing persistent configuration."""
        controller, _service, config_store = workflow_controller

        # Start workflow with initial config
        initial_config = SomeWorkflowParams(threshold=100.0, mode="fast")
        initial_sources = ["detector_1"]
        controller.start_workflow(workflow_id, initial_sources, initial_config)

        # Start same workflow with different config
        updated_config = SomeWorkflowParams(threshold=300.0, mode="accurate")
        updated_sources = ["detector_1", "detector_2"]
        controller.start_workflow(workflow_id, updated_sources, updated_config)

        # Assert - check ConfigStore instead of service
        workflow_config_data = config_store.get(workflow_id)
        assert workflow_config_data is not None
        workflow_config = ConfigurationState.model_validate(workflow_config_data)

        # Should have the updated values
        assert workflow_config.source_names == updated_sources
        assert workflow_config.params == {"threshold": 300.0, "mode": "accurate"}

    def test_status_updates_from_service(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
    ):
        """Test that controller handles status updates from service."""
        controller, service, _config_store = workflow_controller

        # Set up callback to capture status
        captured_status = {}

        def capture_status(all_status):
            captured_status.update(all_status)

        controller.subscribe_to_workflow_status_updates(capture_status)

        # Simulate status update from service
        new_status = WorkflowStatus(
            source_name="detector_1",
            workflow_id=workflow_id,
            status=WorkflowStatusType.RUNNING,
        )
        service.simulate_status_update(new_status)

        # Check that controller received the update
        assert captured_status["detector_1"].status == WorkflowStatusType.RUNNING
        assert captured_status["detector_1"].workflow_id == workflow_id

    def test_get_workflow_spec_returns_correct_spec(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
        workflow_spec: WorkflowSpec,
    ):
        """Test that get_workflow_spec returns the correct specification."""
        controller, service, _config_store = workflow_controller

        # Act
        result = controller.get_workflow_spec(workflow_id)

        # Assert
        assert result == workflow_spec
        assert result.title == "Test Workflow"
        assert result.description == "A test workflow for unit testing"

    def test_get_workflow_spec_returns_none_for_nonexistent(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
    ):
        """Test that get_workflow_spec returns None for non-existent workflow."""
        controller, service, _config_store = workflow_controller

        # Act
        result = controller.get_workflow_spec("nonexistent_workflow")

        # Assert
        assert result is None

    def test_get_workflow_config_returns_persistent_config(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        """Test that get_workflow_config returns saved persistent configuration."""
        controller, service, _config_store = workflow_controller
        config = SomeWorkflowParams(threshold=150.0, mode="accurate")

        # Start workflow to create persistent config
        controller.start_workflow(workflow_id, source_names, config)

        # Act
        result = controller.get_workflow_config(workflow_id)

        # Assert
        assert result is not None
        assert result.source_names == source_names
        assert result.params == {"threshold": 150.0, "mode": "accurate"}

    def test_get_workflow_config_returns_none_for_nonexistent(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
    ):
        """Test that get_workflow_config returns None for non-existent workflow."""
        controller, service, _config_store = workflow_controller
        nonexistent_id = WorkflowId(
            instrument='test', namespace='test', name='nonexistent', version=1
        )

        # Act
        result = controller.get_workflow_config(nonexistent_id)

        # Assert
        assert result is None

    def test_subscribe_to_workflow_status_updates_calls_callback_immediately(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
    ):
        """Test that status updates subscription calls callback immediately."""
        controller, service, _config_store = workflow_controller
        callback_called = []

        def test_callback(all_status):
            callback_called.append(all_status)

        # Act - subscribe should trigger immediate callback
        controller.subscribe_to_workflow_status_updates(test_callback)

        # Assert
        assert len(callback_called) == 1
        # Should contain initial status for all sources
        assert len(callback_called[0]) == 2  # detector_1, detector_2

    def test_subscribe_to_workflow_status_updates_calls_callback_on_status_change(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
    ):
        """Test that status updates subscription works correctly."""
        controller, service, _config_store = workflow_controller
        callback_called = []

        def test_callback(all_status):
            callback_called.append(all_status)

        # Subscribe (will trigger immediate callback)
        controller.subscribe_to_workflow_status_updates(test_callback)
        initial_calls = len(callback_called)

        # Trigger status update
        status = WorkflowStatus(
            source_name="detector_1",
            workflow_id=workflow_id,
            status=WorkflowStatusType.RUNNING,
        )
        service.simulate_status_update(status)

        # Assert
        assert len(callback_called) == initial_calls + 1
        # Check that the status was updated
        latest_status = callback_called[-1]
        assert latest_status["detector_1"].status == WorkflowStatusType.RUNNING

    def test_controller_initializes_all_sources_with_unknown_status(
        self,
        fake_service: FakeWorkflowConfigService,
        workflow_registry: dict[WorkflowId, WorkflowSpec],
    ):
        """Test that controller initializes all sources with UNKNOWN status."""
        source_names = ["detector_1", "detector_2", "detector_3"]
        controller = WorkflowController(
            service=fake_service,
            source_names=source_names,
            workflow_registry=workflow_registry,
        )

        # Set up callback to capture initial status
        captured_status = {}

        def capture_status(all_status):
            captured_status.update(all_status)

        controller.subscribe_to_workflow_status_updates(capture_status)

        # Assert
        assert len(captured_status) == 3
        for source_name in source_names:
            assert source_name in captured_status
            assert captured_status[source_name].status == WorkflowStatusType.UNKNOWN
            assert captured_status[source_name].source_name == source_name
            assert captured_status[source_name].workflow_id is None

    def test_workflow_status_callback_exception_handling(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
    ):
        """Test that exceptions in workflow status callbacks are handled gracefully."""
        controller, service, _config_store = workflow_controller

        def failing_callback(all_status: dict[str, WorkflowStatus]):
            raise Exception("Test exception")

        def working_callback(all_status: dict[str, WorkflowStatus]):
            working_callback.called = True
            working_callback.received_status = all_status

        working_callback.called = False
        working_callback.received_status = {}

        # Subscribe both callbacks
        controller.subscribe_to_workflow_status_updates(failing_callback)
        controller.subscribe_to_workflow_status_updates(working_callback)

        # Reset call count after initial subscription calls
        working_callback.called = False

        # Trigger status update - should not crash and should call working callback
        status = WorkflowStatus(
            source_name="detector_1",
            workflow_id=workflow_id,
            status=WorkflowStatusType.RUNNING,
        )
        service.simulate_status_update(status)

        # Assert working callback was still called despite exception in failing one
        assert working_callback.called is True
        assert "detector_1" in working_callback.received_status
        assert (
            working_callback.received_status["detector_1"].status
            == WorkflowStatusType.RUNNING
        )

    def test_multiple_status_subscriptions_work_correctly(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
    ):
        """Test that multiple status update subscriptions work correctly."""
        controller, service, _config_store = workflow_controller
        callback1_calls = []
        callback2_calls = []

        def callback1(all_status: dict[str, WorkflowStatus]):
            callback1_calls.append(all_status)

        def callback2(all_status: dict[str, WorkflowStatus]):
            callback2_calls.append(all_status)

        # Subscribe both
        controller.subscribe_to_workflow_status_updates(callback1)
        controller.subscribe_to_workflow_status_updates(callback2)

        # Clear initial calls
        callback1_calls.clear()
        callback2_calls.clear()

        # Trigger update
        status = WorkflowStatus(
            source_name="detector_1",
            workflow_id=workflow_id,
            status=WorkflowStatusType.RUNNING,
        )
        service.simulate_status_update(status)

        # Assert both were called
        assert len(callback1_calls) == 1
        assert len(callback2_calls) == 1
        # Check that both received the same status and validate structure
        assert "detector_1" in callback1_calls[0]
        assert "detector_2" in callback1_calls[0]  # Should contain all sources
        assert callback1_calls[0]["detector_1"].status == WorkflowStatusType.RUNNING
        assert callback2_calls[0]["detector_1"].status == WorkflowStatusType.RUNNING
        # Verify the callbacks receive copies (not the same dict instance)
        assert callback1_calls[0] is not callback2_calls[0]

    def test_start_workflow_with_empty_source_names_list(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
    ):
        """Test that start_workflow works with empty source names list."""
        controller, service, config_store = workflow_controller
        config = SomeWorkflowParams(threshold=100.0)

        # Act
        controller.start_workflow(workflow_id, [], config)

        # Assert
        sent_configs = service.get_sent_configs()
        assert len(sent_configs) == 0  # No configs sent to sources

        # Should still save persistent config
        persistent_config_data = config_store.get(workflow_id)
        assert persistent_config_data is not None
        persistent_config = ConfigurationState.model_validate(persistent_config_data)
        assert persistent_config.source_names == []

    def test_callback_receives_complete_workflow_status_dict(
        self,
        workflow_controller: tuple[
            WorkflowController, FakeWorkflowConfigService, dict[WorkflowId, dict]
        ],
        workflow_id: WorkflowId,
        source_names: list[str],
    ):
        """Test that status callbacks receive complete status dict for all sources."""
        controller, service, _config_store = workflow_controller
        received_status = {}

        def capture_status(all_status: dict[str, WorkflowStatus]):
            received_status.update(all_status)

        controller.subscribe_to_workflow_status_updates(capture_status)

        # Verify initial state contains all sources
        assert len(received_status) == len(source_names)
        for source_name in source_names:
            assert source_name in received_status
            assert received_status[source_name].status == WorkflowStatusType.UNKNOWN

        # Clear and trigger update for one source
        received_status.clear()
        status = WorkflowStatus(
            source_name="detector_1",
            workflow_id=workflow_id,
            status=WorkflowStatusType.RUNNING,
        )
        service.simulate_status_update(status)

        # Should still receive status for all sources, not just the updated one
        assert len(received_status) == len(source_names)
        assert received_status["detector_1"].status == WorkflowStatusType.RUNNING
        assert received_status["detector_2"].status == WorkflowStatusType.UNKNOWN
