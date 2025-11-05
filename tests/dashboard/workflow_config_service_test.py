# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.config.keys import WORKFLOW_CONFIG, WORKFLOW_STATUS
from ess.livedata.config.models import ConfigKey
from ess.livedata.config.schema_registry import get_schema_registry
from ess.livedata.config.workflow_spec import (
    WorkflowConfig,
    WorkflowId,
    WorkflowStatus,
    WorkflowStatusType,
)
from ess.livedata.dashboard.config_service import ConfigService
from ess.livedata.dashboard.message_bridge import FakeMessageBridge
from ess.livedata.dashboard.schema_validator import PydanticSchemaValidator
from ess.livedata.dashboard.workflow_config_service import (
    ConfigServiceAdapter,
    WorkflowConfigService,
)


@pytest.fixture
def fake_message_bridge():
    """Create a fake message bridge for testing."""
    return FakeMessageBridge[ConfigKey, dict]()


@pytest.fixture
def config_service(fake_message_bridge):
    """Create a ConfigService with fake message bridge."""
    schema_validator = PydanticSchemaValidator(get_schema_registry())
    return ConfigService(schema_validator, fake_message_bridge)


@pytest.fixture
def workflow_config_service(config_service) -> WorkflowConfigService:
    """Create a ConfigServiceAdapter for testing."""
    return ConfigServiceAdapter(config_service)


@pytest.fixture
def sample_workflow_id():
    """Create a sample workflow ID for testing."""
    return WorkflowId(
        instrument="INSTR",
        namespace="data_reduction",
        name="test_workflow",
        version=1,
    )


@pytest.fixture
def sample_workflow_config(sample_workflow_id: WorkflowId):
    """Create a sample workflow config for testing."""
    return WorkflowConfig(identifier=sample_workflow_id, params={"param1": "value1"})


@pytest.fixture
def sample_workflow_status(sample_workflow_id):
    """Create a sample workflow status for testing."""
    return WorkflowStatus(
        source_name="source1",
        workflow_id=sample_workflow_id,
        status=WorkflowStatusType.RUNNING,
        message="Running successfully",
    )


def test_send_workflow_config(
    workflow_config_service, sample_workflow_config, fake_message_bridge
):
    """Test sending workflow config to a source."""
    source_name = "source1"
    workflow_config_service.send_workflow_config(source_name, sample_workflow_config)

    # Check that message was published
    published_messages = fake_message_bridge.get_published_messages()
    assert len(published_messages) == 1

    key, value = published_messages[0]
    assert key.source_name == source_name
    assert key.service_name == WORKFLOW_CONFIG.service_name
    assert key.key == WORKFLOW_CONFIG.key
    workflow_config = WorkflowConfig.model_validate(value)
    assert workflow_config == sample_workflow_config


def test_subscribe_to_workflow_status(
    workflow_config_service, sample_workflow_status, fake_message_bridge
):
    """Test subscribing to workflow status updates for a source."""
    source_name = "source1"
    received_statuses = []

    def callback(status: WorkflowStatus) -> None:
        received_statuses.append(status)

    workflow_config_service.subscribe_to_workflow_status(source_name, callback)

    # Simulate incoming workflow status update
    status_key = WORKFLOW_STATUS.create_key(source_name=source_name)
    serialized_status = sample_workflow_status.model_dump(mode='json')
    fake_message_bridge.add_incoming_message((status_key, serialized_status))
    workflow_config_service._config_service.process_incoming_messages()

    assert len(received_statuses) == 1
    assert received_statuses[0].source_name == sample_workflow_status.source_name
    assert received_statuses[0].workflow_id == sample_workflow_status.workflow_id
    assert received_statuses[0].status == sample_workflow_status.status
    assert received_statuses[0].message == sample_workflow_status.message


def test_subscribe_to_workflow_status_different_sources(
    workflow_config_service, sample_workflow_status, fake_message_bridge
):
    """Test that workflow status subscriptions are source-specific."""
    source1_statuses = []
    source2_statuses = []

    def source1_callback(status: WorkflowStatus) -> None:
        source1_statuses.append(status)

    def source2_callback(status: WorkflowStatus) -> None:
        source2_statuses.append(status)

    workflow_config_service.subscribe_to_workflow_status("source1", source1_callback)
    workflow_config_service.subscribe_to_workflow_status("source2", source2_callback)

    # Send update for source1 only
    status_key = WORKFLOW_STATUS.create_key(source_name="source1")
    serialized_status = sample_workflow_status.model_dump(mode='json')
    fake_message_bridge.add_incoming_message((status_key, serialized_status))
    workflow_config_service._config_service.process_incoming_messages()

    assert len(source1_statuses) == 1
    assert len(source2_statuses) == 0


def test_multiple_subscribers_same_key(
    workflow_config_service, sample_workflow_status, fake_message_bridge
):
    """Test that multiple subscribers to the same key all receive updates."""
    source_name = "source1"
    received_statuses_1 = []
    received_statuses_2 = []

    def callback1(status: WorkflowStatus) -> None:
        received_statuses_1.append(status)

    def callback2(status: WorkflowStatus) -> None:
        received_statuses_2.append(status)

    workflow_config_service.subscribe_to_workflow_status(source_name, callback1)
    workflow_config_service.subscribe_to_workflow_status(source_name, callback2)

    # Send update
    status_key = WORKFLOW_STATUS.create_key(source_name=source_name)
    serialized_status = sample_workflow_status.model_dump(mode='json')
    fake_message_bridge.add_incoming_message((status_key, serialized_status))
    workflow_config_service._config_service.process_incoming_messages()

    assert len(received_statuses_1) == 1
    assert len(received_statuses_2) == 1
    assert received_statuses_1[0].workflow_id == sample_workflow_status.workflow_id
    assert received_statuses_2[0].workflow_id == sample_workflow_status.workflow_id


def test_callback_receives_existing_data_on_subscription(
    workflow_config_service, sample_workflow_status, fake_message_bridge
):
    """Test that new subscribers receive existing data immediately."""
    source_name = "source1"

    # First, set up some existing data
    status_key = WORKFLOW_STATUS.create_key(source_name=source_name)
    serialized_status = sample_workflow_status.model_dump(mode='json')
    fake_message_bridge.add_incoming_message((status_key, serialized_status))
    workflow_config_service._config_service.process_incoming_messages()

    # Now subscribe and check that callback is called immediately
    received_statuses = []

    def callback(status: WorkflowStatus) -> None:
        received_statuses.append(status)

    workflow_config_service.subscribe_to_workflow_status(source_name, callback)

    assert len(received_statuses) == 1
    assert received_statuses[0].workflow_id == sample_workflow_status.workflow_id
