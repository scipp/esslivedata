# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import json

import pytest

from ess.livedata.config.acknowledgement import (
    AcknowledgementResponse,
    CommandAcknowledgement,
)
from ess.livedata.config.models import ConfigKey
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.message import COMMANDS_STREAM_ID, RESPONSES_STREAM_ID, Message
from ess.livedata.handlers.config_handler import ConfigProcessor, ConfigUpdate
from ess.livedata.kafka.message_adapter import RawConfigItem


class TestConfigUpdate:
    def test_init(self):
        config_key = ConfigKey(
            source_name="source1", service_name="service1", key="test_key"
        )
        value = {"param": 42}
        update = ConfigUpdate(config_key=config_key, value=value)

        assert update.config_key is config_key
        assert update.value is value

        # Test properties
        assert update.source_name == "source1"
        assert update.service_name == "service1"
        assert update.key == "test_key"

    def test_from_raw(self):
        item = RawConfigItem(
            key=b'source1/service1/test_key',
            value=json.dumps({'param': 42}).encode('utf-8'),
        )

        update = ConfigUpdate.from_raw(item)

        assert update.source_name == "source1"
        assert update.service_name == "service1"
        assert update.key == "test_key"
        assert update.value == {'param': 42}

    def test_from_raw_with_wildcards(self):
        item = RawConfigItem(
            key=b'*/*/test_key',
            value=json.dumps({'param': 42}).encode('utf-8'),
        )

        update = ConfigUpdate.from_raw(item)

        assert update.source_name is None
        assert update.service_name is None
        assert update.key == "test_key"
        assert update.value == {'param': 42}

    def test_from_raw_invalid_key(self):
        item = RawConfigItem(
            key=b'invalid_key_format',
            value=json.dumps({'param': 42}).encode('utf-8'),
        )

        with pytest.raises(ValueError, match="Invalid key format"):
            ConfigUpdate.from_raw(item)

    def test_from_raw_invalid_json(self):
        item = RawConfigItem(key=b'source1/service1/test_key', value=b'not valid json')

        with pytest.raises(json.JSONDecodeError):
            ConfigUpdate.from_raw(item)


class TestConfigProcessor:
    def test_init(self):
        mock_job_manager = MockJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=mock_job_manager)
        assert processor._job_manager_adapter is mock_job_manager

    def test_process_workflow_config_message(self):
        mock_job_manager = MockJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=mock_job_manager)

        messages = [
            Message(
                value=RawConfigItem(
                    key=b'source1/service1/workflow_config',
                    value=json.dumps({"workflow": "test_workflow"}).encode('utf-8'),
                ),
                timestamp=123456789,
                stream=COMMANDS_STREAM_ID,
            )
        ]

        result_messages = processor.process_messages(messages)

        # Verify job manager was called
        assert len(mock_job_manager.workflow_calls) == 1
        assert mock_job_manager.workflow_calls[0] == (
            "source1",
            {"workflow": "test_workflow"},
        )

        # Verify response messages go to RESPONSES stream
        assert len(result_messages) == 1
        message = result_messages[0]
        assert message.stream == RESPONSES_STREAM_ID
        assert isinstance(message.value, CommandAcknowledgement)
        assert message.value.response == AcknowledgementResponse.ACK

    def test_process_job_command_message(self):
        mock_job_manager = MockJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=mock_job_manager)

        job_command = JobCommand(message_id="test-msg-id", action=JobAction.reset)
        messages = [
            Message(
                value=RawConfigItem(
                    key=b'source1/service1/job_command',
                    value=job_command.model_dump_json().encode('utf-8'),
                ),
                timestamp=123456789,
                stream=COMMANDS_STREAM_ID,
            )
        ]

        result_messages = processor.process_messages(messages)

        # Verify job manager was called
        assert len(mock_job_manager.job_command_calls) == 1
        assert mock_job_manager.job_command_calls[0] == (
            "source1",
            job_command.model_dump(),
        )

        # Verify response messages (ack for commands with message_id)
        assert len(result_messages) == 1
        assert result_messages[0].value.message_id == "test-msg-id"

    def test_process_job_command_without_message_id_no_ack(self):
        mock_job_manager = MockJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=mock_job_manager)

        job_command = JobCommand(action=JobAction.reset)  # No message_id
        messages = [
            Message(
                value=RawConfigItem(
                    key=b'source1/service1/job_command',
                    value=job_command.model_dump_json().encode('utf-8'),
                ),
                timestamp=123456789,
                stream=COMMANDS_STREAM_ID,
            )
        ]

        result_messages = processor.process_messages(messages)

        # Verify job manager was called
        assert len(mock_job_manager.job_command_calls) == 1

        # No acknowledgement for commands without message_id
        assert len(result_messages) == 0

    def test_process_unknown_config_key(self):
        mock_job_manager = MockJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=mock_job_manager)

        messages = [
            Message(
                value=RawConfigItem(
                    key=b'source1/service1/unknown_key',
                    value=json.dumps("some_value").encode('utf-8'),
                ),
                timestamp=123456789,
                stream=COMMANDS_STREAM_ID,
            )
        ]

        result_messages = processor.process_messages(messages)

        # Should not call job manager for unknown keys
        assert len(mock_job_manager.workflow_calls) == 0
        assert len(mock_job_manager.job_command_calls) == 0
        # Should not produce any response messages for unknown keys
        assert len(result_messages) == 0

    def test_process_multiple_messages_same_batch(self):
        mock_job_manager = MockJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=mock_job_manager)

        job_command = JobCommand(action=JobAction.stop)
        messages = [
            Message(
                value=RawConfigItem(
                    key=b'source1/service1/workflow_config',
                    value=json.dumps({"workflow": "test1"}).encode('utf-8'),
                ),
                timestamp=123456789,
                stream=COMMANDS_STREAM_ID,
            ),
            Message(
                value=RawConfigItem(
                    key=b'source2/service1/job_command',
                    value=job_command.model_dump_json().encode('utf-8'),
                ),
                timestamp=123456790,
                stream=COMMANDS_STREAM_ID,
            ),
        ]

        _ = processor.process_messages(messages)

        # Verify both calls were made
        assert len(mock_job_manager.workflow_calls) == 1
        assert len(mock_job_manager.job_command_calls) == 1
        assert mock_job_manager.workflow_calls[0] == ("source1", {"workflow": "test1"})
        assert mock_job_manager.job_command_calls[0] == (
            "source2",
            job_command.model_dump(),
        )

    def test_process_duplicate_sources_latest_value_wins(self):
        mock_job_manager = MockJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=mock_job_manager)

        messages = [
            Message(
                value=RawConfigItem(
                    key=b'source1/service1/workflow_config',
                    value=json.dumps({"workflow": "first"}).encode('utf-8'),
                ),
                timestamp=123456789,
                stream=COMMANDS_STREAM_ID,
            ),
            Message(
                value=RawConfigItem(
                    key=b'source1/service1/workflow_config',
                    value=json.dumps({"workflow": "latest"}).encode('utf-8'),
                ),
                timestamp=123456790,
                stream=COMMANDS_STREAM_ID,
            ),
        ]

        _ = processor.process_messages(messages)

        # Should only call job manager once with latest value
        assert len(mock_job_manager.workflow_calls) == 1
        assert mock_job_manager.workflow_calls[0] == ("source1", {"workflow": "latest"})

    def test_process_global_source_wildcard(self):
        mock_job_manager = MockJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=mock_job_manager)

        messages = [
            Message(
                value=RawConfigItem(
                    key=b'*/service1/workflow_config',
                    value=json.dumps({"workflow": "global"}).encode('utf-8'),
                ),
                timestamp=123456789,
                stream=COMMANDS_STREAM_ID,
            )
        ]

        _ = processor.process_messages(messages)

        # Should call job manager with None source_name for global updates
        assert len(mock_job_manager.workflow_calls) == 1
        assert mock_job_manager.workflow_calls[0] == (None, {"workflow": "global"})

    def test_global_override_behavior(self):
        mock_job_manager = MockJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=mock_job_manager)

        messages = [
            Message(
                value=RawConfigItem(
                    key=b'source1/service1/workflow_config',
                    value=json.dumps({"workflow": "source1"}).encode('utf-8'),
                ),
                timestamp=123456789,
                stream=COMMANDS_STREAM_ID,
            ),
            Message(
                value=RawConfigItem(
                    key=b'*/service1/workflow_config',
                    value=json.dumps({"workflow": "global"}).encode('utf-8'),
                ),
                timestamp=123456790,
                stream=COMMANDS_STREAM_ID,
            ),
        ]

        _ = processor.process_messages(messages)

        # Global update should clear source-specific updates for same key
        assert len(mock_job_manager.workflow_calls) == 1
        assert mock_job_manager.workflow_calls[0] == (None, {"workflow": "global"})

    def test_message_exception_handling(self):
        mock_job_manager = MockJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=mock_job_manager)

        # Send a message with invalid JSON
        messages = [
            Message(
                value=RawConfigItem(
                    key=b'source1/service1/workflow_config',
                    value=b'not valid json',
                ),
                timestamp=123456789,
                stream=COMMANDS_STREAM_ID,
            )
        ]

        # Exception should be caught, not propagated
        result_messages = processor.process_messages(messages)

        # No calls should be made due to the error
        assert len(mock_job_manager.workflow_calls) == 0
        assert len(result_messages) == 0

    def test_job_manager_exception_handling(self):
        mock_job_manager = MockJobManagerAdapter()
        mock_job_manager.should_raise_exception = True
        processor = ConfigProcessor(job_manager_adapter=mock_job_manager)

        messages = [
            Message(
                value=RawConfigItem(
                    key=b'source1/service1/workflow_config',
                    value=json.dumps({"workflow": "test"}).encode('utf-8'),
                ),
                timestamp=123456789,
                stream=COMMANDS_STREAM_ID,
            )
        ]

        # Exception should be caught, not propagated
        result_messages = processor.process_messages(messages)

        # Call should have been attempted
        assert len(mock_job_manager.workflow_calls) == 1
        # But no results should be returned due to exception
        assert len(result_messages) == 0


class MockJobManagerAdapter:
    def __init__(self):
        self.workflow_calls = []
        self.job_command_calls = []
        self.should_raise_exception = False

    def job_command(self, source_name, command):
        self.job_command_calls.append((source_name, command))
        if self.should_raise_exception:
            raise ValueError("Test exception")
        # Return acknowledgement only if message_id is present
        message_id = command.get("message_id")
        if message_id is None:
            return None
        return CommandAcknowledgement(
            message_id=message_id,
            device=str(command.get("job_id", "all")),
            response=AcknowledgementResponse.ACK,
        )

    def set_workflow_with_config(self, source_name, config):
        self.workflow_calls.append((source_name, config))
        if self.should_raise_exception:
            raise ValueError("Test exception")
        # Return acknowledgement (always for workflow config in mock)
        return CommandAcknowledgement(
            message_id=config.get("message_id", "mock-msg-id"),
            device=source_name,
            response=AcknowledgementResponse.ACK,
        )
