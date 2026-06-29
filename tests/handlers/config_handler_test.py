# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

from ess.livedata.config.acknowledgement import (
    AcknowledgementResponse,
    CommandAcknowledgement,
)
from ess.livedata.config.workflow_spec import JobId, WorkflowConfig, WorkflowId
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.message import COMMANDS_STREAM_ID, RESPONSES_STREAM_ID, Message
from ess.livedata.handlers.config_handler import ConfigProcessor


def _job_id(source: str = "source1") -> JobId:
    return JobId(source_name=source, job_number=uuid.uuid4())


def _workflow_config(source: str = "source1", message_id: str | None = None):
    return WorkflowConfig(
        identifier=WorkflowId(instrument="dummy", name="wf", version=1),
        job_id=_job_id(source),
        message_id=message_id or str(uuid.uuid4()),
    )


def _msg(value):
    return Message(value=value, timestamp=123456789, stream=COMMANDS_STREAM_ID)


class FakeJobManagerAdapter:
    def __init__(self):
        self.workflow_calls: list[WorkflowConfig] = []
        self.job_command_calls: list[JobCommand] = []
        self.should_raise = False

    def job_command(self, command: JobCommand) -> CommandAcknowledgement | None:
        self.job_command_calls.append(command)
        if self.should_raise:
            raise ValueError("Test exception")
        return CommandAcknowledgement(
            message_id=command.message_id,
            device=str(command.job_id) if command.job_id else "all",
            response=AcknowledgementResponse.ACK,
        )

    def set_workflow_with_config(
        self, config: WorkflowConfig
    ) -> CommandAcknowledgement | None:
        self.workflow_calls.append(config)
        if self.should_raise:
            raise ValueError("Test exception")
        return CommandAcknowledgement(
            message_id=config.message_id,
            device=config.job_id.source_name,
            response=AcknowledgementResponse.ACK,
        )


class TestConfigProcessor:
    def test_process_workflow_config_dispatches_to_adapter(self):
        adapter = FakeJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=adapter)
        config = _workflow_config(message_id="msg-1")

        result_messages = processor.process_messages([_msg(config)])

        assert adapter.workflow_calls == [config]
        assert len(result_messages) == 1
        message = result_messages[0]
        assert message.stream == RESPONSES_STREAM_ID
        assert isinstance(message.value, CommandAcknowledgement)
        assert message.value.response == AcknowledgementResponse.ACK
        assert message.value.message_id == "msg-1"

    def test_process_job_command_with_message_id_yields_ack(self):
        adapter = FakeJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=adapter)
        command = JobCommand(message_id="msg-1", action=JobAction.reset)

        result_messages = processor.process_messages([_msg(command)])

        assert adapter.job_command_calls == [command]
        assert len(result_messages) == 1
        assert result_messages[0].value.message_id == "msg-1"

    def test_processes_multiple_commands_in_order(self):
        adapter = FakeJobManagerAdapter()
        processor = ConfigProcessor(job_manager_adapter=adapter)
        config = _workflow_config()
        command = JobCommand(action=JobAction.stop)

        processor.process_messages([_msg(config), _msg(command)])

        assert adapter.workflow_calls == [config]
        assert adapter.job_command_calls == [command]

    def test_adapter_exception_is_caught(self):
        adapter = FakeJobManagerAdapter()
        adapter.should_raise = True
        processor = ConfigProcessor(job_manager_adapter=adapter)

        result_messages = processor.process_messages([_msg(_workflow_config())])

        assert len(adapter.workflow_calls) == 1
        assert result_messages == []
