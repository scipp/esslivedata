# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ess.livedata.config.acknowledgement import (
    AcknowledgementResponse,
    CommandAcknowledgement,
)
from ess.livedata.core.command_dispatcher import CommandDispatcher
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.message import COMMANDS_STREAM_ID, RESPONSES_STREAM_ID, Message


def _msg(value):
    return Message(value=value, timestamp=123456789, stream=COMMANDS_STREAM_ID)


class FakeJobManagerAdapter:
    def __init__(self, *, acknowledge: bool = True):
        self.job_command_calls: list[JobCommand] = []
        self._acknowledge = acknowledge

    def job_command(self, command: JobCommand) -> CommandAcknowledgement | None:
        self.job_command_calls.append(command)
        if not self._acknowledge:
            return None
        return CommandAcknowledgement(
            message_id=command.message_id,
            device=str(command.job_id) if command.job_id else "all",
            response=AcknowledgementResponse.ACK,
        )


class TestCommandDispatcher:
    def test_process_job_command_with_message_id_yields_ack(self):
        adapter = FakeJobManagerAdapter()
        processor = CommandDispatcher(job_manager_adapter=adapter)
        command = JobCommand(message_id="msg-1", action=JobAction.reset)

        result_messages = processor.process_messages([_msg(command)])

        assert adapter.job_command_calls == [command]
        assert len(result_messages) == 1
        assert result_messages[0].stream == RESPONSES_STREAM_ID
        assert result_messages[0].value.message_id == "msg-1"

    def test_processes_multiple_commands_in_order(self):
        adapter = FakeJobManagerAdapter()
        processor = CommandDispatcher(job_manager_adapter=adapter)
        first = JobCommand(action=JobAction.reset)
        second = JobCommand(action=JobAction.reset)

        processor.process_messages([_msg(first), _msg(second)])

        assert adapter.job_command_calls == [first, second]

    def test_command_without_ack_yields_no_response(self):
        adapter = FakeJobManagerAdapter(acknowledge=False)
        processor = CommandDispatcher(job_manager_adapter=adapter)

        result_messages = processor.process_messages(
            [_msg(JobCommand(action=JobAction.reset))]
        )

        assert result_messages == []
