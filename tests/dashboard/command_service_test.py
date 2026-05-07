# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pytest

from ess.livedata.config.workflow_spec import JobId, WorkflowConfig, WorkflowId
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.message import COMMANDS_STREAM_ID
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.fakes import FakeMessageSink


def _workflow_config(source: str = "src") -> WorkflowConfig:
    return WorkflowConfig(
        identifier=WorkflowId(instrument="dummy", name="wf", version=1),
        job_id=JobId(source_name=source, job_number=uuid.uuid4()),
    )


def _job_command(source: str = "src") -> JobCommand:
    return JobCommand(
        job_id=JobId(source_name=source, job_number=uuid.uuid4()),
        action=JobAction.stop,
    )


@pytest.fixture
def fake_sink() -> FakeMessageSink:
    return FakeMessageSink()


@pytest.fixture
def command_service(fake_sink: FakeMessageSink) -> CommandService:
    return CommandService(sink=fake_sink)


class TestCommandService:
    def test_send_single_command(
        self, command_service: CommandService, fake_sink: FakeMessageSink
    ):
        config = _workflow_config()
        command_service.send(config)

        assert len(fake_sink.published_messages) == 1
        messages = fake_sink.published_messages[0]
        assert len(messages) == 1
        msg = messages[0]
        assert msg.stream == COMMANDS_STREAM_ID
        assert msg.value is config

    def test_send_batch(
        self, command_service: CommandService, fake_sink: FakeMessageSink
    ):
        commands = [_workflow_config("a"), _job_command("b"), _workflow_config("c")]
        command_service.send_batch(commands)

        assert len(fake_sink.published_messages) == 1
        messages = fake_sink.published_messages[0]
        assert len(messages) == 3
        for msg, expected in zip(messages, commands, strict=True):
            assert msg.stream == COMMANDS_STREAM_ID
            assert msg.value is expected

    def test_send_empty_batch(
        self, command_service: CommandService, fake_sink: FakeMessageSink
    ):
        command_service.send_batch([])
        assert len(fake_sink.published_messages) == 0

    def test_multiple_sends_create_multiple_batches(
        self, command_service: CommandService, fake_sink: FakeMessageSink
    ):
        command_service.send(_workflow_config("a"))
        command_service.send(_job_command("b"))

        assert len(fake_sink.published_messages) == 2
        assert len(fake_sink.published_messages[0]) == 1
        assert len(fake_sink.published_messages[1]) == 1
