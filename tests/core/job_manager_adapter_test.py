# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pytest

from ess.livedata.config.acknowledgement import (
    AcknowledgementResponse,
    CommandAcknowledgement,
)
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.job_manager_adapter import JobManagerAdapter
from ess.livedata.core.message import RESPONSES_STREAM_ID


class FakeJobManager:
    """Fake JobManager for testing JobManagerAdapter."""

    def __init__(self):
        self.job_command_calls = []
        self.scheduled = []
        self.pending_messages: list = []
        self.should_raise_key_error = False
        self.should_raise_exception = False

    def job_command(self, command: JobCommand) -> None:
        self.job_command_calls.append(command)
        if self.should_raise_key_error:
            raise KeyError(f"Job {command.job_id} not found")
        if self.should_raise_exception:
            raise RuntimeError("Test exception")

    def schedule_job(self, *, source_name, config):
        self.scheduled.append((source_name, config))
        if self.should_raise_exception:
            raise RuntimeError("Test exception")
        return JobId(source_name=source_name, job_number=uuid.uuid4())

    def drain_pending_messages(self):
        drained = self.pending_messages
        self.pending_messages = []
        return drained


class TestJobManagerAdapterJobCommand:
    @pytest.fixture
    def fake_job_manager(self):
        return FakeJobManager()

    @pytest.fixture
    def adapter(self, fake_job_manager):
        return JobManagerAdapter(job_manager=fake_job_manager)

    def test_job_command_success_with_message_id_returns_ack_message(
        self, adapter, fake_job_manager
    ):
        job_id = JobId(source_name="test_source", job_number=uuid.uuid4())
        messages = adapter.job_command(
            source_name="ignored",
            value={
                "action": JobAction.reset.value,
                "job_id": {
                    "source_name": job_id.source_name,
                    "job_number": str(job_id.job_number),
                },
                "message_id": "test-msg-id",
            },
        )

        assert len(messages) == 1
        message = messages[0]
        assert message.stream == RESPONSES_STREAM_ID
        assert isinstance(message.value, CommandAcknowledgement)
        assert message.value.message_id == "test-msg-id"
        assert message.value.response == AcknowledgementResponse.ACK
        assert len(fake_job_manager.job_command_calls) == 1

    def test_job_command_success_without_message_id_returns_empty(
        self, adapter, fake_job_manager
    ):
        messages = adapter.job_command(
            source_name="ignored",
            value={"action": JobAction.reset.value},
        )

        assert messages == []
        assert len(fake_job_manager.job_command_calls) == 1

    def test_job_command_key_error_silently_returns_empty(
        self, adapter, fake_job_manager
    ):
        """KeyError (job not found) is silently ignored.

        Multiple backend services receive the same job command; only the owner
        responds. Non-owners return an empty list rather than ERR.
        """
        fake_job_manager.should_raise_key_error = True

        messages = adapter.job_command(
            source_name="ignored",
            value={
                "action": JobAction.stop.value,
                "message_id": "test-msg-id",
            },
        )

        assert messages == []

    def test_job_command_other_exception_returns_err_message(
        self, adapter, fake_job_manager
    ):
        fake_job_manager.should_raise_exception = True

        messages = adapter.job_command(
            source_name="ignored",
            value={
                "action": JobAction.stop.value,
                "message_id": "test-msg-id",
            },
        )

        assert len(messages) == 1
        ack = messages[0].value
        assert isinstance(ack, CommandAcknowledgement)
        assert ack.message_id == "test-msg-id"
        assert ack.response == AcknowledgementResponse.ERR
        assert "Test exception" in ack.message

    def test_job_command_other_exception_without_message_id_returns_empty(
        self, adapter, fake_job_manager
    ):
        fake_job_manager.should_raise_exception = True

        messages = adapter.job_command(
            source_name="ignored",
            value={"action": JobAction.stop.value},
        )

        assert messages == []


class TestJobManagerAdapterSetWorkflowWithConfig:
    @pytest.fixture
    def fake_job_manager(self):
        return FakeJobManager()

    @pytest.fixture
    def adapter(self, fake_job_manager):
        return JobManagerAdapter(job_manager=fake_job_manager)

    def _config(self, message_id="msg-1"):
        from ess.livedata.config.workflow_spec import WorkflowConfig, WorkflowId

        return WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="wf",
                version=1,
            ),
            message_id=message_id,
        ).model_dump(mode="json")

    def test_returns_only_ack_when_no_pending_messages(self, adapter, fake_job_manager):
        messages = adapter.set_workflow_with_config(
            source_name="src", value=self._config()
        )

        assert len(messages) == 1
        ack = messages[0].value
        assert isinstance(ack, CommandAcknowledgement)
        assert ack.response == AcknowledgementResponse.ACK
        assert messages[0].stream == RESPONSES_STREAM_ID
        assert len(fake_job_manager.scheduled) == 1

    def test_returns_ack_then_drained_messages(self, adapter, fake_job_manager):
        """Pending messages from inline one-shot jobs are drained and returned."""
        from ess.livedata.core.message import (
            STATUS_STREAM_ID,
            Message,
            StreamId,
            StreamKind,
        )

        # Two stand-in messages: one data-stream, one status-stream — exactly
        # what JobManager._run_one_shot stashes for a one-shot job.
        data_message = Message(
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name='r'),
            value='data-payload',
        )
        status_message = Message(stream=STATUS_STREAM_ID, value='status-payload')
        fake_job_manager.pending_messages = [data_message, status_message]

        messages = adapter.set_workflow_with_config(
            source_name="src", value=self._config()
        )

        assert len(messages) == 3
        assert isinstance(messages[0].value, CommandAcknowledgement)
        assert messages[1] is data_message
        assert messages[2] is status_message
        assert fake_job_manager.pending_messages == []

    def test_different_instrument_returns_empty(self, adapter, fake_job_manager):
        from ess.livedata.core.job_manager import DifferentInstrument

        def raise_different(*, source_name, config):
            raise DifferentInstrument()

        fake_job_manager.schedule_job = raise_different

        messages = adapter.set_workflow_with_config(
            source_name="src", value=self._config()
        )

        assert messages == []

    def test_exception_returns_err_ack(self, adapter, fake_job_manager):
        fake_job_manager.should_raise_exception = True

        messages = adapter.set_workflow_with_config(
            source_name="src", value=self._config()
        )

        assert len(messages) == 1
        ack = messages[0].value
        assert isinstance(ack, CommandAcknowledgement)
        assert ack.response == AcknowledgementResponse.ERR
        assert "Test exception" in ack.message
