# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pytest

from ess.livedata.config.acknowledgement import AcknowledgementResponse
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.job_manager_adapter import JobManagerAdapter


class FakeJobManager:
    """Fake JobManager for testing JobManagerAdapter."""

    def __init__(self):
        self.job_command_calls = []
        self.should_raise_key_error = False
        self.should_raise_exception = False

    def job_command(self, command: JobCommand) -> None:
        self.job_command_calls.append(command)
        if self.should_raise_key_error:
            raise KeyError(f"Job {command.job_id} not found")
        if self.should_raise_exception:
            raise RuntimeError("Test exception")


class TestJobManagerAdapter:
    @pytest.fixture
    def fake_job_manager(self):
        return FakeJobManager()

    @pytest.fixture
    def adapter(self, fake_job_manager):
        return JobManagerAdapter(job_manager=fake_job_manager)

    def test_job_command_success_with_message_id(self, adapter, fake_job_manager):
        """Test successful job command returns ACK when message_id is provided."""
        job_id = JobId(source_name="test_source", job_number=uuid.uuid4())
        result = adapter.job_command(
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

        assert result is not None
        assert result.message_id == "test-msg-id"
        assert result.response == AcknowledgementResponse.ACK
        assert len(fake_job_manager.job_command_calls) == 1

    def test_job_command_success_without_message_id(self, adapter, fake_job_manager):
        """Test successful job command returns None when no message_id."""
        result = adapter.job_command(
            source_name="ignored",
            value={"action": JobAction.reset.value},
        )

        assert result is None
        assert len(fake_job_manager.job_command_calls) == 1

    def test_job_command_key_error_silently_ignored(self, adapter, fake_job_manager):
        """Test that KeyError (job not found) is silently ignored.

        When multiple backend services receive the same job command, only the service
        that owns the job should respond. Services that don't have the job should
        silently ignore it (return None) rather than returning an ERR.
        """
        fake_job_manager.should_raise_key_error = True

        job_id = JobId(source_name="test_source", job_number=uuid.uuid4())
        result = adapter.job_command(
            source_name="ignored",
            value={
                "action": JobAction.stop.value,
                "job_id": {
                    "source_name": job_id.source_name,
                    "job_number": str(job_id.job_number),
                },
                "message_id": "test-msg-id",
            },
        )

        # Should return None (no response) instead of ERR
        assert result is None

    def test_job_command_other_exception_returns_err(self, adapter, fake_job_manager):
        """Test that non-KeyError exceptions return ERR acknowledgement."""
        fake_job_manager.should_raise_exception = True

        job_id = JobId(source_name="test_source", job_number=uuid.uuid4())
        result = adapter.job_command(
            source_name="ignored",
            value={
                "action": JobAction.stop.value,
                "job_id": {
                    "source_name": job_id.source_name,
                    "job_number": str(job_id.job_number),
                },
                "message_id": "test-msg-id",
            },
        )

        assert result is not None
        assert result.message_id == "test-msg-id"
        assert result.response == AcknowledgementResponse.ERR
        assert "Test exception" in result.message

    def test_job_command_other_exception_without_message_id_returns_none(
        self, adapter, fake_job_manager
    ):
        """Test that exceptions without message_id return None."""
        fake_job_manager.should_raise_exception = True

        result = adapter.job_command(
            source_name="ignored",
            value={"action": JobAction.stop.value},
        )

        # No message_id means no response even on error
        assert result is None
