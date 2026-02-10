# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for JobOrchestrator initialization and config management."""

from enum import Enum
from typing import Any

import pydantic
import pytest
import scipp as sc
import yaml

from ess.livedata.config.models import ConfigKey
from ess.livedata.config.workflow_spec import (
    WorkflowConfig,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.message import COMMANDS_STREAM_ID
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.config_store import FileBackedConfigStore
from ess.livedata.dashboard.job_orchestrator import JobOrchestrator
from ess.livedata.fakes import FakeMessageSink
from ess.livedata.handlers.config_handler import ConfigUpdate


class WorkflowParams(pydantic.BaseModel):
    """Test params model with defaults."""

    threshold: float = 100.0
    mode: str = "default"


class AuxSources(pydantic.BaseModel):
    """Test aux sources model with defaults."""

    monitor: str = "monitor_1"


class ParamsWithRequiredFields(pydantic.BaseModel):
    """Params model with required fields (no defaults) - like correlation histograms."""

    required_value: float  # No default!


class SampleEnum(str, Enum):
    """Sample enum for verifying enum serialization."""

    OPTION_A = 'option_a'
    OPTION_B = 'option_b'


class ParamsWithEnum(pydantic.BaseModel):
    """Params model with enum field to test serialization."""

    value: float = 50.0
    choice: SampleEnum = SampleEnum.OPTION_A


class SimpleTestOutputs(WorkflowOutputsBase):
    """Simple outputs model for testing."""

    result: sc.DataArray = pydantic.Field(title='Result')


@pytest.fixture
def workflow_with_params() -> WorkflowSpec:
    """Workflow spec with params model."""
    return WorkflowSpec(
        instrument="test",
        namespace="testing",
        name="with_params",
        version=1,
        title="Workflow With Params",
        description="Test workflow with params",
        source_names=["det_1", "det_2"],
        params=WorkflowParams,
        outputs=SimpleTestOutputs,
    )


@pytest.fixture
def workflow_with_params_and_aux() -> WorkflowSpec:
    """Workflow spec with both params and aux sources."""
    return WorkflowSpec(
        instrument="test",
        namespace="testing",
        name="with_params_and_aux",
        version=1,
        title="Workflow With Params and Aux",
        description="Test workflow with params and aux sources",
        source_names=["det_1"],
        params=WorkflowParams,
        aux_sources=AuxSources,
        outputs=SimpleTestOutputs,
    )


@pytest.fixture
def workflow_no_params() -> WorkflowSpec:
    """Workflow spec without params model."""
    return WorkflowSpec(
        instrument="test",
        namespace="testing",
        name="no_params",
        version=1,
        title="Workflow Without Params",
        description="Test workflow without params",
        source_names=["det_1", "det_2"],
        params=None,
        outputs=SimpleTestOutputs,
    )


@pytest.fixture
def workflow_empty_sources() -> WorkflowSpec:
    """Workflow spec with empty source_names."""
    return WorkflowSpec(
        instrument="test",
        namespace="testing",
        name="empty_sources",
        version=1,
        title="Workflow With Empty Sources",
        description="Test workflow with no sources",
        source_names=[],
        params=WorkflowParams,
        outputs=SimpleTestOutputs,
    )


@pytest.fixture
def workflow_params_without_defaults() -> WorkflowSpec:
    """Workflow spec with params that can't be instantiated (required fields)."""
    return WorkflowSpec(
        instrument="test",
        namespace="testing",
        name="params_without_defaults",
        version=1,
        title="Workflow With Required Params",
        description="Test workflow like correlation histograms",
        source_names=["det_1"],
        params=ParamsWithRequiredFields,
        outputs=SimpleTestOutputs,
    )


@pytest.fixture
def workflow_with_enum_params() -> WorkflowSpec:
    """Workflow spec with params containing enum fields."""
    return WorkflowSpec(
        instrument="test",
        namespace="testing",
        name="with_enum_params",
        version=1,
        title="Workflow With Enum Params",
        description="Test workflow with enum params for serialization testing",
        source_names=["det_1"],
        params=ParamsWithEnum,
        outputs=SimpleTestOutputs,
    )


def get_sent_commands(sink: FakeMessageSink) -> list[tuple[ConfigKey, Any]]:
    """Extract all sent commands from the sink."""
    result = []
    for messages in sink.published_messages:
        result.extend(
            (msg.value.config_key, msg.value.value)
            for msg in messages
            if msg.stream == COMMANDS_STREAM_ID and isinstance(msg.value, ConfigUpdate)
        )
    return result


def get_sent_workflow_configs(
    sink: FakeMessageSink,
) -> list[tuple[str, WorkflowConfig]]:
    """Extract workflow configs with source names from the sink."""
    result = []
    for messages in sink.published_messages:
        result.extend(
            (msg.value.config_key.source_name, msg.value.value)
            for msg in messages
            if msg.stream == COMMANDS_STREAM_ID
            and isinstance(msg.value, ConfigUpdate)
            and isinstance(msg.value.value, WorkflowConfig)
        )
    return result


def get_batch_calls(sink: FakeMessageSink) -> list[int]:
    """Extract batch call sizes from the sink."""
    return [len(messages) for messages in sink.published_messages]


class TestJobOrchestratorInitialization:
    """Test JobOrchestrator initialization behavior."""

    def test_all_workflows_initialized_without_config_store(
        self,
        workflow_with_params: WorkflowSpec,
        workflow_no_params: WorkflowSpec,
    ):
        """All workflows in registry should be initialized even without config store."""
        workflow_id_1 = workflow_with_params.get_id()
        workflow_id_2 = workflow_no_params.get_id()
        registry = {
            workflow_id_1: workflow_with_params,
            workflow_id_2: workflow_no_params,
        }

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Both workflows should exist in _workflows
        staged_1 = orchestrator.get_staged_config(workflow_id_1)
        staged_2 = orchestrator.get_staged_config(workflow_id_2)

        assert isinstance(staged_1, dict)
        assert isinstance(staged_2, dict)

    def test_workflow_with_params_gets_default_config(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Workflow with params model should get default params for all sources."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        staged = orchestrator.get_staged_config(workflow_id)
        assert isinstance(staged, dict)
        assert set(staged.keys()) == {"det_1", "det_2"}

        # Should have default param values
        for job_config in staged.values():
            assert job_config.params["threshold"] == 100.0
            assert job_config.params["mode"] == "default"
            assert job_config.aux_source_names == {}

    def test_workflow_no_params_has_empty_staged_jobs(
        self,
        workflow_no_params: WorkflowSpec,
    ):
        """Workflow without params should exist but have empty staged_jobs."""
        workflow_id = workflow_no_params.get_id()
        registry = {workflow_id: workflow_no_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        staged = orchestrator.get_staged_config(workflow_id)
        assert isinstance(staged, dict)
        assert len(staged) == 0

    def test_workflow_empty_sources_has_empty_staged_jobs(
        self,
        workflow_empty_sources: WorkflowSpec,
    ):
        """Workflow with empty source_names should have empty staged_jobs."""
        workflow_id = workflow_empty_sources.get_id()
        registry = {workflow_id: workflow_empty_sources}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        staged = orchestrator.get_staged_config(workflow_id)
        assert isinstance(staged, dict)
        assert len(staged) == 0

    def test_loads_config_from_store(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Should load and use config from store when available."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        # Setup config store with persisted config (raw dict format)
        config_store = {
            str(workflow_id): {
                "jobs": {
                    "det_1": {
                        "params": {"threshold": 50.0, "mode": "custom"},
                        "aux_source_names": {},
                    }
                }
            }
        }

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=config_store,
        )

        staged = orchestrator.get_staged_config(workflow_id)
        assert isinstance(staged, dict)
        assert set(staged.keys()) == {"det_1"}  # Only the configured source

        # Should have loaded param values
        assert staged["det_1"].params["threshold"] == 50.0
        assert staged["det_1"].params["mode"] == "custom"

    def test_loads_invalid_config_without_validation(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Invalid configs are loaded as-is (validation happens later in UI)."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        # Setup config store with invalid param types
        config_store = {
            str(workflow_id): {
                "jobs": {
                    "det_1": {
                        "params": {"threshold": "not_a_float"},  # Invalid type
                        "aux_source_names": {},
                    }
                }
            }
        }

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=config_store,
        )

        staged = orchestrator.get_staged_config(workflow_id)
        assert isinstance(staged, dict)
        # Should use loaded config as-is (including invalid params)
        assert set(staged.keys()) == {"det_1"}
        # Invalid param value is loaded as-is
        assert staged["det_1"].params["threshold"] == "not_a_float"

    def test_loads_aux_sources_from_config(
        self,
        workflow_with_params_and_aux: WorkflowSpec,
    ):
        """Should load aux_source_names from config store."""
        workflow_id = workflow_with_params_and_aux.get_id()
        registry = {workflow_id: workflow_with_params_and_aux}

        config_store = {
            str(workflow_id): {
                "jobs": {
                    "det_1": {
                        "params": {"threshold": 75.0, "mode": "special"},
                        "aux_source_names": {"monitor": "monitor_2"},
                    }
                }
            }
        }

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=config_store,
        )

        staged = orchestrator.get_staged_config(workflow_id)
        assert staged["det_1"].params["threshold"] == 75.0
        assert staged["det_1"].aux_source_names
        assert staged["det_1"].aux_source_names["monitor"] == "monitor_2"

    def test_get_staged_config_never_raises_for_valid_workflow_id(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """get_staged_config should never raise KeyError for valid workflow_id."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Should not raise
        result = orchestrator.get_staged_config(workflow_id)
        assert isinstance(result, dict)

    def test_get_staged_config_raises_for_unknown_workflow_id(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """get_staged_config should raise KeyError for unknown workflow_id."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        unknown_id = WorkflowId(
            instrument="unknown", namespace="unknown", name="unknown", version=99
        )

        # Should raise KeyError for unknown workflow
        with pytest.raises(KeyError):
            orchestrator.get_staged_config(unknown_id)

    def test_get_staged_config_returns_dict(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """get_staged_config returns dict mapping source names to configs."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Should return dict with all sources
        configs = orchestrator.get_staged_config(workflow_id)
        assert isinstance(configs, dict)
        assert set(configs.keys()) == {"det_1", "det_2"}

        # Config params should be dicts
        assert configs["det_1"].params["threshold"] == 100.0
        assert configs["det_2"].params["threshold"] == 100.0

    def test_workflow_with_required_params_gets_empty_state(
        self,
        workflow_params_without_defaults: WorkflowSpec,
    ):
        """Workflow with params that can't be instantiated gets empty WorkflowState."""
        workflow_id = workflow_params_without_defaults.get_id()
        registry = {workflow_id: workflow_params_without_defaults}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Should exist in _workflows but have empty staged_jobs
        staged = orchestrator.get_staged_config(workflow_id)
        assert isinstance(staged, dict)
        assert len(staged) == 0

    def test_enum_params_serialized_to_yaml(
        self,
        workflow_with_enum_params: WorkflowSpec,
        tmp_path,
    ):
        """Enum values in params should be serialized as strings, not enum objects."""
        workflow_id = workflow_with_enum_params.get_id()
        registry = {workflow_id: workflow_with_enum_params}

        # Create a temporary file for config store
        config_file = tmp_path / "test_config.yaml"

        config_store = FileBackedConfigStore(file_path=config_file)

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=config_store,
        )

        # Commit workflow to trigger persistence
        orchestrator.commit_workflow(workflow_id)

        # Read the YAML file directly to verify enum serialization
        with open(config_file) as f:
            saved_data = yaml.safe_load(f)

        # Verify the config was saved
        workflow_key = str(workflow_id)
        assert workflow_key in saved_data

        # Verify enum was serialized as string value, not enum object
        # With per-source config, params are nested under jobs/source_name
        jobs = saved_data[workflow_key]['jobs']
        assert len(jobs) > 0
        first_source_config = next(iter(jobs.values()))
        saved_params = first_source_config['params']
        assert 'choice' in saved_params
        assert saved_params['choice'] == 'option_a'  # String value, not enum
        assert isinstance(saved_params['choice'], str)


class TestJobOrchestratorMutationSafety:
    """Test that JobOrchestrator protects against unintended mutations."""

    def test_stage_config_makes_defensive_copy_of_params(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Modifying params dict after staging should not affect staged config."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Stage config with mutable params dict
        params = {"threshold": 50.0, "mode": "custom"}
        aux_source_names = {}
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params=params,
            aux_source_names=aux_source_names,
        )

        # Modify the original params dict
        params["threshold"] = 999.0
        params["mode"] = "evil"

        # Staged config should NOT be affected
        staged = orchestrator.get_staged_config(workflow_id)
        assert staged["det_1"].params["threshold"] == 50.0
        assert staged["det_1"].params["mode"] == "custom"

    def test_stage_config_makes_defensive_copy_of_aux_source_names(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Modifying aux_source_names after staging should not affect staged config."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Stage config with mutable aux_source_names dict
        params = {"threshold": 50.0}
        aux_source_names = {"monitor": "monitor_1"}
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params=params,
            aux_source_names=aux_source_names,
        )

        # Modify the original aux_source_names dict
        aux_source_names["monitor"] = "monitor_evil"
        aux_source_names["extra"] = "unexpected"

        # Staged config should NOT be affected
        staged = orchestrator.get_staged_config(workflow_id)
        assert staged["det_1"].aux_source_names == {"monitor": "monitor_1"}

    def test_get_staged_config_returns_independent_copy(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Modifying returned config should not affect internal state."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={"monitor": "monitor_1"},
        )

        # Get staged config and modify it
        staged = orchestrator.get_staged_config(workflow_id)
        staged["det_1"].params["threshold"] = 999.0
        staged["det_1"].aux_source_names["monitor"] = "monitor_evil"

        # Get again - should be unchanged
        staged_again = orchestrator.get_staged_config(workflow_id)
        assert staged_again["det_1"].params["threshold"] == 50.0
        assert staged_again["det_1"].aux_source_names["monitor"] == "monitor_1"

    def test_get_active_config_returns_independent_copy(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Modifying returned active config should not affect internal state."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )
        orchestrator.commit_workflow(workflow_id)

        # Get active config and modify it
        active = orchestrator.get_active_config(workflow_id)
        active["det_1"].params["threshold"] = 999.0

        # Get again - should be unchanged
        active_again = orchestrator.get_active_config(workflow_id)
        assert active_again["det_1"].params["threshold"] == 50.0

    def test_load_from_store_creates_independent_job_configs(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Each JobConfig should have independent params/aux_source_names dicts."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        # Config store with multiple sources (each with their own params)
        config_store = {
            str(workflow_id): {
                "jobs": {
                    "det_1": {
                        "params": {"threshold": 50.0, "mode": "custom"},
                        "aux_source_names": {"monitor": "monitor_1"},
                    },
                    "det_2": {
                        "params": {"threshold": 50.0, "mode": "custom"},
                        "aux_source_names": {"monitor": "monitor_1"},
                    },
                }
            }
        }

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=config_store,
        )

        # Modify params for one source
        staged = orchestrator.get_staged_config(workflow_id)
        staged["det_1"].params["threshold"] = 999.0
        staged["det_1"].aux_source_names["monitor"] = "monitor_evil"

        # Get again and check det_2 is not affected
        staged_again = orchestrator.get_staged_config(workflow_id)
        assert staged_again["det_2"].params["threshold"] == 50.0
        assert staged_again["det_2"].aux_source_names["monitor"] == "monitor_1"

    def test_committed_config_not_affected_by_new_staging(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """After commit, modifying staged config should not affect active config."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Stage and commit initial config
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )
        orchestrator.commit_workflow(workflow_id)

        # Stage new config with different params
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 999.0, "mode": "evil"},
            aux_source_names={},
        )

        # Active config should still have original values
        active = orchestrator.get_active_config(workflow_id)
        assert active["det_1"].params["threshold"] == 50.0
        assert active["det_1"].params["mode"] == "custom"

        # Staged config should have new values
        staged = orchestrator.get_staged_config(workflow_id)
        assert staged["det_1"].params["threshold"] == 999.0
        assert staged["det_1"].params["mode"] == "evil"


class TestJobOrchestratorCommit:
    """Test commit_workflow and related job lifecycle operations."""

    def test_commit_workflow_returns_job_ids_for_staged_sources(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify commit_workflow returns correct JobIds for all staged sources."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Stage configs for 2 sources
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )
        orchestrator.stage_config(
            workflow_id,
            source_name="det_2",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )

        # Commit
        job_ids = orchestrator.commit_workflow(workflow_id)

        # Assert: returns list[JobId] with correct source_names and shared job_number
        assert len(job_ids) == 2
        assert {job_id.source_name for job_id in job_ids} == {"det_1", "det_2"}
        # All jobs should share the same job_number
        assert job_ids[0].job_number == job_ids[1].job_number

    def test_commit_workflow_raises_if_nothing_staged(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify commit_workflow raises ValueError if no configs staged."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Clear staged jobs (workflow initialized with defaults from spec)
        orchestrator.clear_staged_configs(workflow_id)

        # commit_workflow() should raise ValueError
        with pytest.raises(ValueError, match="No staged configs"):
            orchestrator.commit_workflow(workflow_id)

    def test_commit_workflow_sends_workflow_configs_to_backend(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify correct WorkflowConfig messages sent for each source."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        fake_sink = FakeMessageSink()
        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=fake_sink),
            workflow_registry=registry,
            config_store=None,
        )

        # Stage configs
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 75.0, "mode": "accurate"},
            aux_source_names={},
        )
        orchestrator.stage_config(
            workflow_id,
            source_name="det_2",
            params={"threshold": 75.0, "mode": "accurate"},
            aux_source_names={},
        )

        # Commit
        job_ids = orchestrator.commit_workflow(workflow_id)

        # Assert: FakeMessageSink received correct WorkflowConfig for each source
        sent_configs = get_sent_workflow_configs(fake_sink)
        assert len(sent_configs) == 2

        # Check each source received a config
        source_names = {config[0] for config in sent_configs}
        assert source_names == {"det_1", "det_2"}

        # Verify config contents
        for source_name, config in sent_configs:
            assert isinstance(config, WorkflowConfig)
            assert config.identifier == workflow_id
            assert config.params["threshold"] == 75.0
            assert config.params["mode"] == "accurate"
            # Job number should match the returned JobIds
            matching_job = next(
                job for job in job_ids if job.source_name == source_name
            )
            assert config.job_number == matching_job.job_number

    def test_commit_workflow_stops_previous_jobs_on_second_commit(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify second commit sends stop commands for previous jobs."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        fake_sink = FakeMessageSink()
        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=fake_sink),
            workflow_registry=registry,
            config_store=None,
        )

        # Commit first workflow
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "first"},
            aux_source_names={},
        )
        orchestrator.stage_config(
            workflow_id,
            source_name="det_2",
            params={"threshold": 50.0, "mode": "first"},
            aux_source_names={},
        )
        first_job_ids = orchestrator.commit_workflow(workflow_id)

        # Clear sink to track second commit
        fake_sink.published_messages.clear()

        # Commit second workflow
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 100.0, "mode": "second"},
            aux_source_names={},
        )
        orchestrator.stage_config(
            workflow_id,
            source_name="det_2",
            params={"threshold": 100.0, "mode": "second"},
            aux_source_names={},
        )
        second_job_ids = orchestrator.commit_workflow(workflow_id)

        # Assert: stop commands sent for first workflow's jobs
        sent_commands = get_sent_commands(fake_sink)
        stop_commands = [
            (key, value)
            for key, value in sent_commands
            if isinstance(value, JobCommand) and value.action == JobAction.stop
        ]

        assert len(stop_commands) == 2
        stopped_job_ids = {cmd[1].job_id for cmd in stop_commands}
        assert stopped_job_ids == set(first_job_ids)

        # Verify new workflow configs were sent
        sent_configs = get_sent_workflow_configs(fake_sink)
        assert len(sent_configs) == 2

        # Verify the new configs have different job_number than old ones
        new_job_number = sent_configs[0][1].job_number
        assert new_job_number != first_job_ids[0].job_number
        assert new_job_number == second_job_ids[0].job_number

    def test_clear_staged_configs_then_commit_raises_error(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify clear + commit interaction raises appropriate error."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Stage some configs
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )

        # Clear staged configs
        orchestrator.clear_staged_configs(workflow_id)

        # Verify configs were cleared
        staged = orchestrator.get_staged_config(workflow_id)
        assert len(staged) == 0

        # Commit after clear should raise ValueError
        with pytest.raises(ValueError, match="No staged configs"):
            orchestrator.commit_workflow(workflow_id)

    def test_persist_config_is_noop_when_config_store_is_none(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Config persistence should silently no-op when config_store is None."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,  # No config store
        )

        # Clear defaults and stage only one source
        orchestrator.clear_staged_configs(workflow_id)
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )

        # Should complete without errors (no-op persistence)
        job_ids = orchestrator.commit_workflow(workflow_id)
        assert len(job_ids) == 1

    def test_persist_config_is_noop_when_staged_jobs_empty(
        self,
        workflow_no_params: WorkflowSpec,
        workflow_with_params: WorkflowSpec,
    ):
        """Config persistence should silently no-op when staged_jobs is empty."""
        workflow_id_no_params = workflow_no_params.get_id()
        workflow_id_with_params = workflow_with_params.get_id()
        registry = {
            workflow_id_no_params: workflow_no_params,
            workflow_id_with_params: workflow_with_params,
        }

        # Use a dict config store to verify it's not modified
        config_store: dict[str, dict] = {}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=config_store,
        )

        # Stage and commit workflow_with_params (should persist)
        orchestrator.stage_config(
            workflow_id_with_params,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )
        orchestrator.commit_workflow(workflow_id_with_params)

        # Config store should have workflow_with_params
        assert str(workflow_id_with_params) in config_store

        # Now try to commit workflow_no_params (has empty staged_jobs)
        # First stage something so we can commit
        orchestrator.stage_config(
            workflow_id_with_params,
            source_name="det_1",
            params={"threshold": 100.0, "mode": "updated"},
            aux_source_names={},
        )

        # Record current config store state
        config_store_before = dict(config_store)

        # Workflow with no params should not add to config store
        # (it has empty staged_jobs from initialization)
        # We can't commit it directly since it has no staged configs,
        # but we can verify the initialization didn't persist it
        assert str(workflow_id_no_params) not in config_store

        # Config store should be unchanged for workflows with empty staged_jobs
        assert config_store == config_store_before


class TestJobOrchestratorStop:
    """Test stop_workflow functionality."""

    def test_stop_workflow_sends_stop_commands_to_backend(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify stop_workflow sends stop commands for all active jobs."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        fake_sink = FakeMessageSink()
        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=fake_sink),
            workflow_registry=registry,
            config_store=None,
        )

        # Start a workflow
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )
        orchestrator.stage_config(
            workflow_id,
            source_name="det_2",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )
        job_ids = orchestrator.commit_workflow(workflow_id)

        # Clear sink to track stop commands
        fake_sink.published_messages.clear()

        # Stop the workflow
        result = orchestrator.stop_workflow(workflow_id)

        # Assert: stop commands sent for all jobs
        assert result is True
        sent_commands = get_sent_commands(fake_sink)
        stop_commands = [
            (key, value)
            for key, value in sent_commands
            if isinstance(value, JobCommand) and value.action == JobAction.stop
        ]

        assert len(stop_commands) == 2
        stopped_job_ids = {cmd[1].job_id for cmd in stop_commands}
        assert stopped_job_ids == set(job_ids)

    def test_stop_workflow_clears_active_job_state(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify stop_workflow clears current job state."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Start a workflow
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )
        orchestrator.commit_workflow(workflow_id)

        # Verify job is active
        assert orchestrator.get_active_job_number(workflow_id) is not None

        # Stop the workflow
        orchestrator.stop_workflow(workflow_id)

        # Verify job is no longer active
        assert orchestrator.get_active_job_number(workflow_id) is None

    def test_stop_workflow_returns_false_when_no_active_jobs(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify stop_workflow returns False when no active jobs."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Don't start any workflow - just try to stop
        result = orchestrator.stop_workflow(workflow_id)

        # Should return False
        assert result is False

    def test_stop_workflow_preserves_staged_config(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify stop_workflow preserves staged configuration for restart."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Clear defaults and stage only det_1
        orchestrator.clear_staged_configs(workflow_id)
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={"monitor": "monitor_1"},
        )
        orchestrator.commit_workflow(workflow_id)

        # Stop the workflow
        orchestrator.stop_workflow(workflow_id)

        # Verify staged config still exists
        staged = orchestrator.get_staged_config(workflow_id)
        assert len(staged) == 1
        assert staged["det_1"].params["threshold"] == 50.0
        assert staged["det_1"].params["mode"] == "custom"
        assert staged["det_1"].aux_source_names["monitor"] == "monitor_1"

    def test_stop_workflow_persists_stopped_state(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify stop_workflow persists stopped state to config store."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        config_store: dict[str, dict] = {}
        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=config_store,
        )

        # Start a workflow
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )
        job_ids = orchestrator.commit_workflow(workflow_id)

        # Verify active job is in config store
        assert 'current_job' in config_store[str(workflow_id)]

        # Stop the workflow
        orchestrator.stop_workflow(workflow_id)

        # Verify current_job is removed from config store
        assert 'current_job' not in config_store[str(workflow_id)]

        # Verify previous_job is in config store
        assert 'previous_job' in config_store[str(workflow_id)]
        assert config_store[str(workflow_id)]['previous_job']['job_number'] == str(
            job_ids[0].job_number
        )

    def test_stop_workflow_allows_immediate_restart(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify workflow can be restarted immediately after stopping."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Clear defaults and stage only det_1
        orchestrator.clear_staged_configs(workflow_id)
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0, "mode": "custom"},
            aux_source_names={},
        )
        first_job_ids = orchestrator.commit_workflow(workflow_id)

        # Stop the workflow
        orchestrator.stop_workflow(workflow_id)

        # Restart with same config
        second_job_ids = orchestrator.commit_workflow(workflow_id)

        # Should succeed and get new job number
        assert len(second_job_ids) == 1
        assert second_job_ids[0].job_number != first_job_ids[0].job_number


class TestJobOrchestratorWorkflowStateVersion:
    """Test version counter mechanism for cross-session change detection."""

    def test_initial_version_is_zero(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Workflow state version starts at zero after initialization."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        assert orchestrator.get_workflow_state_version(workflow_id) == 0

    def test_unknown_workflow_returns_zero(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """get_workflow_state_version returns 0 for unknown workflow IDs."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        unknown_id = WorkflowId(
            instrument="unknown", namespace="unknown", name="unknown", version=99
        )
        assert orchestrator.get_workflow_state_version(unknown_id) == 0

    def test_stage_config_increments_version(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """stage_config increments the workflow state version."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        version_before = orchestrator.get_workflow_state_version(workflow_id)

        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 50.0},
            aux_source_names={},
        )

        assert (
            orchestrator.get_workflow_state_version(workflow_id) == version_before + 1
        )

    def test_clear_staged_configs_increments_version(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """clear_staged_configs increments the workflow state version."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        version_before = orchestrator.get_workflow_state_version(workflow_id)

        orchestrator.clear_staged_configs(workflow_id)

        assert (
            orchestrator.get_workflow_state_version(workflow_id) == version_before + 1
        )

    def test_commit_workflow_increments_version(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """commit_workflow increments the workflow state version."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        version_before = orchestrator.get_workflow_state_version(workflow_id)

        orchestrator.commit_workflow(workflow_id)

        assert orchestrator.get_workflow_state_version(workflow_id) > version_before

    def test_stop_workflow_increments_version(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """stop_workflow increments the workflow state version."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # First commit to have something to stop
        orchestrator.commit_workflow(workflow_id)
        version_before = orchestrator.get_workflow_state_version(workflow_id)

        orchestrator.stop_workflow(workflow_id)

        assert (
            orchestrator.get_workflow_state_version(workflow_id) == version_before + 1
        )

    def test_stop_workflow_does_not_increment_version_if_nothing_to_stop(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """stop_workflow does not increment version when there are no active jobs."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        version_before = orchestrator.get_workflow_state_version(workflow_id)

        # Try to stop without committing first
        orchestrator.stop_workflow(workflow_id)

        assert orchestrator.get_workflow_state_version(workflow_id) == version_before

    def test_staging_transaction_replaces_all_configs(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """staging_transaction allows replacing all configs in one operation."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Stage some initial configs
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 100.0},
            aux_source_names={},
        )

        # Replace with new configs using transaction
        with orchestrator.staging_transaction(workflow_id):
            orchestrator.clear_staged_configs(workflow_id)
            orchestrator.stage_config(
                workflow_id,
                source_name="det_2",
                params={"threshold": 200.0},
                aux_source_names={},
            )
            orchestrator.stage_config(
                workflow_id,
                source_name="det_3",
                params={"threshold": 300.0},
                aux_source_names={},
            )

        # Old config should be gone, new configs present
        staged = orchestrator.get_staged_config(workflow_id)
        assert "det_1" not in staged
        assert "det_2" in staged
        assert "det_3" in staged
        assert staged["det_2"].params == {"threshold": 200.0}
        assert staged["det_3"].params == {"threshold": 300.0}

    def test_staging_transaction_increments_version_exactly_once(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """staging_transaction increments version by 1 for batched operations."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        version_before = orchestrator.get_workflow_state_version(workflow_id)

        with orchestrator.staging_transaction(workflow_id):
            orchestrator.clear_staged_configs(workflow_id)
            orchestrator.stage_config(
                workflow_id,
                source_name="det_1",
                params={"threshold": 100.0},
                aux_source_names={},
            )
            orchestrator.stage_config(
                workflow_id,
                source_name="det_2",
                params={"threshold": 200.0},
                aux_source_names={},
            )
            orchestrator.stage_config(
                workflow_id,
                source_name="det_3",
                params={"threshold": 300.0},
                aux_source_names={},
            )

        # Exactly one version increment despite 4 operations (clear + 3 stage_config)
        assert (
            orchestrator.get_workflow_state_version(workflow_id) == version_before + 1
        )

    def test_nested_staging_transaction_increments_version_once(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Nested staging transactions increment version by exactly 1."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        version_before = orchestrator.get_workflow_state_version(workflow_id)

        with orchestrator.staging_transaction(workflow_id):
            orchestrator.stage_config(
                workflow_id, source_name="det_1", params={}, aux_source_names={}
            )
            # Inner transaction
            with orchestrator.staging_transaction(workflow_id):
                orchestrator.stage_config(
                    workflow_id, source_name="det_2", params={}, aux_source_names={}
                )
            # More staging after inner transaction
            orchestrator.stage_config(
                workflow_id, source_name="det_3", params={}, aux_source_names={}
            )

        # Only one version increment after outer transaction completes
        assert (
            orchestrator.get_workflow_state_version(workflow_id) == version_before + 1
        )

        # All configs should be present
        staged = orchestrator.get_staged_config(workflow_id)
        assert "det_1" in staged
        assert "det_2" in staged
        assert "det_3" in staged

    def test_staging_transaction_rejects_different_workflow(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """staging_transaction raises error if nesting different workflows."""
        workflow_id_1 = workflow_with_params.get_id()

        # Create a second workflow spec with different ID
        from ess.livedata.config.workflow_spec import WorkflowSpec

        workflow_2 = WorkflowSpec(
            instrument="test",
            namespace="testing",
            name="different_workflow",
            version=1,
            title="Different Workflow",
            description="A different workflow for testing",
            stream_kind="detector",
            params=workflow_with_params.params,
        )
        workflow_id_2 = workflow_2.get_id()

        registry = {workflow_id_1: workflow_with_params, workflow_id_2: workflow_2}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Should raise error when trying to nest different workflows
        with orchestrator.staging_transaction(workflow_id_1):
            with pytest.raises(ValueError, match="Cannot nest transactions"):
                with orchestrator.staging_transaction(workflow_id_2):
                    pass


class TestWorkflowAdapterIndependentSourceConfiguration:
    """Test that workflow adapter preserves independently configured sources."""

    def test_adapter_start_action_preserves_other_sources(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Configuring one source should not affect other sources' configs."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Stage initial config for det_1 with specific params
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 100.0, "mode": "fast"},
            aux_source_names={},
        )

        # Create adapter scoped to det_2 and configure it
        adapter = orchestrator.create_workflow_adapter(
            workflow_id, selected_sources=["det_2"]
        )

        # Simulate user clicking "Apply" with different params for det_2
        adapter.start_action(
            selected_sources=["det_2"],
            parameter_values=WorkflowParams(threshold=200.0, mode="slow"),
        )

        # Verify det_1's config is preserved
        staged = orchestrator.get_staged_config(workflow_id)
        assert "det_1" in staged
        assert staged["det_1"].params["threshold"] == 100.0
        assert staged["det_1"].params["mode"] == "fast"

        # Verify det_2 has new config
        assert "det_2" in staged
        assert staged["det_2"].params["threshold"] == 200.0
        assert staged["det_2"].params["mode"] == "slow"

    def test_adapter_start_action_updates_existing_source(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Updating an already-configured source should update its config."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Stage initial config for det_1
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 100.0, "mode": "fast"},
            aux_source_names={},
        )

        # Create adapter scoped to det_1 and update it
        adapter = orchestrator.create_workflow_adapter(
            workflow_id, selected_sources=["det_1"]
        )

        adapter.start_action(
            selected_sources=["det_1"],
            parameter_values=WorkflowParams(threshold=999.0, mode="updated"),
        )

        # Verify det_1's config is updated
        staged = orchestrator.get_staged_config(workflow_id)
        assert staged["det_1"].params["threshold"] == 999.0
        assert staged["det_1"].params["mode"] == "updated"


class TestWorkflowSubscriptions:
    """Test subscribe_to_workflow with WorkflowCallbacks."""

    def test_on_started_called_when_workflow_commits(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """on_started callback is called when workflow is committed."""
        from ess.livedata.dashboard.job_orchestrator import WorkflowCallbacks

        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        started_job_numbers = []
        callbacks = WorkflowCallbacks(
            on_started=lambda job_number: started_job_numbers.append(job_number)
        )
        orchestrator.subscribe_to_workflow(workflow_id, callbacks)

        job_ids = orchestrator.commit_workflow(workflow_id)

        assert len(started_job_numbers) == 1
        assert started_job_numbers[0] == job_ids[0].job_number

    def test_on_stopped_called_when_workflow_stops(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """on_stopped callback is called when workflow is stopped."""
        from ess.livedata.dashboard.job_orchestrator import WorkflowCallbacks

        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        started_job_numbers = []
        stopped_job_numbers = []
        callbacks = WorkflowCallbacks(
            on_started=lambda job_number: started_job_numbers.append(job_number),
            on_stopped=lambda job_number: stopped_job_numbers.append(job_number),
        )
        orchestrator.subscribe_to_workflow(workflow_id, callbacks)

        job_ids = orchestrator.commit_workflow(workflow_id)
        assert len(stopped_job_numbers) == 0

        orchestrator.stop_workflow(workflow_id)

        assert len(stopped_job_numbers) == 1
        assert stopped_job_numbers[0] == job_ids[0].job_number

    def test_on_stopped_not_called_when_no_callback_provided(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """on_stopped None does not cause error when workflow stops."""
        from ess.livedata.dashboard.job_orchestrator import WorkflowCallbacks

        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        started_job_numbers = []
        callbacks = WorkflowCallbacks(
            on_started=lambda job_number: started_job_numbers.append(job_number),
            on_stopped=None,  # Explicitly None
        )
        orchestrator.subscribe_to_workflow(workflow_id, callbacks)

        orchestrator.commit_workflow(workflow_id)
        # Should not raise
        orchestrator.stop_workflow(workflow_id)

    def test_on_stopped_not_called_when_nothing_to_stop(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """on_stopped is not called when there's no active job to stop."""
        from ess.livedata.dashboard.job_orchestrator import WorkflowCallbacks

        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        stopped_job_numbers = []
        callbacks = WorkflowCallbacks(
            on_started=lambda _: None,
            on_stopped=lambda job_number: stopped_job_numbers.append(job_number),
        )
        orchestrator.subscribe_to_workflow(workflow_id, callbacks)

        # Try to stop without committing first
        orchestrator.stop_workflow(workflow_id)

        assert len(stopped_job_numbers) == 0

    def test_on_started_called_immediately_if_workflow_already_running(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """on_started is called immediately if workflow is already running."""
        from ess.livedata.dashboard.job_orchestrator import WorkflowCallbacks

        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_registry=registry,
            config_store=None,
        )

        # Commit workflow first
        job_ids = orchestrator.commit_workflow(workflow_id)

        # Then subscribe
        started_job_numbers = []
        callbacks = WorkflowCallbacks(
            on_started=lambda job_number: started_job_numbers.append(job_number)
        )
        _, was_invoked = orchestrator.subscribe_to_workflow(workflow_id, callbacks)

        assert was_invoked is True
        assert len(started_job_numbers) == 1
        assert started_job_numbers[0] == job_ids[0].job_number
