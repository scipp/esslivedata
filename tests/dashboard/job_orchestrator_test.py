# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for JobOrchestrator initialization and config management."""

import time
from collections.abc import Callable
from enum import StrEnum

import pydantic
import pytest
import scipp as sc
import yaml

from ess.livedata.config.workflow_spec import (
    REDUCTION,
    AuxSources,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.core.desired_state import (
    DesiredStateWriter,
    DesiredTable,
    InMemoryDesiredStateLog,
    fold_updates,
)
from ess.livedata.dashboard.active_job_registry import ActiveJobRegistry
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.config_store import FileBackedConfigStore
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_orchestrator import (
    PENDING_COMMAND_TIMEOUT_SECONDS,
    JobOrchestrator,
)
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.notification_queue import (
    NotificationQueue,
    NotificationType,
)
from ess.livedata.fakes import FakeMessageSink


class WorkflowParams(pydantic.BaseModel):
    """Test params model with defaults."""

    threshold: float = 100.0
    mode: str = "default"


test_aux_sources = AuxSources({'monitor': 'monitor_1'})


class ParamsWithRequiredFields(pydantic.BaseModel):
    """Params model with required fields (no defaults) - like correlation histograms."""

    required_value: float  # No default!


class SampleEnum(StrEnum):
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
        name="with_params",
        version=1,
        title="Workflow With Params",
        description="Test workflow with params",
        source_names=["det_1", "det_2"],
        params=WorkflowParams,
        outputs=SimpleTestOutputs,
        group=REDUCTION,
    )


@pytest.fixture
def workflow_with_params_and_aux() -> WorkflowSpec:
    """Workflow spec with both params and aux sources."""
    return WorkflowSpec(
        instrument="test",
        name="with_params_and_aux",
        version=1,
        title="Workflow With Params and Aux",
        description="Test workflow with params and aux sources",
        source_names=["det_1"],
        params=WorkflowParams,
        aux_sources=test_aux_sources,
        outputs=SimpleTestOutputs,
        group=REDUCTION,
    )


@pytest.fixture
def workflow_no_params() -> WorkflowSpec:
    """Workflow spec without params model."""
    return WorkflowSpec(
        instrument="test",
        name="no_params",
        version=1,
        title="Workflow Without Params",
        description="Test workflow without params",
        source_names=["det_1", "det_2"],
        outputs=SimpleTestOutputs,
        group=REDUCTION,
    )


@pytest.fixture
def workflow_empty_sources() -> WorkflowSpec:
    """Workflow spec with empty source_names."""
    return WorkflowSpec(
        instrument="test",
        name="empty_sources",
        version=1,
        title="Workflow With Empty Sources",
        description="Test workflow with no sources",
        source_names=[],
        params=WorkflowParams,
        outputs=SimpleTestOutputs,
        group=REDUCTION,
    )


@pytest.fixture
def workflow_params_without_defaults() -> WorkflowSpec:
    """Workflow spec with params that can't be instantiated (required fields)."""
    return WorkflowSpec(
        instrument="test",
        name="params_without_defaults",
        version=1,
        title="Workflow With Required Params",
        description="Test workflow like correlation histograms",
        source_names=["det_1"],
        params=ParamsWithRequiredFields,
        outputs=SimpleTestOutputs,
        group=REDUCTION,
    )


@pytest.fixture
def workflow_with_enum_params() -> WorkflowSpec:
    """Workflow spec with params containing enum fields."""
    return WorkflowSpec(
        instrument="test",
        name="with_enum_params",
        version=1,
        title="Workflow With Enum Params",
        description="Test workflow with enum params for serialization testing",
        source_names=["det_1"],
        params=ParamsWithEnum,
        outputs=SimpleTestOutputs,
        group=REDUCTION,
    )


def desired_table(log: InMemoryDesiredStateLog) -> DesiredTable:
    """The desired table the log holds."""
    return fold_updates({}, log.read(0))


def make_orchestrator(
    *specs: WorkflowSpec,
    config_store=None,
    fake_sink: FakeMessageSink | None = None,
    data_service: DataService | None = None,
    job_service: JobService | None = None,
    notification_queue: NotificationQueue | None = None,
    clock: Callable[[], float] = time.time,
    desired_state: DesiredStateWriter | None = None,
) -> JobOrchestrator:
    """Build a JobOrchestrator from one or more WorkflowSpecs for testing."""
    registry = {spec.get_id(): spec for spec in specs}
    ds = data_service if data_service is not None else DataService()
    js = job_service if job_service is not None else JobService()
    return JobOrchestrator(
        desired_state=desired_state or InMemoryDesiredStateLog(),
        command_service=CommandService(
            sink=fake_sink if fake_sink is not None else FakeMessageSink()
        ),
        workflow_registry=registry,
        active_job_registry=ActiveJobRegistry(data_service=ds, job_service=js),
        config_store=config_store,
        notification_queue=notification_queue,
        clock=clock,
    )


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
        orchestrator = make_orchestrator(workflow_with_params, workflow_no_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_no_params)

        staged = orchestrator.get_staged_config(workflow_id)
        assert isinstance(staged, dict)
        assert len(staged) == 0

    def test_workflow_empty_sources_has_empty_staged_jobs(
        self,
        workflow_empty_sources: WorkflowSpec,
    ):
        """Workflow with empty source_names should have empty staged_jobs."""
        workflow_id = workflow_empty_sources.get_id()
        orchestrator = make_orchestrator(workflow_empty_sources)

        staged = orchestrator.get_staged_config(workflow_id)
        assert isinstance(staged, dict)
        assert len(staged) == 0

    def test_loads_config_from_store(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Should load and use config from store when available."""
        workflow_id = workflow_with_params.get_id()

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
        orchestrator = make_orchestrator(
            workflow_with_params, config_store=config_store
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
        orchestrator = make_orchestrator(
            workflow_with_params, config_store=config_store
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
        orchestrator = make_orchestrator(
            workflow_with_params_and_aux, config_store=config_store
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
        orchestrator = make_orchestrator(workflow_with_params)

        # Should not raise
        result = orchestrator.get_staged_config(workflow_id)
        assert isinstance(result, dict)

    def test_get_staged_config_raises_for_unknown_workflow_id(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """get_staged_config should raise KeyError for unknown workflow_id."""
        orchestrator = make_orchestrator(workflow_with_params)

        unknown_id = WorkflowId(instrument="unknown", name="unknown", version=99)

        # Should raise KeyError for unknown workflow
        with pytest.raises(KeyError):
            orchestrator.get_staged_config(unknown_id)

    def test_get_staged_config_returns_dict(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """get_staged_config returns dict mapping source names to configs."""
        workflow_id = workflow_with_params.get_id()
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_params_without_defaults)

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

        # Create a temporary file for config store
        config_file = tmp_path / "test_config.yaml"
        config_store = FileBackedConfigStore(file_path=config_file)
        orchestrator = make_orchestrator(
            workflow_with_enum_params, config_store=config_store
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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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

        orchestrator = make_orchestrator(
            workflow_with_params, config_store=config_store
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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

        # Clear staged jobs (workflow initialized with defaults from spec)
        orchestrator.clear_staged_configs(workflow_id)

        # commit_workflow() should raise ValueError
        with pytest.raises(ValueError, match="No staged configs"):
            orchestrator.commit_workflow(workflow_id)

    def test_commit_workflow_writes_staged_configs_as_one_record(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        workflow_id = workflow_with_params.get_id()
        log = InMemoryDesiredStateLog()
        orchestrator = make_orchestrator(workflow_with_params, desired_state=log)
        for source_name in ("det_1", "det_2"):
            orchestrator.stage_config(
                workflow_id,
                source_name=source_name,
                params={"threshold": 75.0, "mode": "accurate"},
                aux_source_names={},
            )

        job_ids = orchestrator.commit_workflow(workflow_id)

        ((written_id, desired),) = log.read(0)
        assert written_id == workflow_id
        assert desired.job_number == job_ids[0].job_number
        assert set(desired.jobs) == {"det_1", "det_2"}
        for job in desired.jobs.values():
            assert job.params == {"threshold": 75.0, "mode": "accurate"}

    def test_second_commit_replaces_the_record(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Workers stop the superseded jobs; the dashboard sends no stops."""
        workflow_id = workflow_with_params.get_id()
        fake_sink = FakeMessageSink()
        log = InMemoryDesiredStateLog()
        orchestrator = make_orchestrator(
            workflow_with_params, fake_sink=fake_sink, desired_state=log
        )
        first_job_ids = orchestrator.commit_workflow(workflow_id)
        orchestrator.stage_config(
            workflow_id,
            source_name="det_1",
            params={"threshold": 100.0, "mode": "second"},
            aux_source_names={},
        )

        second_job_ids = orchestrator.commit_workflow(workflow_id)

        desired = desired_table(log)[workflow_id]
        assert desired.job_number == second_job_ids[0].job_number
        assert desired.job_number != first_job_ids[0].job_number
        assert desired.jobs["det_1"].params["mode"] == "second"
        assert fake_sink.published_messages == []

    def test_clear_staged_configs_then_commit_raises_error(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify clear + commit interaction raises appropriate error."""
        workflow_id = workflow_with_params.get_id()
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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

        # Use a dict config store to verify it's not modified
        config_store: dict[str, dict] = {}
        orchestrator = make_orchestrator(
            workflow_no_params, workflow_with_params, config_store=config_store
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

    def test_stop_workflow_writes_tombstone(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        workflow_id = workflow_with_params.get_id()
        log = InMemoryDesiredStateLog()
        orchestrator = make_orchestrator(workflow_with_params, desired_state=log)
        orchestrator.commit_workflow(workflow_id)

        assert orchestrator.stop_workflow(workflow_id) is True

        assert log.read(1) == [(workflow_id, None)]
        assert desired_table(log) == {}

    def test_stop_workflow_clears_active_job_state(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify stop_workflow clears current job state."""
        workflow_id = workflow_with_params.get_id()
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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

    def test_stop_workflow_allows_immediate_restart(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """Verify workflow can be restarted immediately after stopping."""
        workflow_id = workflow_with_params.get_id()
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

        assert orchestrator.get_workflow_state_version(workflow_id) == 0

    def test_unknown_workflow_returns_zero(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """get_workflow_state_version returns 0 for unknown workflow IDs."""
        orchestrator = make_orchestrator(workflow_with_params)

        unknown_id = WorkflowId(instrument="unknown", name="unknown", version=99)
        assert orchestrator.get_workflow_state_version(unknown_id) == 0

    def test_stage_config_increments_version(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """stage_config increments the workflow state version."""
        workflow_id = workflow_with_params.get_id()
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

        version_before = orchestrator.get_workflow_state_version(workflow_id)

        orchestrator.commit_workflow(workflow_id)

        assert orchestrator.get_workflow_state_version(workflow_id) > version_before

    def test_stop_workflow_increments_version(
        self,
        workflow_with_params: WorkflowSpec,
    ):
        """stop_workflow increments the workflow state version."""
        workflow_id = workflow_with_params.get_id()
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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
            name="different_workflow",
            version=1,
            title="Different Workflow",
            description="A different workflow for testing",
            stream_kind="detector",
            params=workflow_with_params.params,
            group=REDUCTION,
        )
        workflow_id_2 = workflow_2.get_id()

        orchestrator = make_orchestrator(workflow_with_params, workflow_2)

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
        orchestrator = make_orchestrator(workflow_with_params)

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
        orchestrator = make_orchestrator(workflow_with_params)

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


class TestGetRunningWorkflowSources:
    """Tests for the running-sources accessor used by device derivation."""

    def test_empty_when_no_jobs_committed(self, workflow_with_params: WorkflowSpec):
        orchestrator = make_orchestrator(workflow_with_params)
        assert orchestrator.get_running_workflow_sources() == {}

    def test_reports_committed_sources(self, workflow_with_params: WorkflowSpec):
        workflow_id = workflow_with_params.get_id()
        orchestrator = make_orchestrator(workflow_with_params)

        orchestrator.clear_staged_configs(workflow_id)
        orchestrator.stage_config(
            workflow_id, source_name="det_1", params={}, aux_source_names={}
        )
        orchestrator.commit_workflow(workflow_id)

        assert orchestrator.get_running_workflow_sources() == {workflow_id: {"det_1"}}

    def test_omits_workflow_after_stop(self, workflow_with_params: WorkflowSpec):
        workflow_id = workflow_with_params.get_id()
        orchestrator = make_orchestrator(workflow_with_params)

        orchestrator.commit_workflow(workflow_id)
        assert workflow_id in orchestrator.get_running_workflow_sources()

        orchestrator.stop_workflow(workflow_id)
        assert workflow_id not in orchestrator.get_running_workflow_sources()


class MutableClock:
    """Manually advanceable clock for deterministic expiry tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestPendingCommandExpiry:
    """Test expiry of resets whose acknowledgement never arrives."""

    def test_unacknowledged_reset_produces_error_notification_after_timeout(
        self, workflow_with_params: WorkflowSpec
    ):
        workflow_id = workflow_with_params.get_id()
        clock = MutableClock()
        queue = NotificationQueue()
        orchestrator = make_orchestrator(
            workflow_with_params, notification_queue=queue, clock=clock
        )

        orchestrator.stage_config(
            workflow_id, source_name="det_1", params={}, aux_source_names={}
        )
        orchestrator.commit_workflow(workflow_id)
        orchestrator.reset_workflow(workflow_id)

        clock.advance(PENDING_COMMAND_TIMEOUT_SECONDS + 1.0)
        orchestrator.expire_pending_commands()

        errors = [
            event
            for event in queue.get_all_events()
            if event.notification_type is NotificationType.ERROR
        ]
        assert len(errors) == 1
        assert workflow_with_params.title in errors[0].message

    def test_command_not_expired_before_timeout(
        self, workflow_with_params: WorkflowSpec
    ):
        workflow_id = workflow_with_params.get_id()
        clock = MutableClock()
        queue = NotificationQueue()
        orchestrator = make_orchestrator(
            workflow_with_params, notification_queue=queue, clock=clock
        )

        orchestrator.stage_config(
            workflow_id, source_name="det_1", params={}, aux_source_names={}
        )
        orchestrator.commit_workflow(workflow_id)
        orchestrator.reset_workflow(workflow_id)

        clock.advance(PENDING_COMMAND_TIMEOUT_SECONDS - 1.0)
        orchestrator.expire_pending_commands()

        assert queue.get_all_events() == []

    def test_expired_command_notified_only_once(
        self, workflow_with_params: WorkflowSpec
    ):
        workflow_id = workflow_with_params.get_id()
        clock = MutableClock()
        queue = NotificationQueue()
        orchestrator = make_orchestrator(
            workflow_with_params, notification_queue=queue, clock=clock
        )

        orchestrator.stage_config(
            workflow_id, source_name="det_1", params={}, aux_source_names={}
        )
        orchestrator.commit_workflow(workflow_id)
        orchestrator.reset_workflow(workflow_id)

        clock.advance(PENDING_COMMAND_TIMEOUT_SECONDS + 1.0)
        orchestrator.expire_pending_commands()
        orchestrator.expire_pending_commands()

        errors = [
            event
            for event in queue.get_all_events()
            if event.notification_type is NotificationType.ERROR
        ]
        assert len(errors) == 1
