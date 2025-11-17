# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for JobOrchestrator initialization and config management."""

import pydantic
import pytest

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.configuration_adapter import ConfigurationState
from ess.livedata.dashboard.job_orchestrator import JobOrchestrator
from ess.livedata.dashboard.workflow_config_service import WorkflowConfigService
from ess.livedata.fakes import FakeMessageSink


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
    )


class FakeWorkflowConfigService(WorkflowConfigService):
    """Minimal fake for WorkflowConfigService."""

    def subscribe_to_workflow_status(self, source_name, callback):
        pass


class TestJobOrchestratorInitialization:
    """Test JobOrchestrator initialization behavior."""

    def test_all_workflows_initialized_without_config_store(
        self, workflow_with_params: WorkflowSpec, workflow_no_params: WorkflowSpec
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
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1", "det_2"],
            workflow_registry=registry,
            config_store=None,
        )

        # Both workflows should exist in _workflows
        staged_1 = orchestrator.get_staged_config(workflow_id_1)
        staged_2 = orchestrator.get_staged_config(workflow_id_2)

        assert isinstance(staged_1, dict)
        assert isinstance(staged_2, dict)

    def test_workflow_with_params_gets_default_config(
        self, workflow_with_params: WorkflowSpec
    ):
        """Workflow with params model should get default params for all sources."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1", "det_2"],
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
        self, workflow_no_params: WorkflowSpec
    ):
        """Workflow without params should exist but have empty staged_jobs."""
        workflow_id = workflow_no_params.get_id()
        registry = {workflow_id: workflow_no_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1", "det_2"],
            workflow_registry=registry,
            config_store=None,
        )

        staged = orchestrator.get_staged_config(workflow_id)
        assert isinstance(staged, dict)
        assert len(staged) == 0

    def test_workflow_empty_sources_has_empty_staged_jobs(
        self, workflow_empty_sources: WorkflowSpec
    ):
        """Workflow with empty source_names should have empty staged_jobs."""
        workflow_id = workflow_empty_sources.get_id()
        registry = {workflow_id: workflow_empty_sources}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1", "det_2"],
            workflow_registry=registry,
            config_store=None,
        )

        staged = orchestrator.get_staged_config(workflow_id)
        assert isinstance(staged, dict)
        assert len(staged) == 0

    def test_loads_config_from_store(self, workflow_with_params: WorkflowSpec):
        """Should load and use config from store when available."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        # Setup config store with persisted config
        config_store = {
            workflow_id: ConfigurationState(
                source_names=["det_1"],  # Only one source
                params={"threshold": 50.0, "mode": "custom"},
                aux_source_names={},
            ).model_dump()
        }

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1", "det_2"],
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
        self, workflow_with_params: WorkflowSpec
    ):
        """Invalid configs are loaded as-is (validation happens later in UI)."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        # Setup config store with invalid param types
        config_store = {
            workflow_id: {
                "source_names": ["det_1"],
                "params": {"threshold": "not_a_float"},  # Invalid type
                "aux_source_names": {},
            }
        }

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1", "det_2"],
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
        self, workflow_with_params_and_aux: WorkflowSpec
    ):
        """Should load aux_source_names from config store."""
        workflow_id = workflow_with_params_and_aux.get_id()
        registry = {workflow_id: workflow_with_params_and_aux}

        config_store = {
            workflow_id: ConfigurationState(
                source_names=["det_1"],
                params={"threshold": 75.0, "mode": "special"},
                aux_source_names={"monitor": "monitor_2"},
            ).model_dump()
        }

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1"],
            workflow_registry=registry,
            config_store=config_store,
        )

        staged = orchestrator.get_staged_config(workflow_id)
        assert staged["det_1"].params["threshold"] == 75.0
        assert staged["det_1"].aux_source_names
        assert staged["det_1"].aux_source_names["monitor"] == "monitor_2"

    def test_get_staged_config_never_raises_for_valid_workflow_id(
        self, workflow_with_params: WorkflowSpec
    ):
        """get_staged_config should never raise KeyError for valid workflow_id."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1"],
            workflow_registry=registry,
            config_store=None,
        )

        # Should not raise
        result = orchestrator.get_staged_config(workflow_id)
        assert isinstance(result, dict)

    def test_get_staged_config_raises_for_unknown_workflow_id(
        self, workflow_with_params: WorkflowSpec
    ):
        """get_staged_config should raise KeyError for unknown workflow_id."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1"],
            workflow_registry=registry,
            config_store=None,
        )

        unknown_id = WorkflowId(
            instrument="unknown", namespace="unknown", name="unknown", version=99
        )

        # Should raise KeyError for unknown workflow
        with pytest.raises(KeyError):
            orchestrator.get_staged_config(unknown_id)

    def test_get_staged_config_returns_dict(self, workflow_with_params: WorkflowSpec):
        """get_staged_config returns dict mapping source names to configs."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1", "det_2"],
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
        self, workflow_params_without_defaults: WorkflowSpec
    ):
        """Workflow with params that can't be instantiated gets empty WorkflowState."""
        workflow_id = workflow_params_without_defaults.get_id()
        registry = {workflow_id: workflow_params_without_defaults}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1"],
            workflow_registry=registry,
            config_store=None,
        )

        # Should exist in _workflows but have empty staged_jobs
        staged = orchestrator.get_staged_config(workflow_id)
        assert isinstance(staged, dict)
        assert len(staged) == 0


class TestJobOrchestratorMutationSafety:
    """Test that JobOrchestrator protects against unintended mutations."""

    def test_stage_config_makes_defensive_copy_of_params(
        self, workflow_with_params: WorkflowSpec
    ):
        """Modifying params dict after staging should not affect staged config."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1"],
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
        self, workflow_with_params: WorkflowSpec
    ):
        """Modifying aux_source_names after staging should not affect staged config."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1"],
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
        self, workflow_with_params: WorkflowSpec
    ):
        """Modifying returned config should not affect internal state."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1"],
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
        self, workflow_with_params: WorkflowSpec
    ):
        """Modifying returned active config should not affect internal state."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1"],
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
        self, workflow_with_params: WorkflowSpec
    ):
        """Each JobConfig should have independent params/aux_source_names dicts."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        # Config store with multiple sources
        config_store = {
            workflow_id: ConfigurationState(
                source_names=["det_1", "det_2"],
                params={"threshold": 50.0, "mode": "custom"},
                aux_source_names={"monitor": "monitor_1"},
            ).model_dump()
        }

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1", "det_2"],
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
        self, workflow_with_params: WorkflowSpec
    ):
        """After commit, modifying staged config should not affect active config."""
        workflow_id = workflow_with_params.get_id()
        registry = {workflow_id: workflow_with_params}

        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            workflow_config_service=FakeWorkflowConfigService(),
            source_names=["det_1"],
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
