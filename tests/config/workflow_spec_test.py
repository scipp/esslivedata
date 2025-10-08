# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from enum import Enum

import pytest
from pydantic import BaseModel, Field

from ess.livedata.config.workflow_spec import (
    PersistentWorkflowConfig,
    PersistentWorkflowConfigs,
    WorkflowConfig,
    WorkflowId,
    WorkflowSpec,
)


@pytest.fixture
def sample_workflow_id() -> WorkflowId:
    return WorkflowId(
        instrument="INSTRUMENT",
        namespace="NAMESPACE",
        name="NAME",
        version=1,
    )


@pytest.fixture
def sample_workflow_config(sample_workflow_id: WorkflowId) -> WorkflowConfig:
    return WorkflowConfig(
        identifier=sample_workflow_id,
        params={"param1": 10, "param2": "value"},
    )


class TestPersistentWorkflowConfig:
    def test_ser_deser_empty(self) -> None:
        configs = PersistentWorkflowConfigs()
        dumped = configs.model_dump()
        loaded = PersistentWorkflowConfigs.model_validate(dumped)
        assert configs == loaded

    def test_ser_deser_single(
        self, sample_workflow_id: WorkflowId, sample_workflow_config: WorkflowConfig
    ) -> None:
        pwc = PersistentWorkflowConfig(
            source_names=["source1"], config=sample_workflow_config
        )
        configs = PersistentWorkflowConfigs(configs={sample_workflow_id: pwc})
        dumped = configs.model_dump()
        loaded = PersistentWorkflowConfigs.model_validate(dumped)
        assert configs == loaded

    def test_ser_deser_multiple(
        self, sample_workflow_id: WorkflowId, sample_workflow_config: WorkflowConfig
    ) -> None:
        pwc1 = PersistentWorkflowConfig(
            source_names=["source1"], config=sample_workflow_config
        )
        pwc2 = PersistentWorkflowConfig(
            source_names=["source2", "source3"],
            config=WorkflowConfig(
                identifier=WorkflowId(
                    instrument="INSTRUMENT2",
                    namespace="NAMESPACE2",
                    name="NAME2",
                    version=2,
                ),
                params={"paramA": 5.0},
            ),
        )
        configs = PersistentWorkflowConfigs(
            configs={sample_workflow_id: pwc1, pwc2.config.identifier: pwc2}
        )
        dumped = configs.model_dump()
        loaded = PersistentWorkflowConfigs.model_validate(dumped)
        assert configs == loaded


class TestWorkflowSpecAuxSources:
    """Tests for WorkflowSpec.aux_sources field."""

    def test_workflow_spec_without_aux_sources(self) -> None:
        """Test that WorkflowSpec can be created without aux_sources."""
        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
        )
        assert spec.aux_sources is None

    def test_workflow_spec_with_aux_sources_model(self) -> None:
        """Test WorkflowSpec with aux_sources as a Pydantic model."""

        class MonitorChoice(str, Enum):
            MONITOR1 = "monitor1"
            MONITOR2 = "monitor2"

        class AuxSources(BaseModel):
            monitor: MonitorChoice = Field(
                default=MonitorChoice.MONITOR1,
                title="Monitor",
                description="Select which monitor to use",
            )

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            aux_sources=AuxSources,
        )
        assert spec.aux_sources is AuxSources

    def test_workflow_spec_serialization_with_aux_sources(self) -> None:
        """Test WorkflowSpec with aux_sources serialization."""

        class AuxSources(BaseModel):
            rotation: str = Field(title="Rotation", description="Select rotation")

        spec = WorkflowSpec(
            instrument="test",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            params=None,
            aux_sources=AuxSources,
        )

        # Serialize and deserialize
        dumped = spec.model_dump()
        loaded = WorkflowSpec.model_validate(dumped)

        assert loaded.aux_sources is not None
        # Both should be the same class (not just equal instances)
        assert loaded.aux_sources.__name__ == spec.aux_sources.__name__


class TestWorkflowConfigAuxSourceNames:
    """Tests for WorkflowConfig.aux_source_names field."""

    def test_workflow_config_default_aux_source_names(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test that WorkflowConfig has empty aux_source_names by default."""
        config = WorkflowConfig(identifier=sample_workflow_id)
        assert config.aux_source_names == {}

    def test_workflow_config_with_aux_source_names(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test that WorkflowConfig can store aux_source_names as dict."""
        config = WorkflowConfig(
            identifier=sample_workflow_id,
            aux_source_names={"monitor": "monitor1", "rotation": "rotation_a"},
        )
        assert config.aux_source_names == {
            "monitor": "monitor1",
            "rotation": "rotation_a",
        }

    def test_workflow_config_serialization_with_aux_source_names(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test that WorkflowConfig with aux_source_names serializes correctly."""
        config = WorkflowConfig(
            identifier=sample_workflow_id,
            aux_source_names={"monitor": "monitor2"},
            params={"param1": 10},
        )

        dumped = config.model_dump()
        loaded = WorkflowConfig.model_validate(dumped)

        assert loaded.aux_source_names == {"monitor": "monitor2"}
        assert loaded.params == {"param1": 10}


class TestPersistentWorkflowConfigWithAuxSources:
    """Tests for PersistentWorkflowConfig with aux_source_names."""

    def test_persistent_config_serialization_with_aux_sources(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test that PersistentWorkflowConfig correctly serializes aux_source_names."""
        config = WorkflowConfig(
            identifier=sample_workflow_id,
            aux_source_names={"monitor": "monitor1"},
            params={"param1": 5},
        )
        pwc = PersistentWorkflowConfig(source_names=["source1"], config=config)

        # Test serialization through PersistentWorkflowConfigs
        configs = PersistentWorkflowConfigs(configs={sample_workflow_id: pwc})
        dumped = configs.model_dump()
        loaded = PersistentWorkflowConfigs.model_validate(dumped)

        assert loaded.configs[sample_workflow_id].config.aux_source_names == {
            "monitor": "monitor1"
        }
        assert loaded.configs[sample_workflow_id].config.params == {"param1": 5}
