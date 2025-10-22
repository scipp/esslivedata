# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from enum import Enum

import pytest
from pydantic import BaseModel, Field

from ess.livedata.config.workflow_spec import (
    AuxSourcesBase,
    JobId,
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


class TestWorkflowConfigFromParams:
    """Tests for WorkflowConfig.from_params() helper method."""

    def test_from_params_with_all_arguments(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test from_params with params and aux_source_names."""

        class ParamsModel(BaseModel):
            param1: int = 10
            param2: str = "value"

        class AuxSourcesModel(BaseModel):
            monitor: str = "monitor1"

        params = ParamsModel(param1=20, param2="custom")
        aux_sources = AuxSourcesModel(monitor="monitor1")

        config = WorkflowConfig.from_params(
            workflow_id=sample_workflow_id,
            params=params,
            aux_source_names=aux_sources,
        )

        assert config.identifier == sample_workflow_id
        assert config.params == {"param1": 20, "param2": "custom"}
        assert config.aux_source_names == {"monitor": "monitor1"}
        assert config.job_number is not None  # Should be auto-generated

    def test_from_params_with_none_params(self, sample_workflow_id: WorkflowId) -> None:
        """Test from_params with no params (None)."""
        config = WorkflowConfig.from_params(
            workflow_id=sample_workflow_id,
            params=None,
            aux_source_names=None,
        )

        assert config.identifier == sample_workflow_id
        assert config.params == {}
        assert config.aux_source_names == {}
        assert config.job_number is not None

    def test_from_params_with_custom_job_number(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test from_params with explicit job_number."""
        import uuid

        custom_job_number = uuid.uuid4()

        config = WorkflowConfig.from_params(
            workflow_id=sample_workflow_id,
            params=None,
            job_number=custom_job_number,
        )

        assert config.job_number == custom_job_number

    def test_from_params_serializes_pydantic_model(
        self, sample_workflow_id: WorkflowId
    ) -> None:
        """Test that from_params correctly serializes Pydantic model to dict."""

        class ComplexParams(BaseModel):
            nested_value: int = 5
            string_list: list[str] = ["a", "b"]

        params = ComplexParams(nested_value=10, string_list=["x", "y", "z"])

        config = WorkflowConfig.from_params(
            workflow_id=sample_workflow_id,
            params=params,
        )

        # Should be serialized to dict
        assert isinstance(config.params, dict)
        assert config.params == {"nested_value": 10, "string_list": ["x", "y", "z"]}


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


class TestAuxSourcesBase:
    """Tests for AuxSourcesBase render() method."""

    def test_default_render_returns_model_dump(self) -> None:
        """Test that default render() returns model_dump(mode='json')."""
        from typing import Literal

        class SimpleAuxSources(AuxSourcesBase):
            monitor: Literal['monitor1'] = 'monitor1'

        aux_sources = SimpleAuxSources()
        job_id = JobId(source_name='detector1', job_number='test-uuid-123')

        rendered = aux_sources.render(job_id)

        # Default implementation should return the model dump unchanged
        assert rendered == {'monitor': 'monitor1'}
        assert rendered == aux_sources.model_dump(mode='json')

    def test_custom_render_transforms_stream_names(self) -> None:
        """Test that custom render() can transform stream names."""
        from typing import Literal

        class RoiAuxSources(AuxSourcesBase):
            roi: Literal['roi_rectangle', 'roi_polygon'] = 'roi_rectangle'

            def render(self, job_id: JobId) -> dict[str, str]:
                base = self.model_dump(mode='json')
                return {
                    field: f"{job_id.job_number}/{stream}"
                    for field, stream in base.items()
                }

        aux_sources = RoiAuxSources(roi='roi_polygon')
        job_id = JobId(source_name='detector1', job_number='abc-123')

        rendered = aux_sources.render(job_id)

        assert rendered == {'roi': 'abc-123/roi_polygon'}

    def test_custom_render_with_source_name_prefix(self) -> None:
        """Test custom render() that uses source_name for prefixing."""
        from typing import Literal

        class SourcePrefixedAuxSources(AuxSourcesBase):
            monitor: Literal['monitor1', 'monitor2'] = 'monitor1'

            def render(self, job_id: JobId) -> dict[str, str]:
                base = self.model_dump(mode='json')
                return {
                    field: f"{job_id.source_name}/{stream}"
                    for field, stream in base.items()
                }

        aux_sources = SourcePrefixedAuxSources(monitor='monitor2')
        job_id = JobId(source_name='detector1', job_number='uuid-456')

        rendered = aux_sources.render(job_id)

        assert rendered == {'monitor': 'detector1/monitor2'}

    def test_render_with_multiple_fields(self) -> None:
        """Test render() with multiple aux source fields."""
        from typing import Literal

        class MultiAuxSources(AuxSourcesBase):
            incident_monitor: Literal['monitor1'] = 'monitor1'
            transmission_monitor: Literal['monitor2'] = 'monitor2'

        aux_sources = MultiAuxSources()
        job_id = JobId(source_name='detector1', job_number='test-id')

        rendered = aux_sources.render(job_id)

        assert rendered == {
            'incident_monitor': 'monitor1',
            'transmission_monitor': 'monitor2',
        }
