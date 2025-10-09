# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Test aux_sources serialization with StrEnum fields."""

from enum import StrEnum
from typing import Literal

import pydantic

from ess.livedata.config.workflow_spec import WorkflowConfig, WorkflowId


class TestAuxSourcesSerialization:
    """Test aux_sources serialization with different field types."""

    def test_str_enum_model_dump_produces_strings(self) -> None:
        """Test that StrEnum fields serialize to strings with model_dump."""

        class TimeseriesEnum(StrEnum):
            TS_A = 'timeseries_a'
            TS_B = 'timeseries_b'

        class AuxSourcesModel(pydantic.BaseModel):
            x_param: TimeseriesEnum = TimeseriesEnum.TS_A
            y_param: TimeseriesEnum = TimeseriesEnum.TS_B

        # Create model instance
        model = AuxSourcesModel()

        # Verify model_dump(mode='json') produces dict[str, str]
        aux_dict = model.model_dump(mode='json')
        assert aux_dict == {'x_param': 'timeseries_a', 'y_param': 'timeseries_b'}
        assert all(isinstance(v, str) for v in aux_dict.values())

    def test_literal_model_dump_produces_strings(self) -> None:
        """Test that Literal fields serialize to strings."""

        class AuxSourcesModel(pydantic.BaseModel):
            incident_monitor: Literal['monitor1'] = 'monitor1'
            transmission_monitor: Literal['monitor2'] = 'monitor2'

        # Create model instance
        model = AuxSourcesModel()

        # Verify model_dump(mode='json') produces dict[str, str]
        aux_dict = model.model_dump(mode='json')
        assert aux_dict == {
            'incident_monitor': 'monitor1',
            'transmission_monitor': 'monitor2',
        }
        assert all(isinstance(v, str) for v in aux_dict.values())

    def test_workflow_config_from_params_with_str_enum(self) -> None:
        """Test WorkflowConfig.from_params accepts and serializes StrEnum model."""

        class TimeseriesEnum(StrEnum):
            TS_A = 'timeseries_a'
            TS_B = 'timeseries_b'

        class AuxSourcesModel(pydantic.BaseModel):
            x_param: TimeseriesEnum = TimeseriesEnum.TS_A

        # Create aux sources model
        aux_sources = AuxSourcesModel()

        # Create WorkflowConfig
        workflow_id = WorkflowId(
            instrument='test', namespace='correlation', name='histogram_1d', version=1
        )
        config = WorkflowConfig.from_params(
            workflow_id=workflow_id,
            aux_source_names=aux_sources,
        )

        # Verify serialization produces dict[str, str]
        assert config.aux_source_names == {'x_param': 'timeseries_a'}
        assert all(isinstance(v, str) for v in config.aux_source_names.values())

    def test_workflow_config_from_params_with_literal(self) -> None:
        """Test WorkflowConfig.from_params accepts and serializes Literal model."""

        class AuxSourcesModel(pydantic.BaseModel):
            monitor: Literal['monitor1', 'monitor2'] = 'monitor1'

        # Create aux sources model
        aux_sources = AuxSourcesModel()

        # Create WorkflowConfig
        workflow_id = WorkflowId(
            instrument='test', namespace='test', name='test', version=1
        )
        config = WorkflowConfig.from_params(
            workflow_id=workflow_id,
            aux_source_names=aux_sources,
        )

        # Verify serialization produces dict[str, str]
        assert config.aux_source_names == {'monitor': 'monitor1'}
        assert all(isinstance(v, str) for v in config.aux_source_names.values())

    def test_workflow_config_rejects_non_string_aux_sources(self) -> None:
        """Test that aux_sources with non-string fields are rejected by WorkflowConfig.

        This ensures aux_sources models must only contain string-serializable fields
        (Literal, StrEnum, str) or provide custom serialization.
        """
        import pytest

        # Model with bool field (not string-serializable)
        class InvalidAuxSourcesModel(pydantic.BaseModel):
            use_monitor: bool = True
            monitor_name: str = "monitor1"

        aux_sources = InvalidAuxSourcesModel()

        workflow_id = WorkflowId(
            instrument='test', namespace='test', name='test', version=1
        )

        # Should raise ValidationError when trying to create WorkflowConfig
        with pytest.raises(
            pydantic.ValidationError, match="Input should be a valid string"
        ):
            WorkflowConfig.from_params(
                workflow_id=workflow_id,
                aux_source_names=aux_sources,
            )

    def test_aux_sources_with_custom_serialization(self) -> None:
        """Test that aux_sources can use custom serialization via model_serializer."""
        from pydantic import model_serializer

        class CustomSerializationModel(pydantic.BaseModel):
            """Aux sources model with bool field but custom serialization to strings."""

            use_primary: bool = True
            monitor_id: int = 1

            @model_serializer(mode='wrap', when_used='json')
            def serialize_model(self, serializer, info):
                """Custom serializer that converts all fields to strings."""
                # Get default serialization
                data = serializer(self)
                # Convert all values to strings
                return {
                    k: str(v).lower() if isinstance(v, bool) else str(v)
                    for k, v in data.items()
                }

        aux_sources = CustomSerializationModel(use_primary=True, monitor_id=42)

        # Verify custom serialization produces strings
        dumped = aux_sources.model_dump(mode='json')
        assert dumped == {'use_primary': 'true', 'monitor_id': '42'}
        assert all(isinstance(v, str) for v in dumped.values())

        # Should work with WorkflowConfig
        workflow_id = WorkflowId(
            instrument='test', namespace='test', name='test', version=1
        )
        config = WorkflowConfig.from_params(
            workflow_id=workflow_id,
            aux_source_names=aux_sources,
        )

        assert config.aux_source_names == {'use_primary': 'true', 'monitor_id': '42'}
        assert all(isinstance(v, str) for v in config.aux_source_names.values())
