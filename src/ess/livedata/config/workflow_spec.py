# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""
Models for data reduction workflow widget creation and configuration.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, Field, field_serializer, field_validator

T = TypeVar('T')

JobNumber = uuid.UUID


class WorkflowId(BaseModel, frozen=True):
    instrument: str
    namespace: str
    name: str
    version: int

    def __str__(self) -> str:
        return f"{self.instrument}/{self.namespace}/{self.name}/{self.version}"

    @staticmethod
    def from_string(workflow_id_str: str) -> WorkflowId:
        """Parse WorkflowId from string representation."""
        parts = workflow_id_str.split('/')
        if len(parts) != 4:
            raise ValueError(f"Invalid WorkflowId string format: {workflow_id_str}")
        return WorkflowId(
            instrument=parts[0],
            namespace=parts[1],
            name=parts[2],
            version=int(parts[3]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class JobId:
    source_name: str
    job_number: JobNumber


class ResultKey(BaseModel, frozen=True):
    # If the job produced a DataGroup then it will be serialized as multiple da00
    # messages. Each message corresponds to a single DataArray value the DataGroup.
    # In the case the output_name is set.
    workflow_id: WorkflowId = Field(description="Workflow ID")
    job_id: JobId = Field(description="Job ID")
    output_name: str | None = Field(
        default=None,
        description="Name of the output, if the job produces multiple outputs",
    )


class WorkflowSpec(BaseModel):
    """
    Model for workflow specification.

    This model is used to define a workflow and its parameters. The ESSlivedata
    dashboard uses these to create user interfaces for configuring workflows.
    """

    instrument: str = Field(
        description="Name of the instrument this workflow is associated with."
    )
    namespace: str = Field(
        default='data_reduction',
        description="Namespace for the workflow, used to group workflows logically.",
    )
    name: str = Field(description="Name of the workflow. Used internally.")
    version: int = Field(description="Version of the workflow.")
    title: str = Field(description="Title of the workflow. For display in the UI.")
    description: str = Field(description="Description of the workflow.")
    source_names: list[str] = Field(
        default_factory=list,
        description="List of detector/other streams the workflow can be applied to.",
    )
    aux_sources: type[BaseModel] | None = Field(
        default=None,
        description=(
            "Pydantic model defining auxiliary data sources with their configuration. "
            "Field names define the aux source identifiers, and field types (typically "
            "Literal or Enum) define the available stream choices. Field metadata "
            "(title, description) provides UI information."
        ),
    )
    params: type[BaseModel] | None = Field(description="Model for workflow param.")

    def get_id(self) -> WorkflowId:
        """
        Get a unique identifier for the workflow.

        The identifier is a combination of instrument, namespace, name, and version.
        """
        return WorkflowId(
            instrument=self.instrument,
            namespace=self.namespace,
            name=self.name,
            version=self.version,
        )


@dataclass
class JobSchedule:
    """
    Defines when a job should start and optionally when it should end.

    All timestamps are in nanoseconds since the epoch (UTC) and reference the timestamps
    of the raw data being processed (as opposed to when it should be processed).
    """

    start_time: int | None = None  # When job should start processing
    end_time: int | None = None  # When job should stop (None = no limit)

    def __post_init__(self) -> None:
        """Validate the schedule configuration."""
        if (
            self.end_time is not None
            and self.start_time is not None
            and self.end_time <= self.start_time
        ):
            raise ValueError(
                f"Job end_time={self.end_time} must be greater than start_time="
                f"{self.start_time}, or start_time must be None (immediate start)"
            )

    def should_start(self, current_time: int) -> bool:
        """
        Check if the job should start based on the current time.

        Returns True if the job should start, False otherwise.
        """
        return self.start_time is None or current_time >= self.start_time


class WorkflowConfig(BaseModel):
    """
    Model for workflow configuration.

    This model is used to set the parameter values for a specific workflow. The values
    correspond to the parameters defined in the workflow specification
    :py:class:`WorkflowSpec`.
    """

    identifier: WorkflowId = Field(
        description="Hash of the workflow, used to identify the workflow."
    )
    job_number: JobNumber | None = Field(
        default=None, description=("Unique identifier to identify jobs and job results")
    )
    schedule: JobSchedule = Field(
        default_factory=JobSchedule, description="Schedule for the workflow."
    )
    aux_source_names: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Selected auxiliary source names as a mapping from field name (as defined "
            "in WorkflowSpec.aux_sources) to the selected stream name."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the workflow, as JSON-serialized Pydantic model.",
    )


class PersistentWorkflowConfig(BaseModel):
    """
    Persistent storage for workflow configuration including source selection.

    This model stores both the source names selection and the parameter values
    for a workflow, allowing the UI to restore the last-used configuration.
    """

    source_names: list[str] = Field(
        default_factory=list,
        description="Selected source names for this workflow",
    )
    config: WorkflowConfig = Field(
        description="Configuration for the workflow, including parameter values",
    )


class PersistentWorkflowConfigs(BaseModel):
    """
    Collection of all persistent workflow configurations.

    This model stores persistent configurations for multiple workflows in a single
    config item, making it easy to manage and clean up old configurations.
    """

    configs: dict[WorkflowId, PersistentWorkflowConfig] = Field(
        default_factory=dict,
        description="Persistent configurations indexed by workflow ID",
    )

    @field_serializer('configs')
    def serialize_configs(
        self, configs: dict[WorkflowId, PersistentWorkflowConfig]
    ) -> dict[str, Any]:
        """Serialize configs by converting WorkflowId keys to strings."""
        return {str(k): v.model_dump() for k, v in configs.items()}

    @field_validator('configs', mode='before')
    @classmethod
    def validate_configs(cls, v: Any) -> dict[WorkflowId, PersistentWorkflowConfig]:
        """Validate configs by converting string keys back to WorkflowId objects."""
        if isinstance(v, dict) and v:
            # Check if keys are strings (during deserialization)
            first_key = next(iter(v.keys()))
            if isinstance(first_key, str):
                result = {}
                for k, config_data in v.items():
                    workflow_id = WorkflowId.from_string(k)
                    # Reconstruct PersistentWorkflowConfig
                    if isinstance(config_data, dict):
                        config_data = PersistentWorkflowConfig.model_validate(
                            config_data
                        )
                    result[workflow_id] = config_data
                return result
        return v

    def cleanup_missing_workflows(self, current_workflow_ids: set[WorkflowId]) -> None:
        """Remove configurations for workflows that no longer exist."""
        missing_ids = set(self.configs.keys()) - current_workflow_ids
        for workflow_id in missing_ids:
            del self.configs[workflow_id]


class WorkflowStatusType(str, Enum):
    """
    Status of a workflow execution.

    The idea of the "stopped" status is to have the option of still displaying the data
    in the UI. The UI may then remove the workflow entirely in a separate step. This is
    not implemented yet.
    """

    STARTING = "starting"
    STOPPING = "stopping"
    RUNNING = "running"
    STARTUP_ERROR = "startup_error"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class WorkflowStatus(BaseModel):
    """
    Model for workflow status.

    This model is used to define the status of a workflow, including its ID and status.
    """

    source_name: str = Field(description="Source name the workflow is associated with.")
    workflow_id: WorkflowId | None = Field(
        default=None, description="ID of the workflow."
    )
    status: WorkflowStatusType = Field(
        default=WorkflowStatusType.UNKNOWN, description="Status of the workflow."
    )
    message: str = Field(
        default='', description="Optional message providing additional information."
    )
    timestamp: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp when the status was created or updated.",
    )
