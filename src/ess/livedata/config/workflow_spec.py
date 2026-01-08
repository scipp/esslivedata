# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""
Models for data reduction workflow widget creation and configuration.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

import scipp as sc
from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar('T')

JobNumber = uuid.UUID


class WorkflowOutputsBase(BaseModel):
    """Base class for all workflow output models.

    Provides common configuration for output models, including support for
    arbitrary types like scipp.DataArray.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DefaultOutputs(WorkflowOutputsBase):
    """Default outputs model for workflows that don't specify outputs.

    Provides a single 'result' field for simple workflows.
    """

    result: sc.DataArray = Field(title='Result', description='Workflow output.')


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

    def __str__(self) -> str:
        """
        String representation for use in stream names and identifiers.

        Returns '{source_name}/{job_number}' to ensure unique identification
        across detectors in multi-detector workflows.
        """
        return f"{self.source_name}/{self.job_number}"


class AuxSourcesBase(BaseModel):
    """
    Base class for auxiliary source models.

    Auxiliary source models define the available auxiliary data streams that a workflow
    can consume. Subclasses should define fields with Literal or Enum types to specify
    the available stream choices.

    The `render()` method can be overridden to transform field values into job-specific
    or source-specific stream names for routing purposes.
    """

    def render(self, job_id: JobId) -> dict[str, str]:
        """
        Render auxiliary source stream names for a specific job.

        The default implementation returns the model values unchanged, preserving
        backward compatibility with existing workflows.

        Parameters
        ----------
        job_id:
            The job identifier, containing both source_name and job_number.
            Subclasses can use this to create job-specific or source-specific
            stream names (e.g., "{job_id.job_number}/roi_rectangle").

        Returns
        -------
        :
            Mapping from field names to stream names for routing. The keys are the
            field names defined in the model, and the values are the stream names
            that the job should subscribe to.
        """
        return self.model_dump(mode='json')


class ResultKey(BaseModel, frozen=True):
    # Workflows produce one or more named outputs. Each output is serialized as a
    # separate da00 message. The output_name identifies which output this key refers to.
    workflow_id: WorkflowId = Field(description="Workflow ID")
    job_id: JobId = Field(description="Job ID")
    output_name: str = Field(
        default='result', description="Name of the workflow output"
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
    outputs: type[BaseModel] = Field(
        default=DefaultOutputs,
        description=(
            "Pydantic model defining workflow outputs with their metadata. "
            "Field names are simplified identifiers (e.g., 'i_of_d_two_theta') "
            "that match keys returned by workflow.finalize(). Field types should "
            "be scipp.DataArray. Field metadata (title, description) provides "
            "human-readable names and explanations for the UI. "
            "\n\n"
            "IMPORTANT: Use default_factory to provide an empty DataArray template "
            "with the correct structure (dims, coords, units). This enables the "
            "dashboard to perform automatic plotter selection before data exists. "
            "Example:\n"
            "    field: sc.DataArray = Field(\n"
            "        default_factory=lambda: sc.DataArray(\n"
            "            sc.zeros(dims=['x'], shape=[0], unit='counts'),\n"
            "            coords={'x': sc.arange('x', 0, unit='m')}\n"
            "        ),\n"
            "        title='Result',\n"
            "        description='Output description'\n"
            "    )"
        ),
    )

    @field_validator('outputs', mode='after')
    @classmethod
    def validate_unique_output_titles(cls, outputs: type[BaseModel]) -> type[BaseModel]:
        """Validate that output titles are unique within the workflow."""
        title_counts: dict[str, list[str]] = defaultdict(list)
        for field_name, field_info in outputs.model_fields.items():
            title = field_info.title if field_info.title is not None else field_name
            title_counts[title].append(field_name)

        duplicates = {
            title: fields for title, fields in title_counts.items() if len(fields) > 1
        }

        if duplicates:
            dup_str = ", ".join(
                f"'{title}' (fields: {', '.join(fields)})"
                for title, fields in duplicates.items()
            )
            raise ValueError(
                f"Output titles must be unique within a workflow. "
                f"Duplicate titles found: {dup_str}"
            )

        return outputs

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

    def get_output_template(self, output_name: str) -> sc.DataArray | None:
        """
        Get a template DataArray for the specified output.

        Returns a DataArray created by the field's default_factory. By convention,
        this should be an empty DataArray (shape=0) that demonstrates the expected
        structure (dims, coords, units) of the workflow output.

        Parameters
        ----------
        output_name:
            Name of the output field.

        Returns
        -------
        :
            DataArray created by the field's default_factory, or None if the field
            is not found or has no default_factory defined.
        """
        field_info = self.outputs.model_fields.get(output_name)
        if field_info is None:
            return None

        # Call the factory to get a fresh template
        if field_info.default_factory:
            return field_info.default_factory()

        return None


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

    Note on message_id vs job_number
    --------------------------------
    These two identifiers serve fundamentally different purposes:

    - ``message_id``: Transient identifier for command/response correlation
      (ACK pattern). Frontend generates it, backend echoes it in
      CommandAcknowledgement. Discarded once ACK is received. Used only for
      the request/response handshake.

    - ``job_number``: Persistent job identity for the entire job lifecycle.
      Used in JobId for result routing, job commands (stop/reset), and data
      correlation.

    Currently this message conflates "configure" and "start" into a single command, so
    both fields are present. Future work (see issue #445) may split into separate
    WorkflowConfig (config-only) and WorkflowStart messages. In that design:

    - WorkflowConfig would have message_id (for ACK, and as a "config handle") but no
      job_number (not starting a job yet).
    - WorkflowStart would have its own message_id (for ACK), job_number (new
      job identity), and a config_ref pointing to a previously ACK'd
      config's message_id.

    This split would enable multiple independent jobs (e.g., frontend + NICOS) to share
    the same configuration while having distinct job lifecycles.
    """

    identifier: WorkflowId = Field(
        description="Hash of the workflow, used to identify the workflow."
    )
    message_id: str | None = Field(
        default=None,
        description=(
            "Transient identifier for command/response correlation. Frontend generates "
            "this UUID and backend echoes it in CommandAcknowledgement responses. "
            "Distinct from job_number which identifies the job itself."
        ),
    )
    job_number: JobNumber | None = Field(
        default=None,
        description=(
            "Persistent job identity used for result routing and job control commands. "
            "Forms part of JobId (source_name + job_number). Distinct from message_id "
            "which is only for command acknowledgement correlation."
        ),
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

    @classmethod
    def from_params(
        cls,
        workflow_id: WorkflowId,
        params: dict | None = None,
        aux_source_names: dict | None = None,
        job_number: JobNumber | None = None,
        message_id: str | None = None,
    ) -> WorkflowConfig:
        """
        Create a WorkflowConfig from parameters.

        Parameters
        ----------
        workflow_id:
            Identifier for the workflow
        params:
            Workflow parameters as dict, or None if no params
        aux_source_names:
            Auxiliary source selections as dict, or None if no aux sources
        job_number:
            Optional job number (generated if not provided)
        message_id:
            Optional message ID for command acknowledgement tracking

        Returns
        -------
        :
            WorkflowConfig instance ready to be sent to backend
        """
        return cls(
            identifier=workflow_id,
            message_id=message_id,
            job_number=job_number if job_number is not None else uuid.uuid4(),
            aux_source_names=aux_source_names or {},
            params=params or {},
        )


def _is_timeseries_output(da: sc.DataArray) -> bool:
    """Check if DataArray represents a timeseries (0-D with time coord)."""
    return da.ndim == 0 and 'time' in da.coords


def find_timeseries_outputs(
    workflow_registry: Mapping[WorkflowId, WorkflowSpec],
) -> list[tuple[WorkflowId, str, str]]:
    """
    Find all timeseries outputs in the workflow registry.

    A timeseries output is a 0-D DataArray with a 'time' coordinate.
    This is determined by checking the default_factory template of each
    output field in the workflow spec.

    Parameters
    ----------
    workflow_registry:
        Registry of workflow specs to search.

    Returns
    -------
    :
        List of (workflow_id, source_name, output_name) tuples for all
        timeseries outputs found. Each source_name from the workflow's
        source_names list is paired with each timeseries output_name.
    """
    results: list[tuple[WorkflowId, str, str]] = []

    for workflow_id, spec in workflow_registry.items():
        if spec.outputs is None:
            continue

        # Find timeseries output fields
        timeseries_outputs: list[str] = []
        for field_name, field_info in spec.outputs.model_fields.items():
            if not field_info.default_factory:
                continue
            try:
                template = field_info.default_factory()
                if _is_timeseries_output(template):
                    timeseries_outputs.append(field_name)
            except Exception:  # noqa: S112
                continue

        # Create entries for each source_name x output combination
        results.extend(
            (workflow_id, source_name, output_name)
            for source_name in spec.source_names
            for output_name in timeseries_outputs
        )

    return results
