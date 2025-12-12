# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
WorkflowTemplate protocol for dynamic workflow spec creation.

WorkflowTemplates are factories that create WorkflowSpec instances from user-provided
configuration. This solves the problem where certain workflows (like correlation
histograms) need dynamic identity - the correlation axis should be part of the workflow
type definition, not a runtime parameter.

Example:
    Instead of one "Correlation Histogram" workflow configured at runtime, we have
    distinct workflow specs like "Temperature Correlation Histogram" and "Pressure
    Correlation Histogram", each with their own unique WorkflowId.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, NewType, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel

from .workflow_spec import JobId, JobNumber, WorkflowId, WorkflowSpec

if TYPE_CHECKING:
    from ess.livedata.dashboard.job_orchestrator import JobConfig

SubscriptionId = NewType('SubscriptionId', UUID)


@runtime_checkable
class WorkflowSubscriber(Protocol):
    """
    Protocol for subscribing to workflow job availability notifications.

    This protocol allows components to be notified when a workflow's job
    becomes available (after commit). This is used by frontend executors
    like CorrelationHistogramExecutor to resolve stable TimeseriesReferences
    to full ResultKeys when the timeseries workflow becomes available.
    """

    def subscribe_to_workflow(
        self, workflow_id: WorkflowId, callback: Callable[[JobNumber], None]
    ) -> tuple[SubscriptionId, bool]:
        """
        Subscribe to workflow job availability notifications.

        The callback will be called with the job_number when:
        1. A workflow is committed (immediately after commit)
        2. Immediately if subscribing and workflow already has an active job

        Parameters
        ----------
        workflow_id
            The workflow to subscribe to.
        callback
            Called with job_number when a job becomes active.

        Returns
        -------
        :
            Tuple of (subscription_id, callback_invoked_immediately).
            subscription_id can be used to unsubscribe.
            callback_invoked_immediately is True if the workflow was already
            running and the callback was invoked synchronously during this call.
        """
        ...

    def unsubscribe(self, subscription_id: SubscriptionId) -> None:
        """
        Unsubscribe from workflow availability notifications.

        Parameters
        ----------
        subscription_id
            The subscription ID returned from subscribe_to_workflow.
        """
        ...


class TemplateInstance(BaseModel):
    """
    Persisted template instance configuration.

    Used to recreate dynamic WorkflowSpecs across dashboard restarts.
    """

    template_name: str
    config: dict  # Serialized template configuration


@runtime_checkable
class JobExecutor(Protocol):
    """
    Handles job lifecycle for workflow execution.

    Implementations handle starting and stopping jobs. Backend workflows use
    an executor that sends commands via CommandService to Kafka. Frontend
    workflows (like correlation histograms) use executors that run computations
    directly in the dashboard process via DataService subscriptions.
    """

    def start_jobs(
        self,
        staged_jobs: dict[str, JobConfig],
        job_number: JobNumber,
    ) -> None:
        """
        Start jobs for all staged sources.

        Parameters
        ----------
        staged_jobs:
            Mapping from source name to job configuration.
        job_number:
            Shared job number for all jobs in this batch.
        """
        ...

    def stop_jobs(self, job_ids: list[JobId]) -> None:
        """
        Stop running jobs.

        Parameters
        ----------
        job_ids:
            List of job IDs to stop.
        """
        ...


@runtime_checkable
class WorkflowTemplate(Protocol):
    """
    Protocol for factories that create WorkflowSpec instances dynamically.

    A WorkflowTemplate defines a category of workflows (e.g., "1D Correlation
    Histogram") and can create specific instances (e.g., "Temperature Correlation
    Histogram")
    based on user configuration.
    """

    @property
    def name(self) -> str:
        """
        Template identifier.

        Used for registration and persistence. Should be stable across versions.

        Example: 'correlation_histogram_1d'
        """
        ...

    @property
    def title(self) -> str:
        """
        Human-readable title for the template.

        Displayed in UI when selecting which template to instantiate.

        Example: '1D Correlation Histogram'
        """
        ...

    def get_configuration_model(self) -> type[BaseModel] | None:
        """
        Get the Pydantic model for UI configuration with enum validation.

        Returns None if the template cannot be configured (e.g., missing data).
        The model defines what the user selects when creating an instance from
        this template (e.g., correlation axis selection), typically with enum
        fields for available choices.

        Returns
        -------
        :
            Pydantic model class for template configuration, or None if the
            template is not yet ready to create instances (e.g., no data available).
        """
        ...

    def get_raw_configuration_model(self) -> type[BaseModel]:
        """
        Get a simple Pydantic model for configuration without enum validation.

        Used for template instantiation where config values are provided directly
        without requiring underlying data to exist yet. This allows templates to
        be instantiated and persisted before data is available.

        Returns
        -------
        :
            Pydantic model class with basic type validation (e.g., strings).
        """
        ...

    def create_workflow_spec(
        self,
        config: BaseModel | dict,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec] | None = None,
    ) -> WorkflowSpec:
        """
        Create a WorkflowSpec from the template configuration.

        The returned spec has a unique WorkflowId that incorporates the
        configuration (e.g., includes the axis name for correlation histograms).

        Parameters
        ----------
        config:
            Configuration as a Pydantic model or dict.
        workflow_registry:
            Optional registry of existing workflow specs. Templates can use this
            to query available outputs (e.g., to find timeseries for correlation
            histograms). If not provided, template-specific defaults apply.

        Returns
        -------
        :
            WorkflowSpec instance ready for registration.
        """
        ...

    def make_instance_id(self, config: BaseModel | dict) -> WorkflowId:
        """
        Generate unique WorkflowId for this instance.

        The ID should incorporate the configuration to ensure uniqueness.
        For example, "correlation/temperature_histogram_1d/v1".

        Parameters
        ----------
        config:
            Configuration as a Pydantic model or dict.

        Returns
        -------
        :
            Unique WorkflowId for this configuration.
        """
        ...

    def make_instance_title(self, config: BaseModel | dict) -> str:
        """
        Generate human-readable title for this instance.

        Displayed in UI to identify this specific workflow instance.

        Parameters
        ----------
        config:
            Configuration as a Pydantic model or dict.

        Returns
        -------
        :
            Human-readable title like 'Temperature Correlation Histogram'.
        """
        ...

    def create_job_executor(
        self,
        config: BaseModel | dict,
        workflow_subscriber: WorkflowSubscriber | None = None,
    ) -> JobExecutor | None:
        """
        Create an executor for this workflow type.

        Templates can provide custom executors for workflows that need special
        execution handling (e.g., frontend-only workflows like correlation
        histograms). Return None to use the default backend executor that sends
        commands via CommandService.

        Parameters
        ----------
        config:
            Configuration as a Pydantic model or dict.
        workflow_subscriber:
            Optional subscriber for workflow availability notifications. Used by
            executors that need to resolve runtime-dependent values (e.g., JobNumber)
            when dependent workflows become available.

        Returns
        -------
        :
            Custom JobExecutor for frontend execution, or None to use the
            default backend executor.
        """
        ...

    def get_available_source_names(self) -> list[str]:
        """
        Get available source names for the workflow configuration UI.

        Returns the list of source names that can be selected when configuring
        a workflow instance. For correlation histograms, this is the list of
        available timeseries that can be correlated.

        Returns
        -------
        :
            List of source names available for selection.
        """
        ...
