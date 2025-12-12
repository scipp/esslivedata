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

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from .workflow_spec import WorkflowId, WorkflowSpec


class TemplateInstance(BaseModel):
    """
    Persisted template instance configuration.

    Used to recreate dynamic WorkflowSpecs across dashboard restarts.
    """

    template_name: str
    config: dict  # Serialized template configuration


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
        Get the Pydantic model for template configuration.

        Returns None if the template cannot be configured (e.g., missing data).
        The model defines what the user selects when creating an instance from
        this template (e.g., correlation axis selection).

        Returns
        -------
        :
            Pydantic model class for template configuration, or None if the
            template is not yet ready to create instances.
        """
        ...

    def create_workflow_spec(self, config: BaseModel) -> WorkflowSpec:
        """
        Create a WorkflowSpec from the template configuration.

        The returned spec has a unique WorkflowId that incorporates the
        configuration (e.g., includes the axis name for correlation histograms).

        Parameters
        ----------
        config:
            Instance of the model returned by get_configuration_model().

        Returns
        -------
        :
            WorkflowSpec instance ready for registration.
        """
        ...

    def make_instance_id(self, config: BaseModel) -> WorkflowId:
        """
        Generate unique WorkflowId for this instance.

        The ID should incorporate the configuration to ensure uniqueness.
        For example, "correlation/temperature_histogram_1d/v1".

        Parameters
        ----------
        config:
            Instance of the model returned by get_configuration_model().

        Returns
        -------
        :
            Unique WorkflowId for this configuration.
        """
        ...

    def make_instance_title(self, config: BaseModel) -> str:
        """
        Generate human-readable title for this instance.

        Displayed in UI to identify this specific workflow instance.

        Parameters
        ----------
        config:
            Instance of the model returned by get_configuration_model().

        Returns
        -------
        :
            Human-readable title like 'Temperature Correlation Histogram'.
        """
        ...
