# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
WorkflowRegistryManager - Manages static and dynamically-created workflow specs.

Combines static workflows from instrument configuration with dynamic workflows
created from templates (e.g., correlation histograms). Dynamic workflows are
persisted to the config store for restoration across sessions.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import pydantic

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec
from ess.livedata.config.workflow_template import TemplateInstance, WorkflowTemplate

from .config_store import ConfigStore

# Key used to store template instances in config store
_TEMPLATE_INSTANCES_KEY = '_template_instances'


class WorkflowRegistryManager:
    """
    Manages a combined view of static and dynamically-created workflow specs.

    Static workflows come from the instrument's workflow factory. Dynamic workflows
    are created from templates (e.g., correlation histograms) and are persisted
    to the config store for restoration across sessions.
    """

    def __init__(
        self,
        static_registry: Mapping[WorkflowId, WorkflowSpec],
        config_store: ConfigStore | None,
        templates: Sequence[WorkflowTemplate],
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the registry manager.

        Parameters
        ----------
        static_registry:
            Registry of static workflows from instrument configuration.
        config_store:
            Optional store for persisting template instances.
        templates:
            Available workflow templates for creating dynamic workflows.
        logger:
            Logger instance for logging.
        """
        self._static_registry = dict(static_registry)
        self._config_store = config_store
        self._templates = {t.name: t for t in templates}
        self._logger = logger

        # Dynamic workflows created from templates
        self._dynamic_registry: dict[WorkflowId, WorkflowSpec] = {}
        # Track which workflows came from which template instance
        self._template_instances: dict[WorkflowId, TemplateInstance] = {}

        # Load persisted template instances
        self._load_template_instances()

    def _load_template_instances(self) -> None:
        """Recreate dynamic specs from persisted template instances."""
        if self._config_store is None:
            return

        instances_data = self._config_store.get(_TEMPLATE_INSTANCES_KEY)
        if not instances_data:
            return

        for workflow_id_str, instance_data in instances_data.items():
            try:
                instance = TemplateInstance.model_validate(instance_data)
                template = self._templates.get(instance.template_name)
                if template is None:
                    self._logger.warning(
                        'Template %s not found for persisted instance %s, skipping',
                        instance.template_name,
                        workflow_id_str,
                    )
                    continue

                # Use raw config model - doesn't require timeseries to exist
                config_model = template.get_raw_configuration_model()
                config = config_model.model_validate(instance.config)
                # Pass combined registry so template can query available outputs
                combined_registry = self.get_registry()
                spec = template.create_workflow_spec(config, combined_registry)
                workflow_id = spec.get_id()

                self._dynamic_registry[workflow_id] = spec
                self._template_instances[workflow_id] = instance
                self._logger.info(
                    'Restored template instance %s from %s',
                    workflow_id,
                    instance.template_name,
                )
            except Exception as e:
                self._logger.warning(
                    'Failed to restore template instance %s: %s',
                    workflow_id_str,
                    e,
                )

    def _persist_template_instances(self) -> None:
        """Persist template instances to config store."""
        if self._config_store is None:
            return

        instances_data = {
            str(wid): instance.model_dump(mode='json')
            for wid, instance in self._template_instances.items()
        }
        self._config_store[_TEMPLATE_INSTANCES_KEY] = instances_data

    def register_from_template(
        self, template_name: str, config: dict
    ) -> WorkflowId | None:
        """
        Create and register a workflow spec from a template.

        Template instantiation does not require timeseries data to exist. The
        config is validated with basic type checking (strings for axis names),
        not against available timeseries. Enum validation against available
        timeseries is done by UI widgets, not here.

        Parameters
        ----------
        template_name:
            Name of the template to use.
        config:
            Configuration dict for the template (e.g., {'x_param': 'temperature'}).

        Returns
        -------
        :
            The WorkflowId of the created workflow, or None if registration failed.
        """
        template = self._templates.get(template_name)
        if template is None:
            self._logger.error('Template %s not found', template_name)
            return None

        # Use raw config model for basic validation (no enum check against data)
        config_model = template.get_raw_configuration_model()

        try:
            validated_config = config_model.model_validate(config)
        except pydantic.ValidationError as e:
            self._logger.error('Invalid config for template %s: %s', template_name, e)
            return None

        # Pass combined registry so template can query available outputs
        combined_registry = self.get_registry()
        spec = template.create_workflow_spec(validated_config, combined_registry)
        workflow_id = spec.get_id()

        # Check if already registered
        if (
            workflow_id in self._static_registry
            or workflow_id in self._dynamic_registry
        ):
            self._logger.warning(
                'Workflow %s already registered, not overwriting', workflow_id
            )
            return workflow_id

        self._dynamic_registry[workflow_id] = spec
        self._template_instances[workflow_id] = TemplateInstance(
            template_name=template_name,
            config=config,
        )
        self._persist_template_instances()

        self._logger.info(
            'Registered workflow %s from template %s', workflow_id, template_name
        )
        return workflow_id

    def unregister(self, workflow_id: WorkflowId) -> bool:
        """
        Remove a template-created workflow.

        Parameters
        ----------
        workflow_id:
            The workflow to remove.

        Returns
        -------
        :
            True if the workflow was removed, False if it was a static workflow
            or not found.
        """
        if workflow_id in self._static_registry:
            self._logger.warning('Cannot unregister static workflow %s', workflow_id)
            return False

        if workflow_id not in self._dynamic_registry:
            self._logger.warning(
                'Workflow %s not found in dynamic registry', workflow_id
            )
            return False

        del self._dynamic_registry[workflow_id]
        del self._template_instances[workflow_id]
        self._persist_template_instances()

        self._logger.info('Unregistered dynamic workflow %s', workflow_id)
        return True

    def get_registry(self) -> Mapping[WorkflowId, WorkflowSpec]:
        """
        Get combined view of static + dynamic workflows.

        Returns
        -------
        :
            Combined registry of all workflows.
        """
        return {**self._static_registry, **self._dynamic_registry}

    def is_template_instance(self, workflow_id: WorkflowId) -> bool:
        """
        Check if a workflow was created from a template.

        Parameters
        ----------
        workflow_id:
            The workflow to check.

        Returns
        -------
        :
            True if the workflow was created from a template.
        """
        return workflow_id in self._template_instances

    def get_templates(self) -> Mapping[str, WorkflowTemplate]:
        """
        Get available templates.

        Returns
        -------
        :
            Mapping from template name to template instance.
        """
        return dict(self._templates)

    def get_template_instance(self, workflow_id: WorkflowId) -> TemplateInstance | None:
        """
        Get the template instance info for a dynamically created workflow.

        Parameters
        ----------
        workflow_id:
            The workflow to look up.

        Returns
        -------
        :
            TemplateInstance if the workflow was created from a template,
            None otherwise.
        """
        return self._template_instances.get(workflow_id)
