# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections import UserDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import scipp as sc
import scippnexus as snx

from ess.livedata.handlers.workflow_factory import Workflow, WorkflowFactory

from .workflow_spec import WorkflowSpec


class InstrumentRegistry(UserDict[str, 'Instrument']):
    """
    Registry for instrument configurations.

    This class is used to register and retrieve instrument configurations
    based on their names. It allows for easy access to the configuration
    settings for different instruments.

    Note that in practice instruments are registered only when their module, creating
    an :py:class:`Instrument`, is imported. ESSlivedata does currently not import all
    instrument modules but only the requested one (since importing can be slow). This
    means that the registry will typically contain only a single instrument.
    """

    def register(self, instrument: Instrument) -> None:
        """Register an instrument configuration."""
        if instrument.name in self:
            raise ValueError(f"Instrument {instrument.name} is already registered.")
        self[instrument.name] = instrument


@dataclass(kw_only=True)
class Instrument:
    """
    Class for instrument configuration.

    This class is used to define the configuration for a specific instrument.
    It includes the stream mapping, processor factory, and other settings
    required for the instrument to function correctly.

    Instances must be explicitly registered with the global registry using
    `instrument_registry.register(instrument)`.
    """

    name: str
    workflow_factory: WorkflowFactory = field(default_factory=WorkflowFactory)
    f144_attribute_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    _detector_numbers: dict[str, sc.Variable] = field(default_factory=dict)
    _nexus_file: str | None = None
    active_namespace: str | None = None
    _detector_group_names: dict[str, str] = field(default_factory=dict)

    @property
    def nexus_file(self) -> str:
        from ess.livedata.handlers.detector_data_handler import (
            get_nexus_geometry_filename,
        )

        if self._nexus_file is None:
            try:
                self._nexus_file = get_nexus_geometry_filename(self.name)
            except ValueError as e:
                raise ValueError(
                    f"Nexus file not set or found for instrument {self.name}."
                ) from e
        return self._nexus_file

    @property
    def detector_names(self) -> list[str]:
        """Get the names of all detectors registered in this instrument."""
        return list(self._detector_numbers.keys())

    def get_detector_group_name(self, name: str) -> str:
        """
        Get the group name for a detector, defaulting to the detector name.

        If the NXdetector is inside an NXdetector_group, this returns the combination of
        the group name and the detector name. Otherwise, just the detector name.
        """
        return self._detector_group_names.get(name, name)

    def add_detector(
        self,
        name: str,
        detector_number: sc.Variable | None = None,
        *,
        detector_group_name: str | None = None,
    ) -> None:
        if detector_number is not None:
            self._detector_numbers[name] = detector_number
            return
        if detector_group_name is not None:
            group_name = f'{detector_group_name}/{name}'
            self._detector_group_names[name] = group_name
        candidate = snx.load(
            self.nexus_file,
            root=f'entry/instrument/{self.get_detector_group_name(name)}/detector_number',
        )
        if not isinstance(candidate, sc.Variable):
            raise ValueError(
                f"Detector {name} not found in {self.nexus_file}. "
                "Please provide a detector_number explicitly."
            )
        self._detector_numbers[name] = candidate

    def get_detector_number(self, name: str) -> sc.Variable:
        return self._detector_numbers[name]

    def register_workflow(
        self,
        *,
        namespace: str = 'data_reduction',
        name: str,
        version: int,
        title: str,
        description: str = '',
        source_names: Sequence[str] | None = None,
    ) -> Callable[[Callable[..., Workflow]], Callable[..., Workflow]]:
        """
        Decorator to register a factory function for creating StreamProcessors.

        This decorator registers a factory function that creates a
        :py:class:`StreamProcessor` for a specific workflow. The decorator automatically
        registers the factory with the processor factory and returns the factory
        function unchanged.

        The factory function may have up to three parameters:
        - `source_name`: The name of the source to process.
        - `params`: A Pydantic model containing parameters for the workflow. The factory
          inspects the type hint of the `params` parameter to determine the correct
          model that the frontend uses to create workflow configuration widgets.
        - `aux_sources`: A Pydantic model containing auxiliary source selections. The
          factory inspects the type hint of the `aux_sources` parameter to determine
          the available auxiliary sources and their configuration options.

        Parameters
        ----------
        name:
            Name to register the workflow under.
        version:
            Version of the factory. This is used to differentiate between different
            versions of the same workflow.
        title:
            Title of the workflow. This is used for display in the UI.
        description:
            Optional description of the factory.
        source_names:
            Optional list of source names that the factory can handle. This is used to
            create a workflow specification.

        Returns
        -------
        Decorator function that registers the factory and returns it unchanged.
        """
        spec = WorkflowSpec(
            instrument=self.name,
            namespace=namespace,
            name=name,
            version=version,
            title=title,
            description=description,
            source_names=list(source_names or []),
            params=None,  # placeholder, filled in from type hint later
            aux_sources=None,  # filled in from type hint later
        )
        return self.workflow_factory.register(spec)


instrument_registry = InstrumentRegistry()
