# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections import UserDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import scipp as sc
import scippnexus as snx

from ess.livedata.handlers.workflow_factory import SpecHandle, WorkflowFactory

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
    detector_names: list[str] = field(default_factory=list)
    monitors: list[str] = field(default_factory=list)
    workflow_factory: WorkflowFactory = field(default_factory=WorkflowFactory)
    f144_attribute_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    _detector_numbers: dict[str, sc.Variable] = field(default_factory=dict)
    _nexus_file: str | None = None
    active_namespace: str | None = None
    _detector_group_names: dict[str, str] = field(default_factory=dict)
    _monitor_workflow_handle: SpecHandle | None = field(default=None, init=False)
    _timeseries_workflow_handle: SpecHandle | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Auto-register standard workflow specs based on instrument metadata."""
        from ess.livedata.handlers.monitor_workflow_specs import (
            register_monitor_workflow_specs,
        )
        from ess.livedata.handlers.timeseries_workflow_specs import (
            register_timeseries_workflow_specs,
        )

        self._monitor_workflow_handle = register_monitor_workflow_specs(
            instrument=self, source_names=self.monitors
        )

        timeseries_names = list(self.f144_attribute_registry.keys())
        self._timeseries_workflow_handle = register_timeseries_workflow_specs(
            instrument=self, source_names=timeseries_names
        )

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

    def get_detector_group_name(self, name: str) -> str:
        """
        Get the group name for a detector, defaulting to the detector name.

        If the NXdetector is inside an NXdetector_group, this returns the combination of
        the group name and the detector name. Otherwise, just the detector name.
        """
        return self._detector_group_names.get(name, name)

    def configure_detector(
        self,
        name: str,
        detector_number: sc.Variable | None = None,
        *,
        detector_group_name: str | None = None,
    ) -> None:
        """
        Configure detector-specific metadata.

        Parameters
        ----------
        name
            Name of the detector (must be in self.detector_names).
        detector_number
            Optional explicit detector_number array (e.g., computed arrays for NMX).
        detector_group_name
            Optional detector group name for nexus file loading.
        """
        if name not in self.detector_names:
            raise ValueError(
                f"Detector {name} not in declared detector_names. "
                f"Available detectors: {self.detector_names}"
            )
        if detector_number is not None:
            self._detector_numbers[name] = detector_number
            return
        if detector_group_name is not None:
            group_name = f'{detector_group_name}/{name}'
            self._detector_group_names[name] = group_name

    def _load_detector_from_nexus(self, name: str) -> None:
        """Load detector_number from nexus file."""
        candidate = snx.load(
            self.nexus_file,
            root=f'entry/instrument/{self.get_detector_group_name(name)}/detector_number',
        )
        if not isinstance(candidate, sc.Variable):
            raise ValueError(
                f"Detector {name} not found in {self.nexus_file}. "
                "Please provide a detector_number explicitly via configure_detector()."
            )
        self._detector_numbers[name] = candidate

    def get_detector_number(self, name: str) -> sc.Variable:
        return self._detector_numbers[name]

    def register_spec(
        self,
        *,
        namespace: str = 'data_reduction',
        name: str,
        version: int,
        title: str,
        description: str = '',
        source_names: Sequence[str] | None = None,
        params: type[Any] | None = None,
        aux_sources: type[Any] | None = None,
        outputs: type[Any],
    ) -> SpecHandle:
        """
        Register workflow spec, return handle for later factory attachment.

        This is the first phase of two-phase registration. The spec is registered
        with explicit parameters and a handle is returned that can be used later
        to attach the factory implementation.

        Parameters
        ----------
        namespace:
            Namespace for the workflow (default: 'data_reduction').
        name:
            Name to register the workflow under.
        version:
            Version of the workflow. This is used to differentiate between different
            versions of the same workflow.
        title:
            Title of the workflow. This is used for display in the UI.
        description:
            Optional description of the workflow.
        source_names:
            Optional list of source names that the workflow can handle. This is used to
            create a workflow specification.
        params:
            Optional Pydantic model class defining workflow parameters. Must be
            explicit (not inferred from factory).
        aux_sources:
            Optional Pydantic model class defining auxiliary data sources. If provided,
            this will be used for validation and UI generation. The auxiliary source
            configuration is handled by the Job layer and is not passed to the workflow
            factory function.
        outputs:
            Pydantic model class defining workflow outputs with metadata.
            Field names should be simplified identifiers (e.g., 'i_of_d_two_theta')
            that match keys returned by workflow.finalize(). Field metadata (title,
            description) provides human-readable information for the UI.

        Returns
        -------
        Handle for attaching factory later.
        """
        spec = WorkflowSpec(
            instrument=self.name,
            namespace=namespace,
            name=name,
            version=version,
            title=title,
            description=description,
            source_names=list(source_names or []),
            params=params,
            aux_sources=aux_sources,
            outputs=outputs,
        )
        return self.workflow_factory.register_spec(spec)

    def load_factories(self) -> None:
        """
        Load and initialize instrument-specific factories.

        This method:
        1. Imports the instrument package (lightweight - just specs)
        2. Auto-attaches standard factories if specs were registered
        3. Calls instrument-specific setup_factories(self)
        4. Auto-loads detector_numbers from nexus for unconfigured detectors
        """
        import importlib

        module = importlib.import_module(f'ess.livedata.config.instruments.{self.name}')

        if self._monitor_workflow_handle is not None:
            from ess.livedata.handlers.monitor_data_handler import (
                MonitorStreamProcessor,
            )

            self._monitor_workflow_handle.attach_factory()(
                MonitorStreamProcessor.create_workflow
            )

        if self._timeseries_workflow_handle is not None:
            from ess.livedata.handlers.timeseries_handler import (
                TimeseriesStreamProcessor,
            )

            self._timeseries_workflow_handle.attach_factory()(
                TimeseriesStreamProcessor.create_workflow
            )

        if hasattr(module, 'setup_factories'):
            module.setup_factories(self)

        for name in self.detector_names:
            if name not in self._detector_numbers:
                try:
                    self._load_detector_from_nexus(name)
                except ValueError:
                    # Nexus file not available or detector not in file
                    pass


instrument_registry = InstrumentRegistry()
