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

    def add_detector(
        self,
        name: str,
        *,
        detector_number: sc.Variable | None = None,
        detector_group_name: str | None = None,
    ) -> None:
        if detector_number is not None:
            self._detector_numbers[name] = detector_number
            return
        group_name = f'{detector_group_name}/{name}' if detector_group_name else name
        self._detector_group_names[name] = group_name
        if self.name == 'odin':
            import h5py

            with h5py.File(self.nexus_file, 'r+') as f:
                det = f['entry/instrument/event_mode_detectors/timepix3']
                trans = det['transformations/translation']
                trans[...] = 1.0
                det.attrs['axes'] = ['x_pixel_offset', 'y_pixel_offset']
                det.attrs['detector_number_indices'] = [0, 1]
        candidate = snx.load(
            self.nexus_file, root=f'entry/instrument/{group_name}/detector_number'
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
        aux_source_names: Sequence[str] | None = None,
    ) -> Callable[[Callable[..., Workflow]], Callable[..., Workflow]]:
        """
        Decorator to register a factory function for creating StreamProcessors.

        This decorator registers a factory function that creates a
        :py:class:`StreamProcessor` for a specific workflow. The decorator automatically
        registers the factory with the processor factory and returns the factory
        function unchanged.

        The factory function may have two parameters:
        - `source_name`: The name of the source to process.
        - `params`: A Pydantic model containing parameters for the workflow. The factory
          inspects the type hint of the `params` parameter to determine the correct
          model that the frontend uses to create workflow configuration widgets.

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
        aux_source_names:
            List of auxiliary source names that the workflow needs.

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
            aux_source_names=list(aux_source_names or []),
        )
        return self.workflow_factory.register(spec)


instrument_registry = InstrumentRegistry()
