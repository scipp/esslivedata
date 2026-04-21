# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections import UserDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ess.livedata.handlers.detector_view_specs import SpectrumViewSpec

import pydantic
import scipp as sc
import scippnexus as snx

from ess.livedata.handlers.workflow_factory import SpecHandle, WorkflowFactory

from .workflow_spec import AuxSources, WorkflowSpec


class SourceMetadata(pydantic.BaseModel):
    """Metadata for a data source (detector, monitor, or timeseries).

    Parameters
    ----------
    title:
        Human-readable title for display in the UI.
    description:
        Longer description shown in tooltips.
    """

    title: str = pydantic.Field(description="Human-readable title for UI display")
    description: str = pydantic.Field(
        default='', description="Longer description for tooltips"
    )


@dataclass
class LogicalViewConfig:
    """Configuration for a single logical detector view."""

    name: str
    title: str
    description: str
    source_names: list[str]
    transform: Callable[[sc.DataArray, str], sc.DataArray] | None
    roi_support: bool = True
    output_ndim: int | None = None
    reduction_dim: str | list[str] | None = None
    spectrum_view: SpectrumViewSpec | None = None


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
    source_metadata: dict[str, SourceMetadata] = field(default_factory=dict)
    _detector_numbers: dict[str, sc.Variable] = field(default_factory=dict)
    _nexus_file: str | None = None
    active_namespace: str | None = None
    _detector_group_names: dict[str, str] = field(default_factory=dict)
    _timeseries_workflow_handle: SpecHandle | None = field(default=None, init=False)
    _logical_views: list[LogicalViewConfig] = field(default_factory=list, init=False)
    _logical_view_handles: dict[str, SpecHandle] = field(
        default_factory=dict, init=False
    )
    _pixellated_monitors: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        """Auto-register standard workflow specs based on instrument metadata."""
        from ess.livedata.handlers.timeseries_workflow_specs import (
            register_timeseries_workflow_specs,
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

    def configure_pixellated_monitor(
        self,
        name: str,
        detector_number: sc.Variable | None = None,
    ) -> None:
        """Mark a monitor as pixellated (has meaningful per-pixel event IDs).

        This tells the adapter to emit ``DetectorEvents`` (preserving pixel_id)
        instead of plain ``MonitorEvents`` for this source.

        Parameters
        ----------
        name
            Name of the monitor (must be in self.monitors).
        detector_number
            Optional explicit detector_number array. If not provided,
            ``load_factories`` will attempt to load it from the NeXus file.
        """
        if name not in self.monitors:
            raise ValueError(
                f"Source '{name}' not in declared monitors. "
                f"Available monitors: {self.monitors}"
            )
        self._pixellated_monitors.add(name)
        if detector_number is not None:
            self._detector_numbers[name] = detector_number

    @property
    def pixellated_monitor_sources(self) -> frozenset[str]:
        """Source names of monitors registered as pixellated."""
        return frozenset(self._pixellated_monitors)

    def get_source_title(self, source_name: str) -> str:
        """Get display title for a source, falling back to source_name.

        Parameters
        ----------
        source_name:
            Internal source name (e.g., detector name, monitor name).

        Returns
        -------
        :
            Human-readable title if defined, otherwise the source_name itself.
        """
        if metadata := self.source_metadata.get(source_name):
            return metadata.title
        return source_name

    def get_source_description(self, source_name: str) -> str:
        """Get description for a source.

        Parameters
        ----------
        source_name:
            Internal source name (e.g., detector name, monitor name).

        Returns
        -------
        :
            Description if defined, otherwise an empty string.
        """
        if metadata := self.source_metadata.get(source_name):
            return metadata.description
        return ''

    def add_logical_view(
        self,
        *,
        name: str,
        title: str,
        description: str,
        source_names: Sequence[str],
        transform: Callable[[sc.DataArray, str], sc.DataArray] | None = None,
        namespace: str = 'detector_data',
        roi_support: bool = True,
        output_ndim: int | None = None,
        reduction_dim: str | list[str] | None = None,
        spectrum_view: SpectrumViewSpec | None = None,
    ) -> SpecHandle:
        """
        Register a logical detector view.

        This registers the view spec immediately (lightweight) and stores the
        configuration for later factory attachment during load_factories().

        Parameters
        ----------
        name:
            Unique name for the view within the given namespace.
        title:
            Human-readable title for the view.
        description:
            Description of the view.
        source_names:
            List of source names this view applies to.
        transform:
            Function that transforms raw detector data to the view output.
            Signature: ``(da: DataArray, source_name: str) -> DataArray``.
            The ``source_name`` identifies which detector bank the data is from,
            allowing a single transform to handle multiple banks with different
            parameters (e.g., different fold sizes).
            If reduction_dim is specified, the transform should NOT include
            summing - that is handled separately to enable proper ROI index mapping.
            If None, identity (no reshaping).
        namespace:
            Service namespace this view belongs to. Determines which service
            runs the workflow (e.g. ``'detector_data'`` or ``'monitor_data'``).
        roi_support:
            Whether ROI selection is supported for this view.
        output_ndim:
            Number of dimensions for spatial outputs.
        reduction_dim:
            Dimension(s) to sum over after applying transform. If specified,
            enables proper ROI support by tracking which input pixels contribute
            to each output pixel.
        spectrum_view:
            Optional ``SpectrumViewSpec`` enabling a ``spectrum_view`` output
            derived from the cumulative accumulated histogram via a
            per-instrument transform.

        Returns
        -------
        :
            Handle for the registered spec.
        """
        from ess.livedata.handlers.detector_view_specs import (
            DetectorROIAuxSources,
            make_detector_view_outputs,
            make_detector_view_params,
        )

        outputs = make_detector_view_outputs(
            output_ndim, roi_support=roi_support, spectrum_view=spectrum_view
        )
        params = make_detector_view_params(spectrum_view=spectrum_view)
        handle = self.register_spec(
            namespace=namespace,
            name=name,
            version=1,
            title=title,
            description=description,
            source_names=list(source_names),
            aux_sources=DetectorROIAuxSources() if roi_support else None,
            params=params,
            outputs=outputs,
        )
        self._logical_view_handles[name] = handle
        self._logical_views.append(
            LogicalViewConfig(
                name=name,
                title=title,
                description=description,
                source_names=list(source_names),
                transform=transform,
                roi_support=roi_support,
                output_ndim=output_ndim,
                reduction_dim=reduction_dim,
                spectrum_view=spectrum_view,
            )
        )
        return handle

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
        aux_sources: AuxSources | None = None,
        outputs: type[Any],
        reset_on_run_transition: bool = True,
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
            Optional declarative auxiliary source definitions. If provided,
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
            reset_on_run_transition=reset_on_run_transition,
        )
        return self.workflow_factory.register_spec(spec)

    def load_factories(self) -> None:
        """
        Load and initialize instrument-specific factories.

        This method:
        1. Imports the instrument package (lightweight - just specs)
        2. Auto-attaches timeseries factory if specs were registered
        3. Auto-attaches logical view factories if views were registered
        4. Calls instrument-specific setup_factories(self)
        5. Auto-loads detector_numbers from nexus for unconfigured detectors
        """
        import importlib

        module = importlib.import_module(f'ess.livedata.config.instruments.{self.name}')

        if self._timeseries_workflow_handle is not None:
            from ess.livedata.handlers.timeseries_handler import (
                TimeseriesStreamProcessor,
            )

            self._timeseries_workflow_handle.attach_factory()(
                TimeseriesStreamProcessor.create_workflow
            )

        if self._logical_views:
            from ess.livedata.handlers.detector_view import (
                DetectorViewFactory,
                InstrumentDetectorSource,
            )
            from ess.livedata.handlers.detector_view import (
                LogicalViewConfig as ScilineLogicalViewConfig,
            )

            for config in self._logical_views:
                handle = self._logical_view_handles[config.name]
                view_config = ScilineLogicalViewConfig(
                    transform=config.transform,
                    reduction_dim=config.reduction_dim,
                    roi_support=config.roi_support,
                    spectrum_view=config.spectrum_view,
                )
                factory = DetectorViewFactory(
                    data_source=InstrumentDetectorSource(self),
                    view_config=view_config,
                )
                handle.attach_factory()(factory.make_workflow)

        if hasattr(module, 'setup_factories'):
            module.setup_factories(self)

        for name in (*self.detector_names, *self._pixellated_monitors):
            if name not in self._detector_numbers:
                try:
                    self._load_detector_from_nexus(name)
                except (ValueError, KeyError):
                    # NeXus file not available, or detector_number not found at
                    # the expected path (e.g., monitors lack a detector_number
                    # dataset — they must provide it via configure_pixellated_monitor)
                    pass


instrument_registry = InstrumentRegistry()
