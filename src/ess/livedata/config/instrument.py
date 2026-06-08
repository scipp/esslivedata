# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections import UserDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ess.livedata.handlers.detector_view_specs import SpectrumViewSpec

import pydantic
import scipp as sc
import scippnexus as snx

from ess.livedata.handlers.workflow_factory import (
    SpecHandle,
    WorkflowFactory,
)

from .stream import ChainPatchBinding, ContextBinding, Device, F144Stream, Stream
from .value_log import ValueLog
from .workflow_spec import (
    DETECTORS,
    REDUCTION,
    AuxSources,
    WorkflowGroup,
    WorkflowId,
    WorkflowSpec,
)


def _is_chain_patch(binding: ContextBinding) -> bool:
    """A context binding whose ``workflow_key`` is a chain-patching.

    ValueLog subclass.
    """
    key = binding.workflow_key
    return isinstance(key, type) and issubclass(key, ValueLog)


DEFAULT_DIM_TITLES: dict[str, str] = {
    'wavelength': 'λ',
    'time_of_arrival': 'Time of arrival',
    'time_of_flight': 'Time of flight',
    'tof': 'Time of flight',
    'Q': 'Q',
    'dspacing': 'd-spacing',
    'two_theta': '2θ',
}
"""Default display titles for canonical coord/dim names, shared across instruments.

Per-instrument :attr:`Instrument.dim_titles` entries take precedence."""


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
    #: Disk-chopper component names (as in the NeXus geometry artifact). Single
    #: source of truth for the wavelength-LUT factory (which assembles
    #: ``DiskChoppers`` and declares per-chopper setpoint context bindings) and
    #: the ``ChopperSynthesizer`` wired into the timeseries service.
    choppers: list[str] = field(default_factory=list)
    #: Plateau-detection tolerance for chopper delay readbacks, in the delay
    #: stream's own unit (ns for LOKI). Used by ``ChopperSynthesizer`` for both
    #: noise rejection (rolling-window std must be below this) and change
    #: detection (drift since the last lock). 1 us is a tight default suitable
    #: for sub-degree phase tracking on typical ESS choppers; loosen
    #: per-instrument once real readback noise is measured.
    chopper_delay_atol: float = 1000.0
    workflow_factory: WorkflowFactory = field(default_factory=WorkflowFactory)
    streams: dict[str, Stream] = field(default_factory=dict)
    context_bindings: list[ContextBinding] = field(default_factory=list)
    source_metadata: dict[str, SourceMetadata] = field(default_factory=dict)
    dim_titles: dict[str, str] = field(default_factory=dict)
    _detector_numbers: dict[str, sc.Variable] = field(default_factory=dict)
    _nexus_file: str | None = None
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

        from .chopper import declare_chopper_setpoint_streams

        # Choppers are an instrument concern: declaring them implies the
        # synthetic delay_setpoint streams the ChopperSynthesizer emits. Done
        # before the f144 snapshot below so they register as timeseries sources.
        declare_chopper_setpoint_streams(self.streams, self.choppers)

        for binding in self.context_bindings:
            self._validate_binding_stream_name(binding)
        self._timeseries_workflow_handle = register_timeseries_workflow_specs(
            instrument=self, source_names=self._timeseries_source_names()
        )

    def _timeseries_source_names(self) -> list[str]:
        """Plain f144 streams plus merged Device streams, minus device substreams.

        Device substreams are suppressed by :class:`DeviceSynthesizer` and must
        not appear as standalone timeseries entries; the merged Device stream
        takes their place. Sorted alphabetically for stable UI ordering.
        """
        suppressed = {n for d in self.devices.values() for n in d.substream_names}
        return sorted((set(self.f144_streams) - suppressed) | set(self.devices))

    @property
    def f144_streams(self) -> dict[str, F144Stream]:
        """Subset of :attr:`streams` carrying f144 (NXlog) data."""
        return {
            name: stream
            for name, stream in self.streams.items()
            if isinstance(stream, F144Stream)
        }

    @property
    def devices(self) -> dict[str, Device]:
        """Subset of :attr:`streams` carrying synthesised :class:`Device` streams."""
        return {
            name: stream
            for name, stream in self.streams.items()
            if isinstance(stream, Device)
        }

    def _validate_binding_stream_name(self, binding: ContextBinding) -> None:
        if binding.stream_name not in self.streams:
            raise ValueError(
                f"ContextBinding references unknown stream "
                f"{binding.stream_name!r}; declared streams: "
                f"{sorted(self.streams)}"
            )
        if _is_chain_patch(binding):
            self.chain_patch_path(binding)

    def chain_patch_path(self, binding: ContextBinding) -> str:
        """Resolve the NeXus chain entry path patched by a chain-patch binding.

        The patch target is the ``nexus_path`` of the f144 RBV substream the
        binding sources from: for a :class:`Device` stream, the RBV is
        ``streams[stream.value]``; for a plain :class:`F144Stream` it is the
        stream itself. The chain entry and the f144 source-of-truth must
        live at the same NeXus path — the geometry artifact writes the
        placeholder NXlog at the same path the live stream targets — so
        this single field is the source of truth for both.
        """
        stream = self.streams[binding.stream_name]
        rbv = self.streams[stream.value] if isinstance(stream, Device) else stream
        if not isinstance(rbv, F144Stream):
            raise ValueError(
                f"Chain-patch binding {binding.stream_name!r} resolves to "
                f"{type(rbv).__name__}, not F144Stream; chain-patch sources "
                "must carry f144 NXlog payloads"
            )
        if rbv.nexus_path is None:
            raise ValueError(
                f"Chain-patch binding {binding.stream_name!r} resolves to an "
                f"F144Stream with no nexus_path; set nexus_path on the parsed "
                "stream entry so the chain entry path can be derived"
            )
        return rbv.nexus_path

    def add_context_binding(
        self,
        *,
        stream_name: str,
        dependent_sources: Iterable[str],
        workflow_key: Any,
    ) -> None:
        """Register a stream as a context binding for one or more specs.

        Use from ``setup_factories`` to keep heavy Sciline-key imports out
        of ``specs.py``. The stream value is bound to ``workflow_key`` on
        the target pipeline via ``set_context``. Inputs declared at
        construction time go through the same validation in
        ``__post_init__``.

        Chain-patching bindings (live geometry from motion logs) pass a
        :class:`~ess.livedata.config.value_log.ValueLog` subclass as
        ``workflow_key``: :attr:`chain_patch_bindings` collects them by
        ``issubclass`` and resolves the NeXus ``depends_on`` chain path from
        ``stream_name`` (see :meth:`chain_patch_path`) so the wiring step can
        route the value via a fused per-component patched-chain provider.
        """
        binding = ContextBinding(
            stream_name=stream_name,
            dependent_sources=frozenset(dependent_sources),
            workflow_key=workflow_key,
        )
        self._validate_binding_stream_name(binding)
        self.context_bindings.append(binding)

    @property
    def chain_patch_bindings(self) -> list[ChainPatchBinding]:
        """Instrument-scope chain-patch bindings resolved for transform wiring.

        Selects every instrument-scope chain-patch :class:`ContextBinding` (one
        whose ``workflow_key`` is a :class:`ValueLog` subclass) and resolves its
        NeXus transform path via :meth:`chain_patch_path`. Spec-scope bindings
        are not consulted: chain-patch contexts are required to live at
        instrument scope.

        The result is self-contained data the routing layer hands to
        :func:`~ess.livedata.handlers.dynamic_transforms.wire_dynamic_transforms`,
        so the wiring step needs no access to the instrument's stream topology.
        """
        return [
            ChainPatchBinding(
                stream_name=binding.stream_name,
                transform_path=self.chain_patch_path(binding),
                workflow_key=binding.workflow_key,
                dependent_sources=binding.dependent_sources,
            )
            for binding in self.context_bindings
            if _is_chain_patch(binding)
        ]

    def resolve_context_keys(
        self, workflow_id: WorkflowId, source_name: str
    ) -> dict[str, Any]:
        """Resolve the ``ContextBinding`` mapping for a ``(spec, source)`` pair.

        Matches instrument- and spec-scope :class:`ContextBinding` records whose
        ``dependent_sources`` include ``source_name`` and returns
        ``{stream_name: workflow_key}``. ``skip_instrument_contexts`` filters out
        instrument-scope entries — a spec that explicitly declares a binding
        cannot opt out of it via the flag. Context wire names equal their stream
        names, so the returned keys double as the set of gating context streams.
        """
        registration = self.workflow_factory.registration(workflow_id)
        if registration is None:
            return {}
        instrument_bindings = (
            [] if registration.skip_instrument_contexts else self.context_bindings
        )
        return {
            binding.stream_name: binding.workflow_key
            for binding in (*instrument_bindings, *registration.context_bindings)
            if source_name in binding.dependent_sources
        }

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

    def get_dim_title(self, dim: str) -> str:
        """Get display title for a coord/dim name.

        Per-instrument ``dim_titles`` take precedence over
        :data:`DEFAULT_DIM_TITLES`. Falls back to the raw ``dim`` if no
        mapping is defined.
        """
        return self.dim_titles.get(dim, DEFAULT_DIM_TITLES.get(dim, dim))

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
        group: WorkflowGroup = DETECTORS,
        service: str | None = None,
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
            Unique name for the view within the given group.
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
        group:
            Display-oriented :class:`WorkflowGroup` this view belongs to.
        service:
            Name of the backend service responsible for running this workflow.
            Defaults to ``group.name``.
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
            group=group,
            service=service,
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
        group: WorkflowGroup = REDUCTION,
        service: str | None = None,
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
        group:
            Display-oriented :class:`WorkflowGroup` for the workflow
            (default: ``REDUCTION``).
        service:
            Name of the backend service responsible for running this workflow.
            Defaults to ``group.name``.
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
            group=group,
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
        return self.workflow_factory.register_spec(spec, service=service)

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

        self._validate_binding_dependent_sources()
        self._validate_context_binding_wire_name_collisions()
        self._validate_chain_patch_value_log_uniqueness()

        for name in (*self.detector_names, *self._pixellated_monitors):
            if name not in self._detector_numbers:
                try:
                    self._load_detector_from_nexus(name)
                except (ValueError, KeyError):
                    # NeXus file not available, or detector_number not found at
                    # the expected path (e.g., monitors lack a detector_number
                    # dataset — they must provide it via configure_pixellated_monitor)
                    pass

    def _validate_binding_dependent_sources(self) -> None:
        """Raise if any binding lists a source name no registered spec advertises."""
        if not self.context_bindings:
            return
        known_sources: set[str] = set()
        for spec in self.workflow_factory.values():
            known_sources.update(spec.source_names)
        for binding in self.context_bindings:
            unknown = binding.dependent_sources - known_sources
            if unknown:
                raise ValueError(
                    f"ContextBinding for stream {binding.stream_name!r} lists "
                    f"unknown dependent_sources {sorted(unknown)}; no registered "
                    f"spec advertises these source_names"
                )

    def _validate_chain_patch_value_log_uniqueness(self) -> None:
        """Raise if chain-patch ``(stream_name, workflow_key)`` is not bijective.

        Sciline keys identify parameters by class, and
        :func:`~ess.livedata.handlers.dynamic_transforms.wire_dynamic_transforms`
        dedups bindings by ``stream_name`` with last-write-wins semantics. Both
        directions must therefore be unique:

        - Two streams sharing one :class:`ValueLog` subclass would silently
          collapse into a single Sciline node, merging unrelated streams.
        - Two :class:`ValueLog` subclasses for one stream would silently
          drop one binding in the chain-patch provider.

        Repeat entries with identical ``(stream_name, workflow_key)`` (which
        can occur if ``load_factories`` is called multiple times in a
        long-lived process) are allowed: they describe the same binding.
        """
        by_key: dict[type, str] = {}
        by_stream: dict[str, type] = {}
        all_bindings = list(self.context_bindings)
        for reg in self.workflow_factory.registrations():
            all_bindings.extend(reg.context_bindings)
        for binding in all_bindings:
            if not _is_chain_patch(binding):
                continue
            previous_stream = by_key.get(binding.workflow_key)
            if previous_stream is not None and previous_stream != binding.stream_name:
                raise ValueError(
                    f"ValueLog subclass {binding.workflow_key.__name__!r} is shared "
                    f"by streams {previous_stream!r} and {binding.stream_name!r}; "
                    "each chain-patch context must declare its own ValueLog "
                    "subclass to avoid Sciline node collisions"
                )
            previous_key = by_stream.get(binding.stream_name)
            if previous_key is not None and previous_key is not binding.workflow_key:
                raise ValueError(
                    f"Stream {binding.stream_name!r} has conflicting chain-patch "
                    f"declarations: ValueLog subclasses "
                    f"{previous_key.__name__!r} and {binding.workflow_key.__name__!r}"
                )
            by_key[binding.workflow_key] = binding.stream_name
            by_stream[binding.stream_name] = binding.workflow_key

    def _validate_context_binding_wire_name_collisions(self) -> None:
        """Raise if context-stream wire names collide.

        Two collisions are detected:

        - **Instrument-vs-spec.** For every (spec, source) pair where both
          instrument-level and spec-level :class:`ContextBinding` entries
          apply, the ``stream_name`` (which equals the wire name) must be
          unique across the two scopes.
        - **Context-vs-aux.** A context wire name must not match any
          ``aux_sources`` field name on the spec: at ``JobFactory.create``
          time the context and aux mappings are merged into a single
          field→wire dict, and a key clash would silently overwrite the
          aux entry.
        """
        for reg in self.workflow_factory.registrations():
            spec = reg.spec
            aux_field_names: set[str] = (
                set(spec.aux_sources.inputs) if spec.aux_sources is not None else set()
            )
            instrument_bindings = (
                [] if reg.skip_instrument_contexts else self.context_bindings
            )
            for source in spec.source_names:
                instrument_names: set[str] = {
                    binding.stream_name
                    for binding in instrument_bindings
                    if source in binding.dependent_sources
                }
                spec_names: set[str] = {
                    binding.stream_name
                    for binding in reg.context_bindings
                    if source in binding.dependent_sources
                }
                scope_collisions = instrument_names & spec_names
                if scope_collisions:
                    raise ValueError(
                        f"ContextBinding stream-name collision for spec "
                        f"{spec.name!r} on source {source!r}: "
                        f"{sorted(scope_collisions)} declared at both "
                        f"instrument and spec scope"
                    )
                aux_collisions = (instrument_names | spec_names) & aux_field_names
                if aux_collisions:
                    raise ValueError(
                        f"ContextBinding stream-name collision with aux field "
                        f"for spec {spec.name!r} on source {source!r}: "
                        f"{sorted(aux_collisions)} declared as both a "
                        f"context stream and an aux_sources field; the merged "
                        f"field→wire mapping at JobFactory.create would "
                        f"silently overwrite the aux entry"
                    )


instrument_registry = InstrumentRegistry()
