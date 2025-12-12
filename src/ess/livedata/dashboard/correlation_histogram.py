# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import Any, NewType, TypeVar
from uuid import UUID, uuid4

import numpy as np
import pydantic
import scipp as sc

from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    ResultKey,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.config.workflow_template import JobExecutor, WorkflowSubscriber
from ess.livedata.dashboard.job_orchestrator import JobConfig
from ess.livedata.parameter_models import EdgesModel, make_edges

from .data_service import DataService, SubscriptionId
from .data_subscriber import DataSubscriber, MergingStreamAssembler


class TimeseriesReference(pydantic.BaseModel):
    """
    Stable reference to a timeseries workflow output.

    Unlike ResultKey, this reference does not contain a JobNumber which is
    runtime-dependent. This allows correlation histogram configurations to be
    persisted and restored across workflow restarts.

    The JobNumber is resolved at runtime via subscription to the JobOrchestrator.
    """

    workflow_id: WorkflowId
    source_name: str
    output_name: str

    @classmethod
    def from_result_key(cls, key: ResultKey) -> TimeseriesReference:
        """Extract stable reference from a ResultKey."""
        return cls(
            workflow_id=key.workflow_id,
            source_name=key.job_id.source_name,
            output_name=key.output_name or '',
        )

    def to_result_key(self, job_number: JobNumber) -> ResultKey:
        """Resolve to full ResultKey given a job number."""
        return ResultKey(
            workflow_id=self.workflow_id,
            job_id=JobId(source_name=self.source_name, job_number=job_number),
            output_name=self.output_name if self.output_name else None,
        )


# Unique identifier for processors within the controller
ProcessorId = NewType('ProcessorId', UUID)


class EdgesWithUnit(EdgesModel):
    # Frozen so it cannot be changed in the UI but allows us to display the unit. If we
    # wanted to auto-generate an enum to get a dropdown menu, we probably cannot write
    # generic code below, since we would need to create all the models on the fly.
    unit: str = pydantic.Field(..., description="Unit for the edges", frozen=True)

    def make_edges(self, dim: str) -> sc.Variable:
        return make_edges(model=self, dim=dim, unit=self.unit)


class NormalizationParams(pydantic.BaseModel):
    per_second: bool = pydantic.Field(
        default=False,
        description="Divide data by time bin width to obtain a rate. When enabled, "
        "each histogram bin represents a rate (rather than counts), computed as a mean "
        "instead of a sum over all contributions.",
    )


class CorrelationHistogramParams(pydantic.BaseModel):
    # Do we want a start_time param here as well?
    normalization: NormalizationParams = pydantic.Field(
        default_factory=NormalizationParams,
        title="Normalization",
        description="Options for normalizing the correlation histogram.",
    )


class CorrelationHistogram1dParams(CorrelationHistogramParams):
    x_edges: EdgesWithUnit


class CorrelationHistogram2dParams(CorrelationHistogramParams):
    x_edges: EdgesWithUnit
    y_edges: EdgesWithUnit


class CorrelationHistogramOutputs(WorkflowOutputsBase):
    """Outputs for correlation histogram workflows."""

    histogram: sc.DataArray = pydantic.Field(
        title='Correlation Histogram',
        description='Histogram correlating selected timeseries.',
    )


Model = TypeVar('Model', bound=CorrelationHistogramParams)


def _make_edges_field(name: str, coord: sc.DataArray) -> Any:
    """Create a pydantic Field for EdgesWithUnit with defaults from timeseries data."""
    unit = str(coord.unit)
    low = coord.nanmin().value
    high = np.nextafter(coord.nanmax().value, np.inf)
    return _make_edges_field_from_values(name, unit=unit, start=low, stop=high)


def _make_edges_field_from_values(
    name: str,
    *,
    unit: str | None,
    start: float | None,
    stop: float | None,
) -> Any:
    """Create a pydantic Field for EdgesWithUnit with explicit values.

    If unit/start/stop are None, no default is provided and the user must fill them in.
    """
    if unit is not None and start is not None and stop is not None:
        return pydantic.Field(
            default=EdgesWithUnit(start=start, stop=stop, num_bins=50, unit=unit),
            title=f"{name} bins",
            description=f"Define the bin edges for histogramming in {name}.",
        )
    # No defaults available - field will require user input
    return pydantic.Field(
        title=f"{name} bins",
        description=f"Define the bin edges for histogramming in {name}.",
    )


def _create_dynamic_aux_sources_model(
    field_names: list[str], available_options: list[str]
) -> type[pydantic.BaseModel]:
    """
    Dynamically create a Pydantic model for aux sources with StrEnum fields.

    Parameters
    ----------
    field_names
        Names of the fields (e.g., ['x_param', 'y_param'])
    available_options
        Available values for each field (e.g., list of timeseries names)

    Returns
    -------
    type[pydantic.BaseModel]
        Dynamically created Pydantic model class with StrEnum fields
    """
    # Create a dynamic StrEnum with all available options
    TimeseriesEnum = StrEnum(  # type: ignore[misc]
        'TimeseriesEnum', {option: option for option in available_options}
    )

    # Create fields dict for the Pydantic model
    fields: dict[str, Any] = {}
    for field_name in field_names:
        fields[field_name] = (
            TimeseriesEnum,
            pydantic.Field(
                description=f"Select timeseries for {field_name.replace('_', ' ')}"
            ),
        )

    # Dynamically create the Pydantic model
    DynamicAuxSources = pydantic.create_model(  # type: ignore[call-overload]
        'DynamicAuxSources', **fields
    )
    return DynamicAuxSources


class CorrelationHistogramController:
    """
    Manages correlation histogram processors and their DataService subscriptions.

    Processors are tracked by ProcessorId, allowing them to be added and removed
    during workflow restarts.
    """

    def __init__(self, data_service: DataService[ResultKey, sc.DataArray]) -> None:
        self._data_service = data_service
        self._processors: dict[
            ProcessorId, tuple[CorrelationHistogramProcessor, SubscriptionId]
        ] = {}

    def get_data(self, key: ResultKey) -> sc.DataArray:
        """Get data for a given key."""
        return self._data_service[key]

    def set_data(self, key: ResultKey, data: sc.DataArray) -> None:
        """Set data for a given key."""
        self._data_service[key] = data

    def add_correlation_processor(
        self,
        processor_id: ProcessorId,
        processor: CorrelationHistogramProcessor,
        items: dict[ResultKey, sc.DataArray],
    ) -> None:
        """
        Add a correlation histogram processor with DataService subscription.

        Parameters
        ----------
        processor_id:
            Unique identifier for tracking this processor.
        processor:
            The processor to add.
        items:
            Initial data items for the processor.
        """
        from .extractors import FullHistoryExtractor

        # Create subscriber that merges data and sends to processor.
        # Use FullHistoryExtractor to get complete timeseries history needed for
        # correlation histogram computation.
        assembler = MergingStreamAssembler(set(items))
        extractors = {key: FullHistoryExtractor() for key in items}

        # Create factory that sends initial data to processor and returns it
        def processor_pipe_factory(data: dict[ResultKey, sc.DataArray]):
            processor.send(data)
            return processor

        subscriber = DataSubscriber(assembler, processor_pipe_factory, extractors)
        subscription_id = self._data_service.register_subscriber(subscriber)

        # Track processor with its subscription ID
        self._processors[processor_id] = (processor, subscription_id)

    def remove_processor(self, processor_id: ProcessorId) -> None:
        """
        Remove a processor and unregister its DataService subscription.

        Parameters
        ----------
        processor_id:
            The processor ID to remove.
        """
        if processor_id in self._processors:
            _, subscription_id = self._processors.pop(processor_id)
            self._data_service.unregister_subscriber(subscription_id)

    def get_timeseries(self) -> list[ResultKey]:
        return [key for key, da in self._data_service.items() if _is_timeseries(da)]


def _is_timeseries(da: sc.DataArray) -> bool:
    """Check if data represents a timeseries.

    When DataService uses LatestValueExtractor (default), it returns the latest value
    from a timeseries buffer as a 0D scalar with a time coordinate. This function
    identifies such values as originating from a timeseries.

    Parameters
    ----------
    da:
        DataArray to check.

    Returns
    -------
    :
        True if the data is a 0D scalar with a time coordinate.
    """
    return da.ndim == 0 and 'time' in da.coords


class CorrelationHistogramProcessor:
    """Processes correlation histogram updates when data changes."""

    def __init__(
        self,
        data_key: ResultKey,
        coord_keys: list[ResultKey],
        edges_params: list[EdgesWithUnit],
        normalize: bool,
        result_callback: Callable[[sc.DataArray], None],
    ) -> None:
        self._data_key = data_key
        self._result_callback = result_callback
        self._coords = {key: key.job_id.source_name for key in coord_keys}
        edges = {
            dim: edge.make_edges(dim=dim)
            for dim, edge in zip(self._coords.values(), edges_params, strict=True)
        }
        self._histogrammer = CorrelationHistogrammer(edges=edges, normalize=normalize)

    def send(self, data: dict[ResultKey, sc.DataArray]) -> None:
        """Called when data is updated - processes the correlation histogram."""
        coords = {name: data[key] for key, name in self._coords.items()}
        result = self._histogrammer(data[self._data_key], coords=coords)
        self._result_callback(result)


class CorrelationHistogrammer:
    def __init__(self, edges: dict[str, sc.Variable], normalize: bool = False) -> None:
        self._edges = edges
        self._normalize: bool = normalize

    def __call__(
        self, data: sc.DataArray, coords: dict[str, sc.DataArray]
    ) -> sc.DataArray:
        dependent = data.copy(deep=False)
        if self._normalize:
            times = dependent.coords['time']
            widths = (times[1:] - times[:-1]).to(dtype='float64', unit='s')
            widths = sc.concat([widths, widths.median()], dim='time')
            dependent = dependent / widths
        # Note that this implementation is naive and inefficient as timeseries grow.
        # An alternative approach, streaming only the new data and directly updating
        # only the target bin may need to be considered in the future. This would have
        # the downside of not being able to recreate the histogram for past data though,
        # unless we replay the Kafka topic.
        # For now, if we expect timeseries to not update more than once per second, this
        # should be acceptable.
        for dim in self._edges:
            lut = sc.lookup(
                sc.values(coords[dim])
                if coords[dim].variances is not None
                else coords[dim],
                mode='previous',
            )
            dependent.coords[dim] = lut[dependent.coords['time']]
        if self._normalize:
            return dependent.bin(**self._edges).bins.mean()
        return dependent.hist(**self._edges)


class CorrelationHistogramExecutor:
    """
    Executor for correlation histogram workflows.

    This executor handles frontend-only execution of correlation histograms
    by creating processors that subscribe to DataService updates.

    The executor subscribes to timeseries workflows and resolves
    TimeseriesReferences to ResultKeys when the timeseries workflows
    become available. When a timeseries workflow restarts (new JobNumber),
    the executor tears down existing processors and recreates them with
    the new ResultKeys.
    """

    def __init__(
        self,
        controller: CorrelationHistogramController,
        axis_refs: list[TimeseriesReference],
        source_name_to_key: dict[str, ResultKey],
        workflow_id: WorkflowId,
        ndim: int,
        workflow_subscriber: WorkflowSubscriber,
    ) -> None:
        """
        Initialize the executor.

        Parameters
        ----------
        controller:
            Controller for data access and processor registration.
        axis_refs:
            Stable references to the correlation axis timeseries.
        source_name_to_key:
            Mapping from display names to ResultKeys for data sources.
        workflow_id:
            WorkflowId for creating result keys.
        ndim:
            Dimensionality (1 or 2) for determining param model.
        workflow_subscriber:
            Subscriber for workflow availability notifications.
        """
        self._controller = controller
        self._axis_refs = axis_refs
        self._source_name_to_key = source_name_to_key
        self._workflow_id = workflow_id
        self._ndim = ndim
        self._workflow_subscriber = workflow_subscriber

        # Runtime state
        self._workflow_subscriptions: dict[WorkflowId, SubscriptionId] = {}
        self._active_job_numbers: dict[WorkflowId, JobNumber] = {}
        self._active_processors: set[ProcessorId] = set()
        self._staged_jobs: dict[str, JobConfig] | None = None
        self._correlation_job_number: JobNumber | None = None

    def start_jobs(
        self,
        staged_jobs: dict[str, JobConfig],
        job_number: JobNumber,
    ) -> None:
        """
        Start correlation histogram jobs.

        Subscribes to the timeseries workflows for each axis. Processors are
        created when all required timeseries workflows are available.
        """
        self._staged_jobs = staged_jobs
        self._correlation_job_number = job_number

        # Subscribe to each unique timeseries workflow
        for ref in self._axis_refs:
            if ref.workflow_id not in self._workflow_subscriptions:
                sub_id, _invoked = self._workflow_subscriber.subscribe_to_workflow(
                    ref.workflow_id,
                    lambda jn, wid=ref.workflow_id: self._on_timeseries_available(
                        wid, jn
                    ),
                )
                self._workflow_subscriptions[ref.workflow_id] = sub_id

    def _on_timeseries_available(
        self, workflow_id: WorkflowId, job_number: JobNumber
    ) -> None:
        """
        Called when a timeseries workflow becomes available or restarts.

        If this is a restart (not first availability), tears down existing
        processors and recreates them with the new JobNumber.
        """
        old_job_number = self._active_job_numbers.get(workflow_id)
        self._active_job_numbers[workflow_id] = job_number

        # If this is a restart (not first availability), tear down and rebuild
        if old_job_number is not None:
            self._teardown_processors()

        # Check if all required workflows are now available
        if self._all_axes_available():
            self._create_processors()

    def _all_axes_available(self) -> bool:
        """Check if all axis timeseries workflows have reported availability."""
        required = {ref.workflow_id for ref in self._axis_refs}
        return required <= set(self._active_job_numbers.keys())

    def _create_processors(self) -> None:
        """Create processors for all staged jobs using current axis ResultKeys."""
        if self._staged_jobs is None or self._correlation_job_number is None:
            return

        # Resolve axis references to current ResultKeys
        axis_keys = [
            ref.to_result_key(self._active_job_numbers[ref.workflow_id])
            for ref in self._axis_refs
        ]

        # Parse edges from params
        edges = self._parse_edges(self._staged_jobs)

        # Get normalization setting from first job's params
        first_config = next(iter(self._staged_jobs.values()))
        normalize = first_config.params.get('normalization', {}).get(
            'per_second', False
        )

        # Get axis data
        axes = {key: self._controller.get_data(key) for key in axis_keys}

        for source_name in self._staged_jobs:
            data_key = self._source_name_to_key.get(source_name)
            if data_key is None:
                continue

            data_value = self._controller.get_data(data_key)

            # Generate unique processor ID
            processor_id = ProcessorId(uuid4())

            processor = CorrelationHistogramProcessor(
                data_key=data_key,
                coord_keys=axis_keys,
                edges_params=edges,
                normalize=normalize,
                result_callback=self._create_result_callback(
                    data_key, self._correlation_job_number
                ),
            )
            self._controller.add_correlation_processor(
                processor_id, processor, {data_key: data_value, **axes}
            )
            self._active_processors.add(processor_id)

    def _teardown_processors(self) -> None:
        """Remove all active processors."""
        for processor_id in list(self._active_processors):
            self._controller.remove_processor(processor_id)
        self._active_processors.clear()

    def stop_jobs(self, job_ids: list[JobId]) -> None:
        """Stop all processors and unsubscribe from timeseries workflows."""
        self._teardown_processors()
        for sub_id in self._workflow_subscriptions.values():
            self._workflow_subscriber.unsubscribe(sub_id)
        self._workflow_subscriptions.clear()
        self._active_job_numbers.clear()
        self._staged_jobs = None
        self._correlation_job_number = None

    def _parse_edges(self, staged_jobs: dict[str, JobConfig]) -> list[EdgesWithUnit]:
        """Parse edge parameters from staged job config."""
        # All jobs share the same params, so use the first one
        first_config = next(iter(staged_jobs.values()))
        params = first_config.params

        edges = []
        if 'x_edges' in params:
            edges.append(EdgesWithUnit.model_validate(params['x_edges']))
        if 'y_edges' in params and self._ndim == 2:
            edges.append(EdgesWithUnit.model_validate(params['y_edges']))

        return edges

    def _create_result_callback(
        self, data_key: ResultKey, job_number: JobNumber
    ) -> Callable[[sc.DataArray], None]:
        """Create callback for handling histogram results."""
        result_key = ResultKey(
            workflow_id=self._workflow_id,
            job_id=JobId(
                source_name=data_key.job_id.source_name, job_number=job_number
            ),
            output_name='histogram',
        )

        def callback(result: sc.DataArray) -> None:
            self._controller.set_data(result_key, result)

        return callback


# Verify CorrelationHistogramExecutor implements JobExecutor protocol
_: type[JobExecutor] = CorrelationHistogramExecutor  # type: ignore[type-abstract]


# =============================================================================
# WorkflowTemplate implementations for correlation histograms
# =============================================================================


def _format_result_key_brief(key: ResultKey) -> str:
    """Make a brief representation of a ResultKey."""
    if key.output_name is None:
        return f"{key.job_id.source_name}"
    output = key.output_name.split('.')[-1]
    return f"{key.job_id.source_name}: {output}"


def _format_result_key_full(key: ResultKey) -> str:
    """Make a full representation of a ResultKey, fallback if brief not unique."""
    return f"{key.workflow_id}/{key.job_id}/{key.output_name or 'default'}"


def _make_unique_source_name_mapping(
    keys: list[ResultKey],
) -> dict[str, ResultKey]:
    """Create unique display name to ResultKey mapping."""
    mapping = {_format_result_key_brief(key): key for key in keys}
    if len(mapping) != len(keys):
        mapping = {_format_result_key_full(key): key for key in keys}
    return mapping


def _sanitize_for_id(name: str) -> str:
    """Convert a display name to a valid workflow ID component."""
    return name.lower().replace(' ', '_').replace(':', '_').replace('/', '_')


class CorrelationHistogram1dTemplateConfig(pydantic.BaseModel):
    """Configuration for creating a 1D correlation histogram workflow instance."""

    x_axis: TimeseriesReference = pydantic.Field(
        description="Reference to the timeseries to correlate against (X axis)"
    )
    # Optional fields populated from data when available (for UI flow)
    # or specified directly (for config file / programmatic setup)
    x_unit: str | None = pydantic.Field(
        default=None, description="Unit for the X axis edges"
    )
    x_start: float | None = pydantic.Field(
        default=None, description="Start value for X axis range"
    )
    x_stop: float | None = pydantic.Field(
        default=None, description="Stop value for X axis range"
    )


class CorrelationHistogram2dTemplateConfig(pydantic.BaseModel):
    """Configuration for creating a 2D correlation histogram workflow instance."""

    x_axis: TimeseriesReference = pydantic.Field(
        description="Reference to the timeseries to correlate against (X axis)"
    )
    x_unit: str | None = pydantic.Field(
        default=None, description="Unit for the X axis edges"
    )
    x_start: float | None = pydantic.Field(
        default=None, description="Start value for X axis range"
    )
    x_stop: float | None = pydantic.Field(
        default=None, description="Stop value for X axis range"
    )
    y_axis: TimeseriesReference = pydantic.Field(
        description="Reference to the timeseries to correlate against (Y axis)"
    )
    y_unit: str | None = pydantic.Field(
        default=None, description="Unit for the Y axis edges"
    )
    y_start: float | None = pydantic.Field(
        default=None, description="Start value for Y axis range"
    )
    y_stop: float | None = pydantic.Field(
        default=None, description="Stop value for Y axis range"
    )


class CorrelationHistogramTemplateBase(ABC):
    """
    Base class for correlation histogram workflow templates.

    Templates create WorkflowSpec instances where the correlation axis is baked
    into the workflow identity rather than being a runtime parameter. This allows
    "Temperature Correlation Histogram" and "Pressure Correlation Histogram" to
    be tracked as independent workflows.
    """

    def __init__(
        self,
        controller: CorrelationHistogramController,
    ) -> None:
        """
        Initialize the template.

        Parameters
        ----------
        controller:
            Controller for data access and processor registration.
        """
        self._controller = controller
        self._cached_config_model: type[pydantic.BaseModel] | None = None
        self._source_name_to_key: dict[str, ResultKey] | None = None

    def _get_timeseries(self) -> list[ResultKey]:
        """Get available timeseries from the controller."""
        return self._controller.get_timeseries()

    @property
    @abstractmethod
    def name(self) -> str:
        """Template identifier."""

    @property
    @abstractmethod
    def title(self) -> str:
        """Human-readable title for the template."""

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Dimensionality of the correlation histogram (1 or 2)."""

    @property
    @abstractmethod
    def _axis_field_names(self) -> list[str]:
        """Field names for axis sources in the config model."""

    @abstractmethod
    def _get_axis_names_from_dict(self, config: dict) -> list[str]:
        """Extract axis source names from config dict."""

    @abstractmethod
    def _create_dynamic_model_class(self, config: dict) -> type[Model]:
        """Create dynamic parameter model class with bin edge fields from config."""

    def _get_axis_names(self, config: pydantic.BaseModel | dict) -> list[str]:
        """Extract axis source names from config (model or dict)."""
        if isinstance(config, dict):
            return self._get_axis_names_from_dict(config)
        return self._get_axis_names_from_dict(config.model_dump())

    def _refresh_source_mapping(self) -> None:
        """Refresh the source name to ResultKey mapping."""
        timeseries = self._get_timeseries()
        self._source_name_to_key = _make_unique_source_name_mapping(timeseries)

    def get_source_name_to_key(self) -> dict[str, ResultKey]:
        """Get the current source name to ResultKey mapping."""
        if self._source_name_to_key is None:
            self._refresh_source_mapping()
        return self._source_name_to_key or {}

    def get_source_name_to_reference(self) -> dict[str, TimeseriesReference]:
        """Get mapping from display name to stable TimeseriesReference."""
        return {
            name: TimeseriesReference.from_result_key(key)
            for name, key in self.get_source_name_to_key().items()
        }

    def get_available_source_names(self) -> list[str]:
        """Get available source names for the workflow configuration UI."""
        return list(self.get_source_name_to_key().keys())

    def get_configuration_model(self) -> type[pydantic.BaseModel] | None:
        """
        Get the Pydantic model for UI configuration with enum validation.

        Returns None if not enough timeseries are available. This is used by UI
        widgets to present a dropdown of available timeseries.
        """
        self._refresh_source_mapping()
        source_names = list(self.get_source_name_to_key().keys())

        if len(source_names) < self.ndim:
            return None

        # Dynamically create model with StrEnum for axis selection
        return _create_dynamic_aux_sources_model(self._axis_field_names, source_names)

    def get_raw_configuration_model(self) -> type[pydantic.BaseModel]:
        """
        Get a simple Pydantic model for configuration without enum validation.

        Used for template instantiation where TimeseriesReference is provided
        directly without requiring timeseries to exist yet. Includes optional
        unit/start/stop fields for each axis.
        """
        fields: dict[str, Any] = {}
        for field_name in self._axis_field_names:
            # Extract axis prefix (e.g., 'x' from 'x_axis')
            prefix = field_name.split('_')[0]
            # Required: axis reference
            fields[field_name] = (
                TimeseriesReference,
                pydantic.Field(description=f"Timeseries reference for {field_name}"),
            )
            # Optional: unit and range for this axis
            fields[f'{prefix}_unit'] = (
                str | None,
                pydantic.Field(default=None, description=f"Unit for {prefix} axis"),
            )
            fields[f'{prefix}_start'] = (
                float | None,
                pydantic.Field(default=None, description=f"Start for {prefix} axis"),
            )
            fields[f'{prefix}_stop'] = (
                float | None,
                pydantic.Field(default=None, description=f"Stop for {prefix} axis"),
            )
        return pydantic.create_model(  # type: ignore[call-overload]
            'RawConfig', **fields
        )

    def make_instance_id(self, config: pydantic.BaseModel | dict) -> WorkflowId:
        """Generate unique WorkflowId for this instance."""
        axis_names = self._get_axis_names(config)
        sanitized = '_'.join(_sanitize_for_id(name) for name in axis_names)
        return WorkflowId(
            instrument='frontend',
            namespace='correlation',
            name=f'{sanitized}_histogram_{self.ndim}d',
            version=1,
        )

    def make_instance_title(self, config: pydantic.BaseModel | dict) -> str:
        """Generate human-readable title for this instance."""
        axis_names = self._get_axis_names(config)
        if self.ndim == 1:
            return f'{axis_names[0]} Correlation Histogram'
        else:
            return f'{axis_names[0]} vs {axis_names[1]} Correlation Histogram'

    def create_workflow_spec(self, config: pydantic.BaseModel | dict) -> WorkflowSpec:
        """
        Create a WorkflowSpec from the template configuration.

        The axis is baked into the workflow identity. Source names are left empty
        as they are determined dynamically at job start time from available
        timeseries (excluding the selected axis).

        The params model is dynamically created with unit and range defaults from
        the config. Use `enrich_config_from_data` to populate these before calling
        this method if you want defaults derived from timeseries data.
        """
        axis_names = self._get_axis_names(config)
        workflow_id = self.make_instance_id(config)
        title = self.make_instance_title(config)

        # Create dynamic params from config values (unit/start/stop)
        config_dict = config if isinstance(config, dict) else config.model_dump()
        dynamic_params = self._create_dynamic_model_class(config_dict)

        return WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title=title,
            description=(
                f'{self.ndim}D correlation histogram against {", ".join(axis_names)}'
            ),
            # Source names are determined dynamically at job start time
            source_names=[],
            aux_sources=None,  # Axis is now part of identity, not aux_sources
            params=dynamic_params,
            outputs=CorrelationHistogramOutputs,
        )

    @abstractmethod
    def enrich_config_from_data(self, config: dict) -> dict:
        """
        Populate config with unit and range values from timeseries data.

        Call this before `create_workflow_spec` when you want defaults derived
        from actual data. The enriched config should be persisted so that
        the workflow can be recreated without data being available.

        Parameters
        ----------
        config:
            Basic config with axis names (e.g., {'x_param': 'temperature'})

        Returns
        -------
        :
            Enriched config with unit/start/stop values added.

        Raises
        ------
        KeyError:
            If the specified axis timeseries is not available in DataService.
        """

    @abstractmethod
    def get_axis_refs(
        self, config: pydantic.BaseModel | dict
    ) -> list[TimeseriesReference]:
        """
        Get the TimeseriesReferences for the selected axis sources.

        Parameters
        ----------
        config:
            Configuration containing axis selections (either as TimeseriesReference
            objects or dict representations).

        Returns
        -------
        :
            List of TimeseriesReference objects for all axes in this template.
        """

    def create_job_executor(
        self,
        config: pydantic.BaseModel | dict,
        workflow_subscriber: WorkflowSubscriber | None = None,
    ) -> CorrelationHistogramExecutor:
        """
        Create an executor for correlation histogram workflows.

        Returns a CorrelationHistogramExecutor that handles frontend execution
        by creating processors that subscribe to DataService updates.

        Parameters
        ----------
        config:
            Template configuration containing axis selection.
        workflow_subscriber:
            Subscriber for workflow availability notifications. Required for
            resolving TimeseriesReferences to ResultKeys at runtime.

        Returns
        -------
        :
            Executor for this workflow instance.

        Raises
        ------
        ValueError
            If workflow_subscriber is not provided.
        """
        if workflow_subscriber is None:
            msg = "CorrelationHistogramTemplate requires workflow_subscriber"
            raise ValueError(msg)

        axis_refs = self.get_axis_refs(config)
        workflow_id = self.make_instance_id(config)
        return CorrelationHistogramExecutor(
            controller=self._controller,
            axis_refs=axis_refs,
            source_name_to_key=self.get_source_name_to_key(),
            workflow_id=workflow_id,
            ndim=self.ndim,
            workflow_subscriber=workflow_subscriber,
        )


class CorrelationHistogram1dTemplate(CorrelationHistogramTemplateBase):
    """Template for creating 1D correlation histogram workflow instances."""

    @property
    def name(self) -> str:
        return 'correlation_histogram_1d'

    @property
    def title(self) -> str:
        return '1D Correlation Histogram'

    @property
    def ndim(self) -> int:
        return 1

    @property
    def _axis_field_names(self) -> list[str]:
        return ['x_axis']

    def _get_axis_names_from_dict(self, config: dict) -> list[str]:
        x_axis = config['x_axis']
        # Handle dict (from serialization), TimeseriesReference instances,
        # or string (display name from UI model)
        if isinstance(x_axis, dict):
            return [x_axis['source_name']]
        elif isinstance(x_axis, str):
            # Display name from UI - extract the source_name part if it follows
            # the "source_name: output_name" format, otherwise use as-is
            return [x_axis.split(':')[0].strip()]
        return [x_axis.source_name]

    def get_axis_refs(
        self, config: pydantic.BaseModel | dict
    ) -> list[TimeseriesReference]:
        """Get the TimeseriesReferences for the X axis."""
        config_dict = config if isinstance(config, dict) else config.model_dump()
        x_axis = config_dict['x_axis']
        if isinstance(x_axis, dict):
            return [TimeseriesReference.model_validate(x_axis)]
        elif isinstance(x_axis, str):
            # Display name from UI - look up the reference
            ref_mapping = self.get_source_name_to_reference()
            if x_axis in ref_mapping:
                return [ref_mapping[x_axis]]
            raise KeyError(f"Unknown axis display name: {x_axis}")
        return [x_axis]

    def _create_dynamic_model_class(
        self, config: dict
    ) -> type[CorrelationHistogram1dParams]:
        """Create dynamic parameter model class with bin edge fields from config."""
        # Get axis name - handles dict, TimeseriesReference, or string
        x_axis = config['x_axis']
        if isinstance(x_axis, dict):
            x_name = x_axis['source_name']
        elif isinstance(x_axis, str):
            x_name = x_axis.split(':')[0].strip()
        else:
            x_name = x_axis.source_name

        x_field = _make_edges_field_from_values(
            x_name,
            unit=config.get('x_unit'),
            start=config.get('x_start'),
            stop=config.get('x_stop'),
        )

        class Configured1dParams(CorrelationHistogram1dParams):
            x_edges: EdgesWithUnit = x_field

        return Configured1dParams

    def enrich_config_from_data(self, config: dict) -> dict:
        """Populate config with unit and range values from timeseries data."""
        axis_refs = self.get_axis_refs(config)
        # Resolve TimeseriesReference to ResultKey using current mapping
        mapping = self.get_source_name_to_key()
        x_key = self._resolve_ref_to_key(axis_refs[0], mapping)
        x_data = self._controller.get_data(x_key)

        return {
            **config,
            'x_unit': str(x_data.unit),
            'x_start': float(x_data.nanmin().value),
            'x_stop': float(np.nextafter(x_data.nanmax().value, np.inf)),
        }

    def _resolve_ref_to_key(
        self, ref: TimeseriesReference, mapping: dict[str, ResultKey]
    ) -> ResultKey:
        """Resolve TimeseriesReference to ResultKey using current timeseries mapping."""
        for key in mapping.values():
            if (
                key.workflow_id == ref.workflow_id
                and key.job_id.source_name == ref.source_name
                and (key.output_name or '') == ref.output_name
            ):
                return key
        msg = f"Cannot resolve TimeseriesReference {ref} to current timeseries"
        raise KeyError(msg)


class CorrelationHistogram2dTemplate(CorrelationHistogramTemplateBase):
    """Template for creating 2D correlation histogram workflow instances."""

    @property
    def name(self) -> str:
        return 'correlation_histogram_2d'

    @property
    def title(self) -> str:
        return '2D Correlation Histogram'

    @property
    def ndim(self) -> int:
        return 2

    @property
    def _axis_field_names(self) -> list[str]:
        return ['x_axis', 'y_axis']

    def _get_axis_names_from_dict(self, config: dict) -> list[str]:
        result = []
        for axis_field in ('x_axis', 'y_axis'):
            axis = config[axis_field]
            # Handle dict (from serialization), TimeseriesReference instances,
            # or string (display name from UI model)
            if isinstance(axis, dict):
                result.append(axis['source_name'])
            elif isinstance(axis, str):
                # Display name from UI - extract the source_name part if it follows
                # the "source_name: output_name" format, otherwise use as-is
                result.append(axis.split(':')[0].strip())
            else:
                result.append(axis.source_name)
        return result

    def get_axis_refs(
        self, config: pydantic.BaseModel | dict
    ) -> list[TimeseriesReference]:
        """Get the TimeseriesReferences for X and Y axes."""
        config_dict = config if isinstance(config, dict) else config.model_dump()
        ref_mapping = None  # Lazy load if needed
        refs = []
        for field in ('x_axis', 'y_axis'):
            axis = config_dict[field]
            if isinstance(axis, dict):
                refs.append(TimeseriesReference.model_validate(axis))
            elif isinstance(axis, str):
                # Display name from UI - look up the reference
                if ref_mapping is None:
                    ref_mapping = self.get_source_name_to_reference()
                if axis in ref_mapping:
                    refs.append(ref_mapping[axis])
                else:
                    raise KeyError(f"Unknown axis display name: {axis}")
            else:
                refs.append(axis)
        return refs

    def _create_dynamic_model_class(
        self, config: dict
    ) -> type[CorrelationHistogram2dParams]:
        """Create dynamic parameter model class with bin edge fields from config."""

        def get_axis_name(axis_value) -> str:
            """Get axis name - handles dict, TimeseriesReference, or string."""
            if isinstance(axis_value, dict):
                return axis_value['source_name']
            elif isinstance(axis_value, str):
                return axis_value.split(':')[0].strip()
            else:
                return axis_value.source_name

        x_name = get_axis_name(config['x_axis'])
        y_name = get_axis_name(config['y_axis'])

        x_field = _make_edges_field_from_values(
            x_name,
            unit=config.get('x_unit'),
            start=config.get('x_start'),
            stop=config.get('x_stop'),
        )
        y_field = _make_edges_field_from_values(
            y_name,
            unit=config.get('y_unit'),
            start=config.get('y_start'),
            stop=config.get('y_stop'),
        )

        class Configured2dParams(CorrelationHistogram2dParams):
            x_edges: EdgesWithUnit = x_field
            y_edges: EdgesWithUnit = y_field

        return Configured2dParams

    def enrich_config_from_data(self, config: dict) -> dict:
        """Populate config with unit and range values from timeseries data."""
        axis_refs = self.get_axis_refs(config)
        # Resolve TimeseriesReferences to ResultKeys using current mapping
        mapping = self.get_source_name_to_key()
        x_key = self._resolve_ref_to_key(axis_refs[0], mapping)
        y_key = self._resolve_ref_to_key(axis_refs[1], mapping)
        x_data = self._controller.get_data(x_key)
        y_data = self._controller.get_data(y_key)

        return {
            **config,
            'x_unit': str(x_data.unit),
            'x_start': float(x_data.nanmin().value),
            'x_stop': float(np.nextafter(x_data.nanmax().value, np.inf)),
            'y_unit': str(y_data.unit),
            'y_start': float(y_data.nanmin().value),
            'y_stop': float(np.nextafter(y_data.nanmax().value, np.inf)),
        }

    def _resolve_ref_to_key(
        self, ref: TimeseriesReference, mapping: dict[str, ResultKey]
    ) -> ResultKey:
        """Resolve TimeseriesReference to ResultKey using current timeseries mapping."""
        for key in mapping.values():
            if (
                key.workflow_id == ref.workflow_id
                and key.job_id.source_name == ref.source_name
                and (key.output_name or '') == ref.output_name
            ):
                return key
        msg = f"Cannot resolve TimeseriesReference {ref} to current timeseries"
        raise KeyError(msg)
