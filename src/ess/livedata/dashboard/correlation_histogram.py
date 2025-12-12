# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import Any

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
from ess.livedata.config.workflow_template import JobExecutor
from ess.livedata.dashboard.job_orchestrator import JobConfig
from ess.livedata.parameter_models import EdgesModel, make_edges

from .data_service import DataService
from .data_subscriber import DataSubscriber, MergingStreamAssembler


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
    def __init__(self, data_service: DataService[ResultKey, sc.DataArray]) -> None:
        self._data_service = data_service
        self._processors: list[CorrelationHistogramProcessor] = []

    def get_data(self, key: ResultKey) -> sc.DataArray:
        """Get data for a given key."""
        return self._data_service[key]

    def set_data(self, key: ResultKey, data: sc.DataArray) -> None:
        """Set data for a given key."""
        self._data_service[key] = data

    def add_correlation_processor(
        self,
        processor: CorrelationHistogramProcessor,
        items: dict[ResultKey, sc.DataArray],
    ) -> None:
        """Add a correlation histogram processor with DataService subscription."""
        from .extractors import FullHistoryExtractor

        self._processors.append(processor)

        # Create subscriber that merges data and sends to processor.
        # Use FullHistoryExtractor to get complete timeseries history needed for
        # correlation histogram computation.
        # TODO We should update the plotter to operate more efficiently by simply
        # subscribing to the changes. This will likely require a new extractor type as
        # well as changes in the plotter, so we defer this for now.
        assembler = MergingStreamAssembler(set(items))
        extractors = {key: FullHistoryExtractor() for key in items}

        # Create factory that sends initial data to processor and returns it
        def processor_pipe_factory(data: dict[ResultKey, sc.DataArray]):
            processor.send(data)
            return processor

        subscriber = DataSubscriber(assembler, processor_pipe_factory, extractors)
        self._data_service.register_subscriber(subscriber)

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
    """

    def __init__(
        self,
        controller: CorrelationHistogramController,
        axis_keys: list[ResultKey],
        source_name_to_key: dict[str, ResultKey],
        workflow_id: WorkflowId,
        ndim: int,
    ) -> None:
        """
        Initialize the executor.

        Parameters
        ----------
        controller:
            Controller for data access and processor registration.
        axis_keys:
            ResultKeys for the correlation axes (e.g., temperature timeseries).
        source_name_to_key:
            Mapping from display names to ResultKeys for data sources.
        workflow_id:
            WorkflowId for creating result keys.
        ndim:
            Dimensionality (1 or 2) for determining param model.
        """
        self._controller = controller
        self._axis_keys = axis_keys
        self._source_name_to_key = source_name_to_key
        self._workflow_id = workflow_id
        self._ndim = ndim

    def start_jobs(
        self,
        staged_jobs: dict[str, JobConfig],
        job_number: JobNumber,
    ) -> None:
        """Start correlation histogram jobs for all staged sources."""
        # Parse edges from params
        edges = self._parse_edges(staged_jobs)

        # Get normalization setting from first job's params
        first_config = next(iter(staged_jobs.values()))
        normalize = first_config.params.get('normalization', {}).get(
            'per_second', False
        )

        # Get axis data
        axes = {key: self._controller.get_data(key) for key in self._axis_keys}

        for source_name in staged_jobs:
            data_key = self._source_name_to_key.get(source_name)
            if data_key is None:
                continue

            data_value = self._controller.get_data(data_key)

            processor = CorrelationHistogramProcessor(
                data_key=data_key,
                coord_keys=self._axis_keys,
                edges_params=edges,
                normalize=normalize,
                result_callback=self._create_result_callback(data_key, job_number),
            )
            self._controller.add_correlation_processor(
                processor, {data_key: data_value, **axes}
            )

    def stop_jobs(self, job_ids: list[JobId]) -> None:
        """
        Stop running correlation histogram jobs.

        Note: Currently correlation histogram processors cannot be individually
        stopped - they continue running until the dashboard is closed. This is
        a known limitation that would require tracking subscriptions in
        CorrelationHistogramController.
        """
        # TODO: Implement proper cleanup by tracking DataService subscriptions
        # in CorrelationHistogramController and unregistering them here.
        pass

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

    axis_source: str = pydantic.Field(
        description="The timeseries to correlate against (X axis)"
    )


class CorrelationHistogram2dTemplateConfig(pydantic.BaseModel):
    """Configuration for creating a 2D correlation histogram workflow instance."""

    axis_x_source: str = pydantic.Field(
        description="The timeseries to correlate against (X axis)"
    )
    axis_y_source: str = pydantic.Field(
        description="The timeseries to correlate against (Y axis)"
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

        Used for template instantiation where axis names are provided directly
        without requiring timeseries to exist yet.
        """
        fields: dict[str, Any] = {}
        for field_name in self._axis_field_names:
            fields[field_name] = (
                str,
                pydantic.Field(description=f"Axis source name for {field_name}"),
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
        """
        axis_names = self._get_axis_names(config)
        workflow_id = self.make_instance_id(config)
        title = self.make_instance_title(config)

        params = {1: CorrelationHistogram1dParams, 2: CorrelationHistogram2dParams}

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
            params=params[self.ndim],
            outputs=CorrelationHistogramOutputs,
        )

    def get_axis_keys(self, config: pydantic.BaseModel | dict) -> list[ResultKey]:
        """Get the ResultKeys for the selected axis sources."""
        axis_names = self._get_axis_names(config)
        mapping = self.get_source_name_to_key()
        return [mapping[name] for name in axis_names]

    def create_job_executor(
        self, config: pydantic.BaseModel | dict
    ) -> CorrelationHistogramExecutor:
        """
        Create an executor for correlation histogram workflows.

        Returns a CorrelationHistogramExecutor that handles frontend execution
        by creating processors that subscribe to DataService updates.

        Parameters
        ----------
        config:
            Template configuration containing axis selection.

        Returns
        -------
        :
            Executor for this workflow instance.
        """
        axis_keys = self.get_axis_keys(config)
        workflow_id = self.make_instance_id(config)
        return CorrelationHistogramExecutor(
            controller=self._controller,
            axis_keys=axis_keys,
            source_name_to_key=self.get_source_name_to_key(),
            workflow_id=workflow_id,
            ndim=self.ndim,
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
        return ['x_param']

    def _get_axis_names_from_dict(self, config: dict) -> list[str]:
        return [config['x_param']]


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
        return ['x_param', 'y_param']

    def _get_axis_names_from_dict(self, config: dict) -> list[str]:
        return [config['x_param'], config['y_param']]
