# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

import numpy as np
import pydantic
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, JobNumber, ResultKey, WorkflowSpec
from ess.livedata.parameter_models import EdgesModel, make_edges

from .configuration_adapter import ConfigurationAdapter
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


# Note: make_workflow_spec encapsulates workflow configuration in the widely-used
# WorkflowSpec format. In the future, this can be converted to WorkflowConfig and
# submitted to a separate backend service for execution.
def make_workflow_spec(ndim: int) -> WorkflowSpec:
    params = {1: CorrelationHistogram1dParams, 2: CorrelationHistogram2dParams}
    # Note how currently (aux) source names are an empty dummy. The idea is to
    # eventually support a "dynamic" specification of the allowed source names, such
    # that this definition can be shared between backend and frontend. For example, we
    # might specify "any timeseries" or similar.
    return WorkflowSpec(
        instrument='frontend',  # As long as we are running in the frontend
        namespace='correlation',
        name=f'correlation_histogram_{ndim}d',
        title=f'{ndim}D Correlation Histogram',
        version=1,
        description=f'{ndim}D correlation histogram workflow',
        source_names=[],
        aux_sources=None,
        params=params[ndim],
    )


def _make_edges_field(name: str, coord: sc.DataArray) -> Any:
    unit = str(coord.unit)
    low = coord.nanmin().value
    high = np.nextafter(coord.nanmax().value, np.inf)
    return pydantic.Field(
        default=EdgesWithUnit(start=low, stop=high, num_bins=50, unit=unit),
        title=f"{name} bins",
        description=f"Define the bin edges for histogramming in {name}.",
    )


Model = TypeVar('Model', bound=CorrelationHistogramParams)


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


class CorrelationHistogramConfigurationAdapter(ConfigurationAdapter[Model], ABC):
    """
    Combined configuration adapter and workflow controller for correlation histograms.

    This class both generates the UI configuration for correlation histograms and
    handles workflow execution. It implements the ConfigurationAdapter interface
    for UI generation while also managing the actual computation workflow.
    """

    def __init__(self, controller: CorrelationHistogramController) -> None:
        super().__init__(config_state=None)
        self._controller = controller
        self._selected_aux_sources: dict[str, str] | None = None
        self._cached_aux_sources: pydantic.BaseModel | None = None
        # All timeseries except the axis keys can be used as dependent variables. These
        # will thus be shown by the widget in the "source selection" menu.
        timeseries = self._controller.get_timeseries()
        # Try to make unique names for display. This might not be very readable, for
        # discussions on a more general solution see also #430.
        self._source_name_to_key = {
            self._format_brief_result_key(key): key for key in timeseries
        }
        if len(self._source_name_to_key) != len(timeseries):
            self._source_name_to_key = {
                self._format_full_result_key(key): key for key in timeseries
            }
        self._workflow_spec_template = make_workflow_spec(ndim=self.ndim)
        # Cache for dynamically created aux_sources model
        self._aux_sources_model: type[pydantic.BaseModel] | None = None

    def _format_brief_result_key(self, key: ResultKey) -> str:
        """Make a brief representation of a ResultKey."""
        if key.output_name is None:
            return f"{key.job_id.source_name}"
        output = key.output_name.split('.')[-1]
        return f"{key.job_id.source_name}: {output}"

    def _format_full_result_key(self, key: ResultKey) -> str:
        """Make a full representation of a ResultKey, fallback if brief not unique."""
        return f"{key.workflow_id}/{key.job_id}/{key.output_name or 'default'}"

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Dimensionality of the correlation histogram (1 or 2)."""

    @property
    @abstractmethod
    def _aux_field_names(self) -> list[str]:
        """
        Field names for aux sources.

        For example: ['x_param'] for 1D, ['x_param', 'y_param'] for 2D.
        """

    @abstractmethod
    def _create_dynamic_model_class(
        self, coords: dict[str, sc.DataArray]
    ) -> type[Model]:
        """Create dynamic parameter model class with appropriate bin edge fields."""

    @property
    def title(self) -> str:
        return self._workflow_spec_template.title

    @property
    def description(self) -> str:
        return self._workflow_spec_template.description

    @property
    def aux_sources(self) -> type[pydantic.BaseModel] | None:
        """Dynamically create aux_sources model from available timeseries."""
        if self._aux_sources_model is None:
            source_names = list(self._source_name_to_key.keys())
            if len(source_names) < self.ndim:
                # Not enough timeseries available yet
                return None
            self._aux_sources_model = _create_dynamic_aux_sources_model(
                self._aux_field_names, source_names
            )
        return self._aux_sources_model

    def model_class(self) -> type[Model] | None:
        if self._cached_aux_sources is None:
            self._selected_aux_sources = None
            return None

        # Serialize aux sources to get the selected stream names
        aux_dict = self._cached_aux_sources.model_dump(mode='json')

        coords = {
            name: self._controller.get_data(self._source_name_to_key[name])
            for name in aux_dict.values()
        }
        model_class = self._create_dynamic_model_class(coords)

        # Store serialized aux sources for use in start_action
        self._selected_aux_sources = aux_dict
        return model_class

    @property
    def source_names(self) -> list[str]:
        return list(self._source_name_to_key.keys())

    @property
    def initial_source_names(self) -> list[str]:
        # In practice we may have many timeseries. We do not want to auto-populate the
        # selection since typically the user will want to select just one or a few.
        return []

    @property
    def initial_parameter_values(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def _params_to_edges(self, params: Model) -> list[EdgesWithUnit]:
        """Convert parameter model to list of EdgesWithUnit."""

    def start_action(
        self,
        selected_sources: list[str],
        parameter_values: Model,
    ) -> None:
        """
        Execute the correlation histogram workflow with the given parameters.

        Note: Currently the "correlation" jobs run in the frontend process, essentially
        as a postprocessing step when new data arrives. There are considerations around
        moving this into a separate backend service, after the primary services, where
        the WorkflowSpec could be converted to WorkflowConfig for submission.

        Raises
        ------
        ValueError
            If auxiliary sources have not been selected.
        """
        if self._selected_aux_sources is None:
            raise ValueError('Auxiliary sources must be selected before starting')

        data_keys = [
            self._source_name_to_key[source_name] for source_name in selected_sources
        ]
        axis_keys = [
            self._source_name_to_key[source_name]
            for source_name in self._selected_aux_sources.values()
        ]

        data = {key: self._controller.get_data(key) for key in data_keys}
        axes = {key: self._controller.get_data(key) for key in axis_keys}
        job_number = uuid.uuid4()  # New unique job number shared by all workflows

        edges = self._params_to_edges(parameter_values)

        for key, value in data.items():
            processor = CorrelationHistogramProcessor(
                data_key=key,
                coord_keys=axis_keys,
                edges_params=edges,
                normalize=parameter_values.normalization.per_second,
                result_callback=self._create_result_callback(key, job_number),
            )
            self._controller.add_correlation_processor(processor, {key: value, **axes})

    def _create_result_callback(
        self, data_key: ResultKey, job_number: JobNumber
    ) -> Callable[[sc.DataArray], None]:
        """Create callback for handling histogram results."""
        result_key = ResultKey(
            workflow_id=self._workflow_spec_template.get_id(),
            job_id=JobId(
                source_name=data_key.job_id.source_name, job_number=job_number
            ),
            output_name=None,
        )

        def callback(result: sc.DataArray) -> None:
            self._controller.set_data(result_key, result)

        return callback


class CorrelationHistogram1dConfigurationAdapter(
    CorrelationHistogramConfigurationAdapter[CorrelationHistogram1dParams]
):
    @property
    def ndim(self) -> int:
        return 1

    @property
    def _aux_field_names(self) -> list[str]:
        return ['x_param']

    def _create_dynamic_model_class(
        self, coords: dict[str, sc.DataArray]
    ) -> type[CorrelationHistogram1dParams]:
        """Create dynamic parameter model class with appropriate bin edge fields."""
        fields = [_make_edges_field(dim, coord) for dim, coord in coords.items()]

        class Configured1dParams(CorrelationHistogram1dParams):
            x_edges: EdgesWithUnit = fields[0]

        return Configured1dParams

    def _params_to_edges(
        self, params: CorrelationHistogram1dParams
    ) -> list[EdgesWithUnit]:
        return [params.x_edges]


class CorrelationHistogram2dConfigurationAdapter(
    CorrelationHistogramConfigurationAdapter[CorrelationHistogram2dParams]
):
    @property
    def ndim(self) -> int:
        return 2

    @property
    def _aux_field_names(self) -> list[str]:
        return ['x_param', 'y_param']

    def _create_dynamic_model_class(
        self, coords: dict[str, sc.DataArray]
    ) -> type[CorrelationHistogram2dParams]:
        """Create dynamic parameter model class with appropriate bin edge fields."""
        if len(coords) < 2:
            raise ValueError("Two distinct timeseries must be selected for the 2 axes.")
        fields = [_make_edges_field(dim, coord) for dim, coord in coords.items()]

        class Configured2dParams(CorrelationHistogram2dParams):
            x_edges: EdgesWithUnit = fields[0]
            y_edges: EdgesWithUnit = fields[1]

        return Configured2dParams

    def _params_to_edges(
        self, params: CorrelationHistogram2dParams
    ) -> list[EdgesWithUnit]:
        return [params.x_edges, params.y_edges]


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
        self._processors.append(processor)

        # Create subscriber that merges data and sends to processor
        assembler = MergingStreamAssembler(set(items))
        subscriber = DataSubscriber(assembler, processor)
        self._data_service.register_subscriber(subscriber)

    def get_timeseries(self) -> list[ResultKey]:
        return [key for key, da in self._data_service.items() if _is_timeseries(da)]

    def create_1d_config(self) -> CorrelationHistogramConfigurationAdapter:
        """Create configuration adapter for 1D correlation histograms."""
        return CorrelationHistogram1dConfigurationAdapter(controller=self)

    def create_2d_config(self) -> CorrelationHistogramConfigurationAdapter:
        """Create configuration adapter for 2D correlation histograms."""
        return CorrelationHistogram2dConfigurationAdapter(controller=self)


def _is_timeseries(da: sc.DataArray) -> bool:
    return da.dims == ('time',) and 'time' in da.coords


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
