# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Plot Composition System

This module implements a layer-based composition model for plots, enabling
multiple visual elements (layers) to be composed on a shared canvas. Each
layer can have its own data source and update behavior.

Key Concepts
------------

**Layer**: A single visual element in a composed plot. Layers are peers that
compose via HoloViews' ``*`` operator. Each layer has:
- A unique name within the composition
- An element type (determines the visual representation)
- A data source (where data comes from)
- Optional styling parameters

**Data Source**: Abstraction for where layer data comes from:
- ``PipelineSource``: Data from reduction pipeline via StreamManager
- ``StaticSource``: Fixed data provided directly (e.g., peak marker positions)

**Element Factory**: Transforms data into HoloViews elements. For pipeline
sources, existing Plotter classes serve this role.

Key Implementation Insights
---------------------------

1. **DynamicMap Composition**: Each layer must be a separate DynamicMap.
   Interactive streams must attach to the DynamicMap, not the element.
   Pattern: ``DynamicMap(func) * DynamicMap(func)`` works correctly.

2. **DynamicMap Truthiness**: HoloViews DynamicMap evaluates to False when
   empty. Use explicit ``is not None`` checks.

3. **Closure Capture**: Panel sessions require explicit variable capture in
   callbacks. Use ``def fn(data, _var=var)`` pattern.

See docs/developer/plans/plot-composition-system.md for detailed design.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import holoviews as hv
import pydantic

from ess.livedata.config.workflow_spec import JobId, JobNumber, ResultKey, WorkflowId

if TYPE_CHECKING:
    from .stream_manager import StreamManager

logger = logging.getLogger(__name__)


# =============================================================================
# Data Sources
# =============================================================================


class DataSource(Protocol):
    """Protocol for layer data sources."""

    def get_result_keys(self) -> list[ResultKey]:
        """Return ResultKeys this source will provide data for.

        Returns
        -------
        :
            List of ResultKeys. Empty list for sources that don't use ResultKeys
            (e.g., StaticSource).
        """
        ...


@dataclass(frozen=True)
class PipelineSource:
    """Data from reduction pipeline via StreamManager subscription.

    This source subscribes to workflow results through the StreamManager,
    receiving updates as data flows through the reduction pipeline.

    Parameters
    ----------
    workflow_id:
        The workflow producing the data.
    job_number:
        The job number for this subscription.
    source_names:
        List of source names to subscribe to.
    output_name:
        Name of the output in workflow results.
    """

    workflow_id: WorkflowId
    job_number: JobNumber
    source_names: list[str]
    output_name: str = 'result'

    def get_result_keys(self) -> list[ResultKey]:
        """Build ResultKeys for all sources."""
        return [
            ResultKey(
                workflow_id=self.workflow_id,
                job_id=JobId(job_number=self.job_number, source_name=source_name),
                output_name=self.output_name,
            )
            for source_name in self.source_names
        ]


@dataclass(frozen=True)
class StaticSource:
    """Fixed data provided directly.

    Use this for data that doesn't change during the plot's lifetime,
    such as reference values, peak positions, or theoretical curves.

    Parameters
    ----------
    data:
        The static data. Type depends on the element factory:
        - For vlines: list[float] of x positions
        - For hlines: list[float] of y positions
        - For curve: tuple of (x_values, y_values)
    """

    data: Any

    def get_result_keys(self) -> list[ResultKey]:
        """Static sources don't use ResultKeys."""
        return []


# =============================================================================
# Layer Configuration
# =============================================================================


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for a single layer in a composed plot.

    This is the user-facing configuration that specifies what to display
    and where the data comes from.

    Parameters
    ----------
    name:
        Unique identifier for this layer within the composition.
    element:
        Element type: "image", "curve", "vlines", "hlines", etc.
        For pipeline sources, this maps to registered plotters.
    source:
        Where the layer's data comes from.
    params:
        Element-specific styling and configuration parameters.
        For pipeline sources, this should be a Pydantic model matching
        the plotter's parameter spec.
    """

    name: str
    element: str
    source: DataSource
    params: pydantic.BaseModel | dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Element Factories
# =============================================================================


class ElementFactory(ABC):
    """Base class for factories that create HoloViews elements from data."""

    @abstractmethod
    def create_element(self, data: Any) -> hv.Element:
        """Create a HoloViews element from the provided data.

        Parameters
        ----------
        data:
            The data to visualize. Type depends on the factory.

        Returns
        -------
        :
            A HoloViews element (Curve, Image, VLine, etc.)
        """


class VLinesFactory(ElementFactory):
    """Factory for vertical line elements."""

    def __init__(self, **opts):
        """Initialize with HoloViews options.

        Parameters
        ----------
        **opts:
            Options passed to ``hv.VLine.opts()``, e.g., color, line_dash.
        """
        self._opts = opts

    def create_element(self, data: list[float] | None) -> hv.Overlay:
        """Create vertical lines at the given x positions.

        Parameters
        ----------
        data:
            List of x positions for vertical lines.

        Returns
        -------
        :
            Overlay of VLine elements.
        """
        if not data:
            return hv.Overlay([])
        return hv.Overlay([hv.VLine(x).opts(**self._opts) for x in data])


class HLinesFactory(ElementFactory):
    """Factory for horizontal line elements."""

    def __init__(self, **opts):
        """Initialize with HoloViews options.

        Parameters
        ----------
        **opts:
            Options passed to ``hv.HLine.opts()``, e.g., color, line_dash.
        """
        self._opts = opts

    def create_element(self, data: list[float] | None) -> hv.Overlay:
        """Create horizontal lines at the given y positions.

        Parameters
        ----------
        data:
            List of y positions for horizontal lines.

        Returns
        -------
        :
            Overlay of HLine elements.
        """
        if not data:
            return hv.Overlay([])
        return hv.Overlay([hv.HLine(y).opts(**self._opts) for y in data])


# =============================================================================
# Layer State
# =============================================================================


@dataclass
class LayerState:
    """Runtime state for a layer.

    Tracks all artifacts needed for a layer's lifecycle.

    Parameters
    ----------
    config:
        The layer configuration (immutable specification).
    pipe:
        HoloViews Pipe for pushing data updates.
    dmap:
        The DynamicMap that renders this layer.
    factory:
        The element factory for creating visuals.
    """

    config: LayerConfig
    pipe: hv.streams.Pipe | None = None
    dmap: hv.DynamicMap | None = None
    factory: ElementFactory | None = None


# =============================================================================
# Plot Composer
# =============================================================================


class PlotComposer:
    """
    Composes multiple layers into a single plot.

    The composer manages layer lifecycle:
    - Creating Pipes and DynamicMaps for each layer
    - Setting up data subscriptions for pipeline sources
    - Composing all layers via the ``*`` operator

    For Phase 1, this handles:
    - Pipeline sources (using existing Plotter infrastructure)
    - Static sources (VLines, HLines)

    Parameters
    ----------
    stream_manager:
        Manager for creating data streams from pipeline sources.
    plotter_registry:
        Registry of available plotters. If None, uses the global registry.
    logger:
        Optional logger instance.
    """

    def __init__(
        self,
        stream_manager: StreamManager,
        plotter_registry: Any | None = None,
        logger: logging.Logger | None = None,
    ):
        self._stream_manager = stream_manager
        self._logger = logger or logging.getLogger(__name__)

        # Import here to avoid circular imports
        if plotter_registry is None:
            from .plotting import plotter_registry as default_registry

            plotter_registry = default_registry
        self._plotter_registry = plotter_registry

        self._layers: dict[str, LayerState] = {}

    def add_pipeline_layer(
        self,
        config: LayerConfig,
        on_first_data: Callable[[hv.DynamicMap], None] | None = None,
    ) -> None:
        """Add a layer with a pipeline data source.

        The layer's DynamicMap is created when first data arrives.

        Parameters
        ----------
        config:
            Layer configuration with a PipelineSource.
        on_first_data:
            Optional callback invoked when first data arrives,
            receives the created DynamicMap.
        """
        if not isinstance(config.source, PipelineSource):
            raise TypeError(f"Expected PipelineSource, got {type(config.source)}")

        if config.name in self._layers:
            self._logger.warning("Layer %s already exists, removing first", config.name)
            self.remove_layer(config.name)

        source = config.source
        keys = source.get_result_keys()

        # Get plotter spec and validate params
        spec = self._plotter_registry.get_spec(config.element)
        params = config.params
        if isinstance(params, dict):
            params = spec.params(**params) if spec.params else pydantic.BaseModel()

        # Create extractors based on window setting (extracted from params)
        from .plot_params import create_extractors_from_params

        window = getattr(params, 'window', None)
        extractors = create_extractors_from_params(keys, window, spec)

        # Initialize layer state (dmap created on first data)
        state = LayerState(config=config)
        self._layers[config.name] = state

        def on_data_ready(pipe: hv.streams.Pipe) -> None:
            """Create DynamicMap when first data arrives."""
            # Create plotter from registry
            plotter = self._plotter_registry.create_plotter(config.element, params)
            plotter.initialize_from_data(pipe.data)

            # Create DynamicMap with plotter
            dmap = hv.DynamicMap(
                plotter, streams=[pipe], kdims=plotter.kdims, cache_size=1
            )

            # Update state
            state.pipe = pipe
            state.dmap = dmap

            self._logger.info("Pipeline layer %s ready with data", config.name)

            if on_first_data:
                on_first_data(dmap)

        # Set up stream subscription
        self._stream_manager.make_merging_stream(
            extractors, on_first_data=on_data_ready
        )
        self._logger.info("Added pipeline layer: %s", config.name)

    def add_static_layer(self, config: LayerConfig) -> None:
        """Add a layer with static data.

        The DynamicMap is created immediately since data is already available.

        Parameters
        ----------
        config:
            Layer configuration with a StaticSource.
        """
        if not isinstance(config.source, StaticSource):
            raise TypeError(f"Expected StaticSource, got {type(config.source)}")

        if config.name in self._layers:
            self._logger.warning("Layer %s already exists, removing first", config.name)
            self.remove_layer(config.name)

        # Get factory and options based on element type
        params = config.params if isinstance(config.params, dict) else {}
        factory = self._create_static_factory(config.element, params)

        # Create pipe with initial data
        pipe = hv.streams.Pipe(data=config.source.data)

        # Closure capture for Panel session safety
        _factory = factory

        def make_element(data, _f=_factory):
            return _f.create_element(data)

        dmap = hv.DynamicMap(make_element, streams=[pipe])

        state = LayerState(config=config, pipe=pipe, dmap=dmap, factory=factory)
        self._layers[config.name] = state

        self._logger.info("Added static layer: %s", config.name)

    def _create_static_factory(
        self, element_type: str, params: dict[str, Any]
    ) -> ElementFactory:
        """Create a factory for static element types.

        Parameters
        ----------
        element_type:
            The element type ("vlines", "hlines", etc.)
        params:
            Styling parameters for the element.

        Returns
        -------
        :
            An ElementFactory for the element type.

        Raises
        ------
        ValueError
            If element_type is not a supported static type.
        """
        factories = {
            'vlines': VLinesFactory,
            'hlines': HLinesFactory,
        }
        if element_type not in factories:
            raise ValueError(
                f"Unknown static element type: {element_type}. "
                f"Supported: {list(factories.keys())}"
            )
        return factories[element_type](**params)

    def remove_layer(self, name: str) -> None:
        """Remove a layer by name.

        Parameters
        ----------
        name:
            Name of the layer to remove.
        """
        if name not in self._layers:
            self._logger.warning("Layer %s not found", name)
            return

        del self._layers[name]
        self._logger.info("Removed layer: %s", name)

    def update_static_layer(self, name: str, data: Any) -> None:
        """Update data for a static layer.

        Parameters
        ----------
        name:
            Name of the layer to update.
        data:
            New data for the layer.

        Raises
        ------
        KeyError
            If no layer with the given name exists.
        ValueError
            If the layer is not a static layer.
        """
        if name not in self._layers:
            raise KeyError(f"Layer {name} not found")

        state = self._layers[name]
        if not isinstance(state.config.source, StaticSource):
            raise ValueError(f"Layer {name} is not a static layer")

        if state.pipe is not None:
            state.pipe.send(data)
            self._logger.debug("Updated static layer %s", name)

    def get_composition(self) -> hv.DynamicMap | hv.Overlay:
        """Get the composed plot of all ready layers.

        Layers that haven't received data yet are excluded from the composition.

        Returns
        -------
        :
            Composed overlay of all ready layers. If no layers are ready,
            returns an empty DynamicMap.
        """
        # KEY INSIGHT: DynamicMap Truthiness
        # HoloViews DynamicMap evaluates to False when empty. We must use
        # explicit `is not None` check, not truthiness.
        dmaps = [
            state.dmap for state in self._layers.values() if state.dmap is not None
        ]

        if not dmaps:
            return hv.DynamicMap(lambda: hv.Curve([]))

        if len(dmaps) == 1:
            return dmaps[0]

        # Compose via * operator - first layer is "bottom", subsequent on top
        result = dmaps[0]
        for dmap in dmaps[1:]:
            result = result * dmap

        return result

    def get_layer_names(self) -> list[str]:
        """Get list of current layer names.

        Returns
        -------
        :
            Names of all layers (including those waiting for data).
        """
        return list(self._layers.keys())

    def get_ready_layer_names(self) -> list[str]:
        """Get names of layers that have received data and are ready.

        Returns
        -------
        :
            Names of layers with data available.
        """
        return [name for name, state in self._layers.items() if state.dmap is not None]

    def is_layer_ready(self, name: str) -> bool:
        """Check if a layer has received data and is ready.

        Parameters
        ----------
        name:
            Name of the layer to check.

        Returns
        -------
        :
            True if the layer exists and has received data.
        """
        state = self._layers.get(name)
        return state is not None and state.dmap is not None
