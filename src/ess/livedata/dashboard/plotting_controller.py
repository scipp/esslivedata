# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable, Mapping

import pydantic

from ess.livedata.config.workflow_spec import (
    DataKey,
    Temporality,
    WorkflowSpec,
)

from .data_subscriber import DataSubscriber
from .extractors import (
    LatestValueExtractor,
    UpdateExtractor,
    WindowAggregatingExtractor,
)
from .plot_params import TimeWindowMixin, TimeWindowParams, WindowModeMixin
from .plotter_registry import (
    OVERLAY_PATTERNS,
    PlotterSpec,
    plotter_registry,
)
from .roi_publisher import ROIPublisher
from .roi_request_plots import ROIPublisherAware
from .stream_manager import StreamManager


class PlottingController:
    """
    Controller for managing plotting operations and configurations.

    Coordinates between stream managers and plot creation,
    using a two-phase pipeline for creating plots with streaming data.

    Parameters
    ----------
    stream_manager:
        Manager for creating data streams.
    roi_publisher:
        Publisher for ROI updates to Kafka. If None, ROI publishing is disabled.
    """

    def __init__(
        self,
        stream_manager: StreamManager,
        roi_publisher: ROIPublisher | None = None,
    ) -> None:
        self._stream_manager = stream_manager
        self._roi_publisher = roi_publisher

    def get_available_plotters_from_spec(
        self, workflow_spec: WorkflowSpec, view_name: str
    ) -> tuple[dict[str, PlotterSpec], bool]:
        """
        Get available plotters based on workflow spec template (before data exists).

        Uses the view's backing template DataArray from the workflow
        specification to determine compatible plotters. The template is an
        "empty" DataArray with the expected structure (dims, coords, units)
        that allows full validation including custom validators.

        When a template is not available, falls back to returning all registered
        plotters. The boolean flag indicates whether a template was available.

        Parameters
        ----------
        workflow_spec:
            WorkflowSpec object containing output templates.
        view_name:
            The name of the output view to get plotters for.

        Returns
        -------
        :
            Tuple of (plotters_dict, has_template). If has_template is False,
            all registered plotters are returned as a fallback, and the caller
            should warn the user that some plotters may not work with the data.
        """
        template = workflow_spec.get_output_template(view_name)
        if template is None:
            return plotter_registry.get_specs(), False
        return plotter_registry.get_compatible_plotters({view_name: template}), True

    def get_spec(self, plot_name: str) -> PlotterSpec:
        """
        Get the parameter model for a given plotter name.

        Parameters
        ----------
        plot_name:
            Name of the plotter to get the specification for.

        Returns
        -------
        :
            The specification for the requested plotter.
        """
        return plotter_registry.get_spec(plot_name)

    def get_params_factory(
        self, plot_name: str
    ) -> Callable[[tuple[str, ...]], type[pydantic.BaseModel]] | None:
        """Get the dim-specialized params factory for a plotter, if any."""
        return plotter_registry.get_params_factory(plot_name)

    def get_static_plotters(self) -> dict[str, PlotterSpec]:
        """
        Get available static plotters (for overlays without data sources).

        Returns
        -------
        :
            Dictionary of static plotter names to their specifications.
        """
        return plotter_registry.get_static_plotters()

    def get_available_overlays(
        self,
        workflow_spec: WorkflowSpec,
        base_plotter_name: str,
    ) -> list[tuple[str, str, str]]:
        """
        Get overlay suggestions for a base layer.

        Returns overlay options that are compatible with the base plotter
        and available in the workflow's outputs.

        Parameters
        ----------
        workflow_spec:
            The workflow specification for the base layer.
        base_plotter_name:
            Name of the base layer's plotter (e.g., "image").

        Returns
        -------
        :
            List of (output_name, plotter_name, plotter_title) tuples for
            overlays that are available based on the workflow's outputs.
        """
        patterns = OVERLAY_PATTERNS.get(base_plotter_name, [])
        if not patterns:
            return []

        # Check which outputs are available in the workflow spec
        if workflow_spec.outputs is None:
            return []

        output_fields = workflow_spec.outputs.model_fields
        available_overlays: list[tuple[str, str, str]] = []

        for output_name, plotter_name in patterns:
            # Check if the required output exists in the workflow spec
            if output_name not in output_fields:
                continue

            # Get the plotter title for display
            try:
                spec = plotter_registry.get_spec(plotter_name)
                plotter_title = spec.title
            except KeyError:
                continue

            available_overlays.append((output_name, plotter_name, plotter_title))

        return available_overlays

    def setup_pipeline(
        self,
        keys_by_role: dict[str, list[DataKey]],
        plot_name: str,
        params: dict | pydantic.BaseModel,
        on_update: Callable[[], None],
        temporality: Mapping[DataKey, Temporality | None],
    ) -> DataSubscriber:
        """
        Set up data pipeline for any plot type.

        This is the unified interface for setting up data pipelines that works
        for both single-source and multi-source layers. PlotOrchestrator should
        use this method exclusively.

        Parameters
        ----------
        keys_by_role
            DataKeys grouped by role, derived from the plot config.
            E.g., {"primary": [...], "x_axis": [...]}
        plot_name
            Name of the plotter to use.
        params
            Plotter parameters as a dict or validated Pydantic model.
        on_update
            Callback invoked when any of the keys changed; see
            :py:class:`DataSubscriber`.
        temporality
            Declared :class:`Temporality` of the field each key resolved to,
            ``None`` where the workflow is not in the registry. Aggregation is
            rejected for the keys it must not be applied to.

        Returns
        -------
        :
            The data subscriber. Can be unregistered via
            DataService.unregister_subscriber() to stop receiving updates
            (e.g., when workflow restarts).
        """
        spec = plotter_registry.get_spec(plot_name)
        # Validate params if dict, pass through if already a model
        if isinstance(params, dict):
            params = spec.params(**params)

        window = params.time_window if isinstance(params, TimeWindowMixin) else None

        # Flatten keys for extractor creation
        all_keys = [key for keys in keys_by_role.values() for key in keys]

        # Standard path: single subscription with role-aware assembly
        extractors = create_extractors_from_params(all_keys, window, temporality, spec)
        return self._stream_manager.make_stream(
            keys_by_role=keys_by_role,
            on_update=on_update,
            extractors=extractors,
        )

    def create_plotter(
        self,
        plot_name: str,
        params: dict | pydantic.BaseModel,
    ):
        """
        Create a plotter instance for the given name and parameters.

        Parameters
        ----------
        plot_name:
            The name of the plotter to create.
        params:
            The plotter parameters as a dict or validated Pydantic model.

        Returns
        -------
        :
            A Plotter instance configured with the given parameters.
        """
        plotter = plotter_registry.create_plotter(plot_name, params=params)
        # ROI request plotters need the ROI publisher
        if isinstance(plotter, ROIPublisherAware):
            plotter.set_roi_publisher(self._roi_publisher)
        return plotter

    def is_overlayable(self, plot_name: str, params: dict | pydantic.BaseModel) -> bool:
        """Whether a layer with this config can share a cell with other layers.

        Tables and layout-mode plotters produce elements that cannot be fused
        via ``hv.Overlay``. Overlayability is not a static flag: for the general
        plot it derives from the params (``combine_mode``), so we build the
        plotter to ask it.

        This method only gates the overlay UI (disabling "Add layer", hiding
        overlay buttons). A config that fails to build is therefore treated as
        overlayable rather than blocked here; a genuinely broken config raises
        again at actual plotter creation, which is the authoritative path.
        """
        try:
            plotter = self.create_plotter(plot_name, params=params)
        except Exception:
            return True
        return getattr(plotter, 'is_overlayable', True)


def _reject_cumulative_aggregation(
    temporality: Mapping[DataKey, Temporality | None],
) -> None:
    """Raise if a window aggregation would be applied to a cumulative field.

    Every message of a cumulative field carries the whole history, so summing
    consecutive ones double-counts. The window controls are hidden for views
    without a per-update field, but ``WorkflowSpec.field_for`` falls back to the
    cumulative field, so a persisted config built when the view still had a
    per-update field would otherwise aggregate one silently.
    """
    outputs = sorted(
        {
            key.output_name
            for key, declared in temporality.items()
            if declared is Temporality.cumulative
        }
    )
    if not outputs:
        return
    raise ValueError(
        f"Cannot aggregate a time window over {', '.join(outputs)}: each message "
        "holds the total accumulated since the run started, so summing them counts "
        "the same events repeatedly. Select 'since run start' mode, or a window "
        "over a per-update output."
    )


def hidden_window_fields(
    params_class: type[pydantic.BaseModel], workflow_spec: WorkflowSpec, view_name: str
) -> frozenset[str]:
    """Return the params fields to hide because the view cannot back them.

    Which fields those are is the params class's business -- see
    :meth:`WindowModeMixin.hidden_fields`; this only supplies the windowing
    options the named view has real backing fields for.

    Permissive for unknown views and for params without a window mode.
    """
    if not issubclass(params_class, WindowModeMixin):
        return frozenset()
    options = workflow_spec.windowing_options(view_name)
    if not options:
        return frozenset()
    return params_class.hidden_fields(options)


def since_start_available(workflow_spec: WorkflowSpec, view_name: str) -> bool:
    """Return whether ``since_start`` mode resolves to a real cumulative field.

    ``False`` for per-update-only views, where selecting ``since_start`` would
    silently fall back to the per-update field (see ``WorkflowSpec.field_for``).
    Permissive for unknown views.
    """
    if workflow_spec.get_output_view(view_name) is None:
        return True
    return 'since_start' in workflow_spec.windowing_options(view_name)


def create_extractors_from_params(
    keys: list[DataKey],
    window: TimeWindowParams | None,
    temporality: Mapping[DataKey, Temporality | None],
    spec: PlotterSpec | None = None,
) -> dict[DataKey, UpdateExtractor]:
    """
    Create extractors based on plotter spec and window configuration.

    Parameters
    ----------
    keys:
        Result keys to create extractors for.
    window:
        Window parameters for extraction mode and aggregation.
        If None, falls back to LatestValueExtractor.
    temporality:
        Declared :class:`Temporality` per key, ``None`` where unknown. Only
        consulted when an aggregating extractor would be built, since that is
        the only construction a cumulative field cannot survive.
    spec:
        Optional plotter specification. If provided and contains a required
        extractor, that extractor type is used.

    Returns
    -------
    :
        Dictionary mapping result keys to extractor instances.
    """
    # Plotter requires specific extractor (e.g., TimeSeriesPlotter)
    if spec is not None and spec.data_requirements.required_extractor is not None:
        extractor_type = spec.data_requirements.required_extractor
        return {key: extractor_type() for key in keys}

    # No fixed requirement - check if window params provided. Stream choice is
    # encoded in the DataKey, so only aggregation over several updates needs a
    # dedicated extractor.
    if window is not None and window.aggregates_updates():
        _reject_cumulative_aggregation(temporality)
        return {
            key: WindowAggregatingExtractor(
                window_duration_seconds=window.window_duration_seconds,
                aggregation=window.aggregation,
            )
            for key in keys
        }
    return {key: LatestValueExtractor() for key in keys}
