# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Correlation histogram plotters and related data structures.

This module contains the plotter implementations for correlation histograms,
along with simplified parameter models used by the PlotConfigModal wizard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pydantic
import scipp as sc

from ess.livedata.config.workflow_spec import ResultKey

from .correlation_histogram import CorrelationHistogrammer, CorrelationHistogramParams
from .plot_params import PlotDisplayParams1d, PlotDisplayParams2d
from .plots import ImagePlotter, LinePlotter


class Bin1dParams(pydantic.BaseModel):
    """Bin parameters for 1D correlation histograms."""

    x_bins: int = pydantic.Field(
        default=50,
        ge=1,
        le=1000,
        title="X Bins",
        description="Number of bins for X axis (range auto-determined from data).",
    )


class Bin2dParams(pydantic.BaseModel):
    """Bin parameters for 2D correlation histograms."""

    x_bins: int = pydantic.Field(
        default=50,
        ge=1,
        le=1000,
        title="X Bins",
        description="Number of bins for X axis (range auto-determined from data).",
    )
    y_bins: int = pydantic.Field(
        default=50,
        ge=1,
        le=1000,
        title="Y Bins",
        description="Number of bins for Y axis (range auto-determined from data).",
    )


class SimplifiedCorrelationHistogram1dParams(
    CorrelationHistogramParams, PlotDisplayParams1d
):
    """Simplified params for 1D correlation histogram with auto-determined ranges.

    Used by PlotConfigModal wizard. The plotter auto-determines bin edges from data.
    Inherits display options (layout, plot_scale, ticks, plot_aspect) from
    PlotDisplayParams1d.
    """

    bins: Bin1dParams = pydantic.Field(
        default_factory=Bin1dParams,
        title="Histogram Bins",
        description="Bin configuration for the histogram.",
    )


class SimplifiedCorrelationHistogram2dParams(
    CorrelationHistogramParams, PlotDisplayParams2d
):
    """Simplified params for 2D correlation histogram with auto-determined ranges.

    Used by PlotConfigModal wizard. The plotter auto-determines bin edges from data.
    Inherits display options (layout, plot_scale, ticks, plot_aspect) from
    PlotDisplayParams2d.
    """

    bins: Bin2dParams = pydantic.Field(
        default_factory=Bin2dParams,
        title="Histogram Bins",
        description="Bin configuration for the histogram.",
    )


# Map plotter names to their simplified param classes for PlotConfigModal
SIMPLIFIED_CORRELATION_PARAMS: dict[str, type[CorrelationHistogramParams]] = {
    'correlation_histogram_1d': SimplifiedCorrelationHistogram1dParams,
    'correlation_histogram_2d': SimplifiedCorrelationHistogram2dParams,
}

# Plotter names that are correlation histogram types
CORRELATION_HISTOGRAM_PLOTTERS = frozenset(
    {'correlation_histogram_1d', 'correlation_histogram_2d'}
)


@dataclass
class CorrelationHistogramData:
    """Structured data for correlation histogram plotters.

    Separates data sources (to be histogrammed) from axis sources (defining bins).
    """

    data_sources: dict[ResultKey, sc.DataArray]
    """Primary data sources to histogram. Multiple sources = multiple histograms."""

    axis_data: dict[str, sc.DataArray]
    """Axis data for correlation. Keys are 'x' and optionally 'y'."""


class CorrelationHistogramAssembler:
    """Assembler for correlation histogram data with explicit data/axis separation.

    Produces structured CorrelationHistogramData with separate data sources and
    axis sources. Requires all axis sources and at least one data source before
    triggering. Supports progressive arrival of additional data sources.
    """

    def __init__(
        self, data_keys: list[ResultKey], axis_keys: dict[str, ResultKey]
    ) -> None:
        """
        Initialize the assembler.

        Parameters
        ----------
        data_keys
            Keys for data sources to histogram. Can be multiple (one histogram each).
        axis_keys
            Keys for axis sources. Maps axis name ('x', 'y') to ResultKey.
        """
        self._data_keys = data_keys
        self._axis_keys = axis_keys
        self._all_keys = set(data_keys) | set(axis_keys.values())

    @property
    def keys(self) -> set[ResultKey]:
        """Return the set of all data keys this assembler depends on."""
        return self._all_keys

    @property
    def requires_all_keys(self) -> bool:
        """Override to use can_trigger instead of simple all-or-nothing."""
        # Return False to let DataSubscriber use can_trigger logic
        return False

    def can_trigger(self, available_keys: set[ResultKey]) -> bool:
        """Check if minimum requirements are met for triggering.

        Requires all axis sources + at least one data source.
        """
        has_all_axes = set(self._axis_keys.values()) <= available_keys
        has_any_data = bool(set(self._data_keys) & available_keys)
        return has_all_axes and has_any_data

    def assemble(self, data: dict[ResultKey, Any]) -> CorrelationHistogramData:
        """Assemble structured correlation histogram data.

        Returns
        -------
        :
            CorrelationHistogramData with data_sources dict (keyed by ResultKey)
            and axis_data dict (keyed by axis name 'x', 'y').
        """
        return CorrelationHistogramData(
            data_sources={k: data[k] for k in self._data_keys if k in data},
            axis_data={
                name: data[key] for name, key in self._axis_keys.items() if key in data
            },
        )


def _compute_edges_from_data(
    data: sc.DataArray, num_bins: int, dim: str
) -> sc.Variable:
    """Compute histogram edges from data range.

    Parameters
    ----------
    data
        DataArray containing the data to determine range from.
    num_bins
        Number of bins for the histogram.
    dim
        Dimension name for the edges.

    Returns
    -------
    :
        Scipp Variable with evenly spaced bin edges.
    """
    values = sc.values(data) if data.variances is not None else data
    low = float(values.nanmin().value)
    high = float(np.nextafter(values.nanmax().value, np.inf))
    return sc.linspace(dim, low, high, num_bins + 1, unit=data.unit)


class CorrelationHistogram1dPlotter:
    """Plotter for 1D correlation histograms.

    Receives structured CorrelationHistogramData with data sources and axis data.
    Computes a histogram for EACH data source, creating overlay/layout as configured.
    """

    kdims: list[str] | None = None

    def __init__(
        self, params: SimplifiedCorrelationHistogram1dParams, **kwargs
    ) -> None:
        self._num_bins = params.bins.x_bins
        self._normalize = params.normalization.per_second
        # Store display params directly (inherited from PlotDisplayParams1d)
        self._plot_scale = params.plot_scale
        self._ticks = params.ticks
        self._layout = params.layout
        self._plot_aspect = params.plot_aspect
        self._histogrammer: CorrelationHistogrammer | None = None
        # Internal LinePlotter for rendering with display options
        self._renderer: LinePlotter | None = None

    def initialize_from_data(self, data: CorrelationHistogramData) -> None:
        """Initialize histogram edges and renderer from first data arrival."""
        if 'x' not in data.axis_data:
            raise ValueError("Correlation histogram requires x-axis data")

        # Initialize histogram edges from axis data
        x_axis_data = data.axis_data['x']
        edges = {'x': _compute_edges_from_data(x_axis_data, self._num_bins, 'x')}
        self._histogrammer = CorrelationHistogrammer(
            edges=edges, normalize=self._normalize
        )

        # Create internal LinePlotter with display options
        self._renderer = LinePlotter(
            scale_opts=self._plot_scale,
            tick_params=self._ticks,
            layout_params=self._layout,
            aspect_params=self._plot_aspect,
            value_margin_factor=0.1,
        )

    def __call__(self, data: CorrelationHistogramData) -> Any:
        """Compute histograms for all data sources and render."""
        if self._histogrammer is None or self._renderer is None:
            self.initialize_from_data(data)
            if self._histogrammer is None or self._renderer is None:
                raise RuntimeError("Failed to initialize histogrammer or renderer")

        # Compute histogram for EACH data source
        histograms: dict[ResultKey, sc.DataArray] = {}
        for key, source_data in data.data_sources.items():
            histograms[key] = self._histogrammer(source_data, coords=data.axis_data)

        # Delegate rendering to LinePlotter (handles overlay/layout)
        return self._renderer(histograms)

    @classmethod
    def from_params(cls, params: SimplifiedCorrelationHistogram1dParams):
        """Factory method for plotter registry."""
        return cls(params=params)


class CorrelationHistogram2dPlotter:
    """Plotter for 2D correlation histograms.

    Receives structured CorrelationHistogramData with data sources and axis data.
    Computes a 2D histogram for EACH data source, creating overlay/layout as configured.
    """

    kdims: list[str] | None = None

    def __init__(
        self, params: SimplifiedCorrelationHistogram2dParams, **kwargs
    ) -> None:
        self._x_bins = params.bins.x_bins
        self._y_bins = params.bins.y_bins
        self._normalize = params.normalization.per_second
        # Store display params directly (inherited from PlotDisplayParams2d)
        self._plot_scale = params.plot_scale
        self._ticks = params.ticks
        self._layout = params.layout
        self._plot_aspect = params.plot_aspect
        self._histogrammer: CorrelationHistogrammer | None = None
        # Internal ImagePlotter for rendering with display options
        self._renderer: ImagePlotter | None = None

    def initialize_from_data(self, data: CorrelationHistogramData) -> None:
        """Initialize histogram edges and renderer from first data arrival."""
        if 'x' not in data.axis_data or 'y' not in data.axis_data:
            raise ValueError("2D correlation histogram requires x-axis and y-axis data")

        # Initialize histogram edges from axis data
        edges = {
            'x': _compute_edges_from_data(data.axis_data['x'], self._x_bins, 'x'),
            'y': _compute_edges_from_data(data.axis_data['y'], self._y_bins, 'y'),
        }
        self._histogrammer = CorrelationHistogrammer(
            edges=edges, normalize=self._normalize
        )

        # Create internal ImagePlotter with display options
        self._renderer = ImagePlotter(
            scale_opts=self._plot_scale,
            tick_params=self._ticks,
            layout_params=self._layout,
            aspect_params=self._plot_aspect,
            value_margin_factor=0.1,
        )

    def __call__(self, data: CorrelationHistogramData) -> Any:
        """Compute 2D histograms for all data sources and render."""
        if self._histogrammer is None or self._renderer is None:
            self.initialize_from_data(data)
            if self._histogrammer is None or self._renderer is None:
                raise RuntimeError("Failed to initialize histogrammer or renderer")

        # Compute histogram for EACH data source
        histograms: dict[ResultKey, sc.DataArray] = {}
        for key, source_data in data.data_sources.items():
            histograms[key] = self._histogrammer(source_data, coords=data.axis_data)

        # Delegate rendering to ImagePlotter (handles overlay/layout)
        return self._renderer(histograms)

    @classmethod
    def from_params(cls, params: SimplifiedCorrelationHistogram2dParams):
        """Factory method for plotter registry."""
        return cls(params=params)
