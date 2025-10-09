# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""This file contains utilities for creating plots in the dashboard."""

from abc import ABC, abstractmethod
from typing import Any

import holoviews as hv
import numpy as np
import scipp as sc

from ess.livedata.config.workflow_spec import ResultKey

from .autoscaler import Autoscaler
from .plot_params import (
    LayoutParams,
    PlotAspect,
    PlotAspectType,
    PlotParams1d,
    PlotParams2d,
    PlotParams3d,
    PlotScale,
    PlotScaleParams,
    PlotScaleParams2d,
)
from .scipp_to_holoviews import to_holoviews


def remove_bokeh_logo(plot, element):
    """Remove Bokeh logo from plots."""
    plot.state.toolbar.logo = None


class Plotter(ABC):
    """
    Base class for plots that support autoscaling.
    """

    def __init__(
        self,
        *,
        aspect_params: PlotAspect | None = None,
        layout_params: LayoutParams | None = None,
        **kwargs,
    ):
        """
        Initialize the plotter.

        Parameters
        ----------
        layout_params:
            Layout parameters for combining multiple datasets. If None, uses defaults.
        **kwargs:
            Additional keyword arguments passed to the autoscaler if created.
        """
        self.autoscaler_kwargs = kwargs
        self.autoscalers: dict[ResultKey, Autoscaler] = {}
        self.layout_params = layout_params or LayoutParams()
        aspect_params = aspect_params or PlotAspect()

        # Note: The way Holoviews (or Bokeh?) determines the axes and data sizing seems
        # to be broken in weird ways. This happens in particular when we return a Layout
        # of multiple plots. Axis ranges that cover less than one unit in data space are
        # problematic in particular, but I have not been able to nail down the exact
        # conditions. Plots will then either have zero frame width or height, or be very
        # small, etc. It is therefore important to set either width or height when using
        # data_aspect or aspect='equal'.
        # However, even that does not solve all problem, for example we can end up with
        # whitespace between plots in a layout.
        self._sizing_opts: dict[str, Any]
        match aspect_params.aspect_type:
            case PlotAspectType.free:
                self._sizing_opts = {}
            case PlotAspectType.equal:
                self._sizing_opts = {'aspect': 'equal'}
            case PlotAspectType.square:
                self._sizing_opts = {'aspect': 'square'}
            case PlotAspectType.aspect:
                self._sizing_opts = {'aspect': aspect_params.ratio}
            case PlotAspectType.data_aspect:
                self._sizing_opts = {'data_aspect': aspect_params.ratio}
        if aspect_params.fix_width:
            self._sizing_opts['frame_width'] = aspect_params.width
        if aspect_params.fix_height:
            self._sizing_opts['frame_height'] = aspect_params.height
        self._sizing_opts['responsive'] = True

    def __call__(
        self, data: dict[ResultKey, sc.DataArray]
    ) -> hv.Overlay | hv.Layout | hv.Element:
        """Create one or more plots from the given data."""
        plots: list[hv.Element] = []
        try:
            for data_key, da in data.items():
                plot_element = self.plot(da, data_key)
                # Add label from data_key if the plot supports it
                if hasattr(plot_element, 'relabel'):
                    plot_element = plot_element.relabel(data_key.job_id.source_name)
                plots.append(plot_element)
        except Exception as e:
            plots = [
                hv.Text(0.5, 0.5, f"Error: {e}").opts(
                    text_align='center', text_baseline='middle'
                )
            ]

        plots = [self._apply_generic_options(p) for p in plots]

        if len(plots) == 1:
            return plots[0]
        if self.layout_params.combine_mode == 'overlay':
            return hv.Overlay(plots)
        return hv.Layout(plots).cols(self.layout_params.layout_columns)

    def _apply_generic_options(self, plot_element: hv.Element) -> hv.Element:
        """Apply generic options like height, responsive, hooks to a plot element."""
        base_opts = {
            'hooks': [remove_bokeh_logo],
            **self._sizing_opts,
        }
        return plot_element.opts(**base_opts)

    def _update_autoscaler_and_get_framewise(
        self, data: sc.DataArray, data_key: ResultKey
    ) -> bool:
        """Update autoscaler with data and return whether bounds changed."""
        if data_key not in self.autoscalers:
            self.autoscalers[data_key] = Autoscaler(**self.autoscaler_kwargs)
        return self.autoscalers[data_key].update_bounds(data)

    @abstractmethod
    def plot(self, data: sc.DataArray, data_key: ResultKey) -> Any:
        """Create a plot from the given data. Must be implemented by subclasses."""


class LinePlotter(Plotter):
    """Plotter for line plots from scipp DataArrays."""

    def __init__(
        self,
        scale_opts: PlotScaleParams,
        **kwargs,
    ):
        """
        Initialize the image plotter.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._base_opts = {
            'logx': True if scale_opts.x_scale == PlotScale.log else False,
            'logy': True if scale_opts.y_scale == PlotScale.log else False,
        }

    @classmethod
    def from_params(cls, params: PlotParams1d):
        """Create LinePlotter from PlotParams1d."""
        return cls(
            value_margin_factor=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
        )

    def plot(self, data: sc.DataArray, data_key: ResultKey) -> hv.Curve:
        """Create a line plot from a scipp DataArray."""
        # TODO Currently we do not plot histograms or else we get a bar chart that is
        # not looking great if we have many bins.
        if data.coords.is_edges(data.dim):
            da = data.assign_coords({data.dim: sc.midpoints(data.coords[data.dim])})
        else:
            da = data
        framewise = self._update_autoscaler_and_get_framewise(da, data_key)

        curve = to_holoviews(da)
        return curve.opts(framewise=framewise, **self._base_opts)


class ImagePlotter(Plotter):
    """Plotter for 2D images from scipp DataArrays."""

    def __init__(
        self,
        scale_opts: PlotScaleParams2d,
        **kwargs,
    ):
        """
        Initialize the image plotter.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._scale_opts = scale_opts
        self._base_opts = {
            'colorbar': True,
            'cmap': 'viridis',
            'logx': True if scale_opts.x_scale == PlotScale.log else False,
            'logy': True if scale_opts.y_scale == PlotScale.log else False,
            'logz': True if scale_opts.color_scale == PlotScale.log else False,
        }

    @classmethod
    def from_params(cls, params: PlotParams2d):
        """Create ImagePlotter from PlotParams2d."""
        return cls(
            value_margin_factor=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
        )

    def plot(self, data: sc.DataArray, data_key: ResultKey) -> hv.Image:
        """Create a 2D plot from a scipp DataArray."""
        data = data.to(dtype='float64')

        # Only mask data when using log color scale
        if self._scale_opts.color_scale == PlotScale.log:
            # With logz=True we need to exclude zero values: The value bounds
            # calculation should properly adjust the color limits. Since zeros can never
            # be included we want to adjust to the lowest positive value.
            masked = data.assign(
                sc.where(
                    data.data <= sc.scalar(0.0, unit=data.unit),
                    sc.scalar(np.nan, unit=data.unit, dtype=data.dtype),
                    data.data,
                )
            )
            plot_data = masked
        else:
            plot_data = data

        framewise = self._update_autoscaler_and_get_framewise(plot_data, data_key)
        # We are using the masked data here since Holoviews (at least with the Bokeh
        # backend) show values below the color limits with the same color as the lowest
        # value in the colormap, which is not what we want for, e.g., zeros on a log
        # scale plot. The nan values will be shown as transparent.
        histogram = to_holoviews(plot_data)
        return histogram.opts(framewise=framewise, **self._base_opts)


class SlicerPlotter(Plotter):
    """Plotter for 3D data with interactive slicing."""

    def __init__(
        self,
        scale_opts: PlotScaleParams2d,
        **kwargs,
    ):
        """
        Initialize the slicer plotter.

        Parameters
        ----------
        scale_opts:
            Scaling options for axes and color.
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._scale_opts = scale_opts
        self._slice_dim: str | None = None
        self._max_slice_idx: int | None = None
        self._current_slice_index: int = 0
        self.slider_widget = None  # Will be set by PlottingController

        # Create custom stream for slice selection
        # This will automatically create a slider widget
        SliceIndex = hv.streams.Stream.define('SliceIndex', slice_index=0)
        self.slice_stream = SliceIndex()

        # Base options for the image plot (similar to ImagePlotter)
        self._base_opts = {
            'colorbar': True,
            'cmap': 'viridis',
            'logx': scale_opts.x_scale == PlotScale.log,
            'logy': scale_opts.y_scale == PlotScale.log,
            'logz': scale_opts.color_scale == PlotScale.log,
        }

    @classmethod
    def from_params(cls, params: PlotParams3d):
        """Create SlicerPlotter from PlotParams3d."""
        return cls(
            scale_opts=params.plot_scale,
            value_margin_factor=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
        )

    def _determine_slice_dim(self, data: sc.DataArray) -> str:
        """
        Determine which dimension to slice along.

        Uses the first dimension of the data (typically the slowest varying).
        """
        if data.ndim != 3:
            raise ValueError(f"Expected 3D data, got {data.ndim}D")
        # Use the first dimension
        return data.dims[0]

    def _get_slice_index(self) -> int:
        """Get current slice index, clipped to valid range."""
        if self._max_slice_idx is None:
            return self._current_slice_index

        # Clip to valid range
        return min(self._current_slice_index, self._max_slice_idx)

    def _format_slice_label(self, data: sc.DataArray, slice_idx: int) -> str:
        """Format a label showing the current slice position."""
        max_idx = data.sizes[self._slice_dim] - 1

        # Try to get coordinate value at this slice
        if self._slice_dim in data.coords:
            coord = data.coords[self._slice_dim]
            # Handle both edge and point coordinates
            if data.coords.is_edges(self._slice_dim):
                # For edges, show the bin center
                value = sc.midpoints(coord, dim=self._slice_dim)[slice_idx]
            else:
                value = coord[slice_idx]

            # Format with unit if available (and not dimensionless)
            if str(value.unit) != 'dimensionless':
                label = f"{self._slice_dim}={value.value:.3g} {value.unit!s}"
            else:
                label = f"{self._slice_dim}={value.value:.3g}"
            label += f" (slice {slice_idx}/{max_idx})"
        else:
            # No coordinate, just show index
            label = f"{self._slice_dim}[{slice_idx}/{max_idx}]"

        return label

    def plot(self, data: sc.DataArray, data_key: ResultKey) -> hv.Image:
        """
        Create a 2D image from a slice of 3D data.

        Parameters
        ----------
        data:
            3D DataArray to slice.
        data_key:
            Key identifying this data.

        Returns
        -------
        :
            A HoloViews Image element showing the selected slice.
        """
        # Determine slice dimension on first call
        if self._slice_dim is None:
            self._slice_dim = self._determine_slice_dim(data)

        # Validate that slice_dim exists in the data
        if self._slice_dim not in data.dims:
            raise ValueError(
                f"Slice dimension '{self._slice_dim}' not found in data. "
                f"Available dimensions: {data.dims}"
            )

        # Update max slice index
        new_max = data.sizes[self._slice_dim] - 1
        if self._max_slice_idx is None or self._max_slice_idx != new_max:
            self._max_slice_idx = new_max
            # Update slider widget bounds if available
            if self.slider_widget is not None:
                self.slider_widget.end = new_max

        slice_idx = self._get_slice_index()

        # Slice the 3D data to get 2D
        sliced_data = data[self._slice_dim, slice_idx]

        # Apply log masking if needed (same as ImagePlotter)
        if self._scale_opts.color_scale == PlotScale.log:
            plot_data = sliced_data.to(dtype='float64')
            plot_data = plot_data.assign(
                sc.where(
                    plot_data.data <= sc.scalar(0.0, unit=plot_data.unit),
                    sc.scalar(np.nan, unit=plot_data.unit, dtype=plot_data.dtype),
                    plot_data.data,
                )
            )
        else:
            plot_data = sliced_data.to(dtype='float64')

        # Update autoscaler and get framewise flag
        framewise = self._update_autoscaler_and_get_framewise(plot_data, data_key)

        # Create the image
        image = to_holoviews(plot_data)

        # Add slice information to title
        slice_label = self._format_slice_label(data, slice_idx)
        title = f"{data.name or 'Data'} - {slice_label}"

        return image.opts(framewise=framewise, title=title, **self._base_opts)

    def __call__(
        self, data: dict[ResultKey, sc.DataArray], slice_index: int = 0, **kwargs
    ) -> hv.Overlay | hv.Layout | hv.Element:
        """
        Create plots from 3D data, slicing at the given index.

        This method is called by HoloViews DynamicMap with stream parameters.

        Parameters
        ----------
        data:
            Dictionary of 3D DataArrays to plot.
        slice_index:
            Index of the slice to display (from slice_stream).
        **kwargs:
            Additional keyword arguments from other streams (ignored).

        Returns
        -------
        :
            HoloViews element(s) showing the sliced data.
        """
        # Store the slice index from the stream parameter
        self._current_slice_index = slice_index

        # Use parent implementation which calls self.plot() for each dataset
        return super().__call__(data)
