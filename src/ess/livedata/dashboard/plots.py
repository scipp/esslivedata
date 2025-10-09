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

    def initialize_from_data(self, data: dict[ResultKey, sc.DataArray]) -> None:
        """
        Initialize plotter state from initial data.

        Called before creating the DynamicMap to allow plotters to
        inspect the data and set up interactive dimensions.

        Parameters
        ----------
        data:
            Initial data dictionary.
        """
        # Default implementation does nothing; subclasses can override
        return

    @property
    def kdims(self) -> list[hv.Dimension] | None:
        """
        Return key dimensions for interactive widgets.

        Returns
        -------
        :
            List of HoloViews Dimension objects for creating interactive widgets,
            or None if the plotter doesn't require interactive dimensions.
        """
        return None

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
        self._dim_names: list[str] | None = None
        self._dim_sizes: list[int] | None = None

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

    def initialize_from_data(self, data: dict[ResultKey, sc.DataArray]) -> None:
        """
        Initialize the slicer from initial data.

        Extracts dimension names and sizes from the first data array.

        Parameters
        ----------
        data:
            Dictionary of initial data arrays.
        """
        if not data:
            return

        # Get the first data array to inspect its shape
        first_data = next(iter(data.values()))

        if first_data.ndim != 3:
            raise ValueError(f"Expected 3D data, got {first_data.ndim}D")

        # Store dimension names and sizes
        self._dim_names = list(first_data.dims)
        self._dim_sizes = [first_data.sizes[dim] for dim in self._dim_names]

    @property
    def kdims(self) -> list[hv.Dimension] | None:
        """
        Return kdims for interactive widgets: 1 dimension selector + 3 sliders.

        Returns
        -------
        :
            List containing 4 HoloViews Dimensions (selector + 3 sliders),
            or None if not yet initialized.
        """
        if self._dim_names is None or self._dim_sizes is None:
            return None

        # Create dimension selector with actual dimension names
        dim_selector = hv.Dimension(
            'slice_dim',
            values=self._dim_names,
            default=self._dim_names[0],
            label='Slice Dimension',
        )

        # Create 3 sliders, one for each dimension
        sliders = [
            hv.Dimension(
                f'{dim_name}_index',
                range=(0, size - 1),
                default=0,
                label=f'{dim_name} index',
            )
            for dim_name, size in zip(self._dim_names, self._dim_sizes, strict=True)
        ]

        return [dim_selector, *sliders]

    def _get_slice_index(self, requested_idx: int, max_idx: int) -> int:
        """Get slice index, clipped to valid range."""
        return min(max(0, requested_idx), max_idx)

    def _format_value(self, value: sc.Variable) -> str:
        """Format a scipp Variable for display, showing value and unit."""
        try:
            # Try compact format first (works for most dtypes)
            return f"{value:c}"
        except ValueError:
            # Compact formatting not supported (e.g., datetime64)
            # Format as "value unit" or just "value" if no unit or dimensionless
            if value.unit is None or value.unit == sc.units.dimensionless:
                return str(value.value)
            return f"{value.value} {value.unit}"

    def _format_slice_label(
        self, data: sc.DataArray, slice_dim: str, slice_idx: int
    ) -> str:
        """Format a label showing the current slice position."""
        max_idx = data.sizes[slice_dim] - 1

        # Try to get coordinate value at this slice
        if slice_dim in data.coords:
            coord = data.coords[slice_dim]
            # Handle both edge and point coordinates
            if data.coords.is_edges(slice_dim):
                # For edges, show the bin center
                value = sc.midpoints(coord, dim=slice_dim)[slice_idx]
            else:
                value = coord[slice_idx]

            # Format the value
            value_str = self._format_value(value)
            label = f"{slice_dim}={value_str} (slice {slice_idx}/{max_idx})"
        else:
            # No coordinate, just show index
            label = f"{slice_dim}[{slice_idx}/{max_idx}]"

        return label

    def plot(
        self, data: sc.DataArray, data_key: ResultKey, slice_dim: str, slice_idx: int
    ) -> hv.Image:
        """
        Create a 2D image from a slice of 3D data.

        Parameters
        ----------
        data:
            3D DataArray to slice.
        data_key:
            Key identifying this data.
        slice_dim:
            Name of the dimension to slice along.
        slice_idx:
            Index of the slice to display.

        Returns
        -------
        :
            A HoloViews Image element showing the selected slice.
        """
        # Validate that slice_dim exists in the data
        if slice_dim not in data.dims:
            raise ValueError(
                f"Slice dimension '{slice_dim}' not found in data. "
                f"Available dimensions: {data.dims}"
            )

        # Clip slice index to valid range
        max_idx = data.sizes[slice_dim] - 1
        slice_idx = self._get_slice_index(slice_idx, max_idx)

        # Slice the 3D data to get 2D
        sliced_data = data[slice_dim, slice_idx]

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

        # Update autoscaler with full 3D data to establish global bounds.
        # This ensures consistent color scale and axis ranges across all slices.
        # Slice coords/values are subsets of 3D data, so one call suffices.
        framewise = self._update_autoscaler_and_get_framewise(data, data_key)

        # Create the image
        image = to_holoviews(plot_data)

        # Add slice information to title
        slice_label = self._format_slice_label(data, slice_dim, slice_idx)
        title = f"{data.name or 'Data'} - {slice_label}"

        return image.opts(framewise=framewise, title=title, **self._base_opts)

    def __call__(
        self, data: dict[ResultKey, sc.DataArray], slice_dim: str = '', **slice_indices
    ) -> hv.Overlay | hv.Layout | hv.Element:
        """
        Create plots from 3D data, slicing at the given index along selected dimension.

        This method is called by HoloViews DynamicMap with kdims parameters.

        Parameters
        ----------
        data:
            Dictionary of 3D DataArrays to plot.
        slice_dim:
            Name of the dimension to slice along (from kdims selector).
        **slice_indices:
            Keyword arguments containing '{dim_name}_index' for each dimension.
            Only the index corresponding to slice_dim is used.

        Returns
        -------
        :
            HoloViews element(s) showing the sliced data.
        """
        # Build list of plots
        plots: list[hv.Element] = []
        try:
            for data_key, da in data.items():
                # Get the slice index for the selected dimension
                slice_idx = slice_indices.get(f'{slice_dim}_index', 0)

                plot_element = self.plot(da, data_key, slice_dim, slice_idx)
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
