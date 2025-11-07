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
    PlotParamsSlidingWindow,
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

    @staticmethod
    def _make_2d_base_opts(scale_opts: PlotScaleParams2d) -> dict[str, Any]:
        """
        Create base options for 2D image plots.

        Parameters
        ----------
        scale_opts:
            Scaling options for axes and color.

        Returns
        -------
        :
            Dictionary of base options for HoloViews image plots.
        """
        return {
            'colorbar': True,
            'cmap': 'viridis',
            'logx': scale_opts.x_scale == PlotScale.log,
            'logy': scale_opts.y_scale == PlotScale.log,
            'logz': scale_opts.color_scale == PlotScale.log,
        }

    @staticmethod
    def _prepare_2d_image_data(data: sc.DataArray, use_log_scale: bool) -> sc.DataArray:
        """
        Convert to float64 and mask non-positive values for log scale.

        With logz=True we need to exclude zero values: The value bounds
        calculation should properly adjust the color limits. Since zeros can never
        be included we want to adjust to the lowest positive value.

        Parameters
        ----------
        data:
            Input data array.
        use_log_scale:
            Whether to apply log scale masking.

        Returns
        -------
        :
            Prepared data array with appropriate dtype and masking.
        """
        plot_data = data.to(dtype='float64')
        if use_log_scale:
            plot_data = plot_data.assign(
                sc.where(
                    plot_data.data <= sc.scalar(0.0, unit=plot_data.unit),
                    sc.scalar(np.nan, unit=plot_data.unit, dtype=plot_data.dtype),
                    plot_data.data,
                )
            )
        return plot_data

    def __call__(
        self, data: dict[ResultKey, sc.DataArray], **kwargs
    ) -> hv.Overlay | hv.Layout | hv.Element:
        """Create one or more plots from the given data."""
        plots: list[hv.Element] = []
        try:
            for data_key, da in data.items():
                plot_element = self.plot(da, data_key, **kwargs)
                # Add label from data_key if the plot supports it
                if hasattr(plot_element, 'relabel'):
                    label = data_key.job_id.source_name
                    if data_key.output_name is not None:
                        label = f'{label}/{data_key.output_name}'
                    plot_element = plot_element.relabel(label)
                plots.append(plot_element)
        except Exception as e:
            plots = [
                hv.Text(0.5, 0.5, f"Error: {e}").opts(
                    text_align='center', text_baseline='middle'
                )
            ]

        if len(plots) == 0:
            plots = [
                hv.Text(0.5, 0.5, 'No data').opts(
                    text_align='center', text_baseline='middle'
                )
            ]

        plots = [self._apply_generic_options(p) for p in plots]

        if self.layout_params.combine_mode == 'overlay':
            return hv.Overlay(plots)
        if len(plots) == 1:
            return plots[0]
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
    def plot(self, data: sc.DataArray, data_key: ResultKey, **kwargs) -> Any:
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

    def plot(self, data: sc.DataArray, data_key: ResultKey, **kwargs) -> hv.Curve:
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
        self._base_opts = self._make_2d_base_opts(scale_opts)

    @classmethod
    def from_params(cls, params: PlotParams2d):
        """Create ImagePlotter from PlotParams2d."""
        return cls(
            value_margin_factor=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
        )

    def plot(self, data: sc.DataArray, data_key: ResultKey, **kwargs) -> hv.Image:
        """Create a 2D plot from a scipp DataArray."""
        # Prepare data with appropriate dtype and log scale masking
        use_log_scale = self._scale_opts.color_scale == PlotScale.log
        plot_data = self._prepare_2d_image_data(data, use_log_scale)

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
        self._kdims: list[hv.Dimension] | None = None
        self._base_opts = self._make_2d_base_opts(scale_opts)

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

        Creates kdims from the first data array.

        Parameters
        ----------
        data:
            Dictionary of initial data arrays.
        """
        if not data:
            raise ValueError("No data provided to initialize_from_data")
        # Get first data array to create kdims
        first_data = next(iter(data.values()))

        if first_data.ndim != 3:
            raise ValueError(f"Expected 3D data, got {first_data.ndim}D")

        # Create kdims from the data
        dim_names = list(first_data.dims)

        # Create dimension selector with actual dimension names
        dim_selector = hv.Dimension(
            'slice_dim',
            values=dim_names,
            default=dim_names[0],
            label='Slice Dimension',
        )

        # Create 3 sliders, one for each dimension
        sliders = []
        for dim_name in dim_names:
            if dim_name in first_data.coords:
                coord = first_data.coords[dim_name]
                # Use coordinate values for the slider
                # For bin-edge coordinates, use midpoints
                if first_data.coords.is_edges(dim_name):
                    coord_values = sc.midpoints(coord, dim=dim_name).values
                else:
                    coord_values = coord.values
                sliders.append(
                    hv.Dimension(
                        f'{dim_name}_value',
                        values=coord_values,
                        default=coord_values[0],
                        label=dim_name,
                        unit=str(coord.unit),
                    )
                )
            else:
                # Fall back to integer indices
                size = first_data.sizes[dim_name]
                sliders.append(
                    hv.Dimension(
                        f'{dim_name}_index',
                        range=(0, size - 1),
                        default=0,
                        label=f'{dim_name} index',
                    )
                )

        self._kdims = [dim_selector, *sliders]

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
        return self._kdims

    def plot(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        *,
        slice_dim: str = '',
        **kwargs,
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
        **kwargs:
            Additional keyword arguments including either '{slice_dim}_value'
            (coordinate) or '{slice_dim}_index' (integer) for the slice position.

        Returns
        -------
        :
            A HoloViews Image element showing the selected slice.
        """

        # Determine if we're using coordinate values or integer indices
        if (coord_value := kwargs.get(f'{slice_dim}_value')) is not None:
            # Use coordinate-based indexing with scipp's label-based indexing
            # Get unit from the data's coordinate
            coord = data.coords[slice_dim]
            slice_idx = sc.scalar(coord_value, unit=coord.unit)
        else:
            # Fall back to integer index
            slice_idx = kwargs.get(f'{slice_dim}_index', 0)

        # Slice the 3D data to get 2D
        sliced_data = data[slice_dim, slice_idx]

        # Prepare data with appropriate dtype and log scale masking
        use_log_scale = self._scale_opts.color_scale == PlotScale.log
        plot_data = self._prepare_2d_image_data(sliced_data, use_log_scale)

        # Update autoscaler with full 3D data to establish global bounds.
        # This ensures consistent color scale and axis ranges across all slices.
        framewise = self._update_autoscaler_and_get_framewise(data, data_key)

        # Create the image
        image = to_holoviews(plot_data)

        return image.opts(framewise=framewise, **self._base_opts)


class SlidingWindowPlotter(Plotter):
    """Plotter for time-series data with sliding window summation."""

    def __init__(
        self,
        max_window_length: float,
        time_dim: str,
        scale_opts_1d: PlotScaleParams,
        scale_opts_2d: PlotScaleParams2d,
        **kwargs,
    ):
        """
        Initialize the sliding window plotter.

        Parameters
        ----------
        max_window_length:
            Maximum window length in seconds.
        time_dim:
            Name of the time dimension to sum over.
        scale_opts_1d:
            Scaling options for 1D plots (when input is 2D).
        scale_opts_2d:
            Scaling options for 2D plots (when input is 3D).
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._max_window_length = max_window_length
        self._time_dim = time_dim
        self._scale_opts_1d = scale_opts_1d
        self._scale_opts_2d = scale_opts_2d
        self._base_opts_1d = {
            'logx': scale_opts_1d.x_scale == PlotScale.log,
            'logy': scale_opts_1d.y_scale == PlotScale.log,
        }
        self._base_opts_2d = self._make_2d_base_opts(scale_opts_2d)
        self._kdims: list[hv.Dimension] | None = None

    @classmethod
    def from_params(cls, params: PlotParamsSlidingWindow):
        """Create SlidingWindowPlotter from PlotParamsSlidingWindow."""
        return cls(
            max_window_length=params.max_window_length,
            time_dim=params.time_dim,
            scale_opts_1d=params.plot_scale_1d,
            scale_opts_2d=params.plot_scale_2d,
            value_margin_factor=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
        )

    def initialize_from_data(self, data: dict[ResultKey, sc.DataArray]) -> None:
        """
        Initialize the sliding window from initial data.

        Creates kdims for the window length slider.

        Parameters
        ----------
        data:
            Dictionary of initial data arrays.
        """
        if not data:
            raise ValueError("No data provided to initialize_from_data")

        # Create window length slider from 1s to max_window_length
        window_slider = hv.Dimension(
            'window_length',
            range=(1.0, self._max_window_length),
            default=min(10.0, self._max_window_length),
            label='Window Length',
            unit='s',
        )

        self._kdims = [window_slider]

    @property
    def kdims(self) -> list[hv.Dimension] | None:
        """
        Return kdims for interactive window length slider.

        Returns
        -------
        :
            List containing the window length slider Dimension,
            or None if not yet initialized.
        """
        return self._kdims

    def plot(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        *,
        window_length: float = 10.0,
        **kwargs,
    ) -> hv.Curve | hv.Image:
        """
        Create a plot from time-series data with sliding window summation.

        Parameters
        ----------
        data:
            2D or 3D DataArray with a time dimension.
        data_key:
            Key identifying this data.
        window_length:
            Length of the sliding window in seconds.
        **kwargs:
            Additional keyword arguments (unused).

        Returns
        -------
        :
            A HoloViews Curve (for 2D input) or Image (for 3D input) element.
        """
        if self._time_dim not in data.dims:
            raise ValueError(
                f"Time dimension '{self._time_dim}' not found in data. "
                f"Available dimensions: {list(data.dims)}"
            )

        if self._time_dim not in data.coords:
            raise ValueError(
                f"Time dimension '{self._time_dim}' has no coordinate. "
                "Cannot perform time-based windowing."
            )

        # Get the time coordinate
        time_coord = data.coords[self._time_dim]

        # Get the maximum time value
        if data.coords.is_edges(self._time_dim):
            max_time = time_coord[-1]
        else:
            max_time = time_coord[data.sizes[self._time_dim] - 1]

        # Calculate the window start time
        window_start = max_time - sc.scalar(window_length, unit=time_coord.unit)

        # Slice the data to get only the last window_length seconds
        windowed_data = data[self._time_dim, window_start:]

        # Sum over the time dimension
        summed_data = windowed_data.sum(self._time_dim)

        # Determine output dimensionality and plot accordingly
        if summed_data.ndim == 1:
            # 2D input → 1D output (line plot)
            # Handle histogram vs curve based on edges
            if summed_data.coords.is_edges(summed_data.dim):
                da = summed_data.assign_coords(
                    {summed_data.dim: sc.midpoints(summed_data.coords[summed_data.dim])}
                )
            else:
                da = summed_data

            framewise = self._update_autoscaler_and_get_framewise(da, data_key)
            curve = to_holoviews(da)
            return curve.opts(framewise=framewise, **self._base_opts_1d)

        elif summed_data.ndim == 2:
            # 3D input → 2D output (image plot)
            use_log_scale = self._scale_opts_2d.color_scale == PlotScale.log
            plot_data = self._prepare_2d_image_data(summed_data, use_log_scale)

            framewise = self._update_autoscaler_and_get_framewise(plot_data, data_key)
            image = to_holoviews(plot_data)
            return image.opts(framewise=framewise, **self._base_opts_2d)

        else:
            raise ValueError(
                f"After summing over '{self._time_dim}', expected 1D or 2D data, "
                f"got {summed_data.ndim}D"
            )
