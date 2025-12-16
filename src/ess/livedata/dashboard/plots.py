# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""This file contains utilities for creating plots in the dashboard."""

from abc import ABC, abstractmethod
from typing import Any, cast

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
    PlotParamsBars,
    PlotScale,
    PlotScaleParams,
    PlotScaleParams2d,
)
from .scipp_to_holoviews import to_holoviews


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
    def _convert_bin_edges_to_midpoints(
        data: sc.DataArray, dim: str | None = None
    ) -> sc.DataArray:
        """
        Convert bin-edge coordinates to midpoints for curve plotting.

        Histograms with many narrow bins don't display well - the black outlines
        dominate. Converting to midpoint coordinates yields a Curve instead.

        Parameters
        ----------
        data:
            DataArray that may have bin-edge coordinates.
        dim:
            Dimension to convert. If None, uses the single dimension of 1D data.

        Returns
        -------
        :
            DataArray with midpoint coordinates if edges were present.
        """
        if dim is None:
            dim = data.dim
        if dim in data.coords and data.coords.is_edges(dim):
            return data.assign_coords({dim: sc.midpoints(data.coords[dim])})
        return data

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
                    label = f'{data_key.job_id.source_name}/{data_key.output_name}'
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
            return hv.Overlay(plots).opts(shared_axes=True)
        if len(plots) == 1:
            return plots[0]
        return hv.Layout(plots).cols(self.layout_params.layout_columns)

    def _apply_generic_options(self, plot_element: hv.Element) -> hv.Element:
        """Apply generic options like aspect ratio to a plot element."""
        return plot_element.opts(**self._sizing_opts)

    def _update_autoscaler_and_get_framewise(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        *,
        coord_data: sc.DataArray | None = None,
    ) -> bool:
        """Update autoscaler with data and return whether bounds changed."""
        if data_key not in self.autoscalers:
            self.autoscalers[data_key] = Autoscaler(**self.autoscaler_kwargs)
        return self.autoscalers[data_key].update_bounds(data, coord_data=coord_data)

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
            grow_threshold=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
        )

    def plot(self, data: sc.DataArray, data_key: ResultKey, **kwargs) -> hv.Curve:
        """Create a line plot from a scipp DataArray."""
        da = self._convert_bin_edges_to_midpoints(data)
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
            grow_threshold=0.1,
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
    """Plotter for 3D data with interactive slicing or flattening."""

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
        self._last_slice_dim: str | None = None
        self._last_slice_idx: int | float | None = None
        self._last_mode: str | None = None

    @classmethod
    def from_params(cls, params: PlotParams3d):
        """Create SlicerPlotter from PlotParams3d."""
        return cls(
            scale_opts=params.plot_scale,
            grow_threshold=0.1,
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

        # Mode selector: slice or flatten
        mode_selector = hv.Dimension(
            'mode',
            values=['slice', 'flatten'],
            default='slice',
            label='Mode',
        )

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
            if (
                coord := first_data.coords.get(dim_name)
            ) is not None and coord.ndim == 1:
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

        self._kdims = [mode_selector, dim_selector, *sliders]

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
        mode: str = 'slice',
        slice_dim: str = '',
        **kwargs,
    ) -> hv.Image:
        """
        Create a 2D image from 3D data by slicing or flattening.

        Parameters
        ----------
        data:
            3D DataArray to process.
        data_key:
            Key identifying this data.
        mode:
            Either 'slice' to select a single slice, or 'flatten' to concatenate
            the outer two dimensions into one.
        slice_dim:
            For 'slice' mode: dimension to slice along.
            For 'flatten' mode: ignored (always flattens outer two dims).
        **kwargs:
            For 'slice' mode: '{slice_dim}_value' (coordinate) or
            '{slice_dim}_index' (integer) for the slice position.

        Returns
        -------
        :
            A HoloViews Image element showing the 2D result.
        """
        if mode == 'flatten':
            plot_data = self._flatten_outer_dims(data)
        else:
            plot_data = self._slice_data(data, slice_dim, kwargs)

        # Prepare data with appropriate dtype and log scale masking
        use_log_scale = self._scale_opts.color_scale == PlotScale.log
        plot_data = self._prepare_2d_image_data(plot_data, use_log_scale)

        # Detect if mode or displayed dimensions changed
        mode_changed = self._last_mode is not None and self._last_mode != mode
        dim_changed = (
            self._last_slice_dim is not None and self._last_slice_dim != slice_dim
        )
        self._last_mode = mode
        self._last_slice_dim = slice_dim

        # Update autoscaler: use 3D data for value (color) bounds to ensure
        # consistent color scale, but use 2D plot_data for coordinate (axis) bounds
        # so we properly track ranges even with 2D coords (which become 1D
        # after slicing).
        framewise = self._update_autoscaler_and_get_framewise(
            data, data_key, coord_data=plot_data
        )

        # Force rescale if mode or displayed dimensions changed
        if mode_changed or dim_changed:
            framewise = True

        image = to_holoviews(plot_data)
        return image.opts(framewise=framewise, **self._base_opts)

    def _slice_data(
        self, data: sc.DataArray, slice_dim: str, kwargs: dict
    ) -> sc.DataArray:
        """Slice 3D data along the specified dimension."""
        # Determine if we're using coordinate values or integer indices
        if (coord_value := kwargs.get(f'{slice_dim}_value')) is not None:
            coord = data.coords[slice_dim]
            slice_idx = sc.scalar(coord_value, unit=coord.unit)
        else:
            slice_idx = kwargs.get(f'{slice_dim}_index', 0)
        return data[slice_dim, slice_idx]

    def _flatten_outer_dims(self, data: sc.DataArray) -> sc.DataArray:
        """Flatten outer two dimensions, keeping inner dimension separate."""
        dims = list(data.dims)
        outer_dims = dims[:2]
        # Condinionally use the inner of the flattened dims as output dim name. It might
        # seem natural to use something like flat_dim = '/'.join(outer_dims) instead,
        # but in practice the actually causes more trouble, since we lose connection to
        # the relevant coords.
        if dims[1] in data.coords:
            flat_dim = dims[1]
        else:
            flat_dim = '/'.join(outer_dims)
        flat = data.flatten(dims=outer_dims, to=flat_dim)
        return flat


class BarsPlotter(Plotter):
    """Plotter for bar charts of 0D scalar data."""

    def __init__(
        self,
        *,
        horizontal: bool = False,
        **kwargs,
    ):
        """
        Initialize the bars plotter.

        Parameters
        ----------
        horizontal:
            If True, bars are horizontal; if False, bars are vertical.
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._horizontal = horizontal

    @classmethod
    def from_params(cls, params: PlotParamsBars):
        """Create BarsPlotter from PlotParamsBars."""
        return cls(
            horizontal=params.orientation.horizontal,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
        )

    def plot(self, data: sc.DataArray, data_key: ResultKey, **kwargs) -> hv.Bars:
        """Create a bar chart from a 0D scipp DataArray."""
        if data.ndim != 0:
            raise ValueError(f"Expected 0D data, got {data.ndim}D")

        label = data_key.job_id.source_name
        value = float(data.value)
        bars = hv.Bars(
            [(label, value)], kdims=['source'], vdims=[data_key.output_name or '']
        )
        opts = {'invert_axes': self._horizontal, 'show_legend': False, 'toolbar': None}
        if self._horizontal:
            opts['yrotation'] = 45
        else:
            opts['xrotation'] = 25
        return cast(hv.Bars, bars.opts(**opts))


class Overlay1DPlotter(Plotter):
    """
    Plotter that slices 2D data along the first dimension and overlays as 1D curves.

    Takes 2D data with dims [slice_dim, plot_dim] and creates an overlay of 1D curves,
    one for each position along the first dimension. Useful for visualizing multiple
    spectra (e.g., ROI spectra) from a single 2D array.

    Colors are assigned by coordinate value (not position) when coordinates are
    integer-like, providing stable color identity across updates.
    """

    def __init__(
        self,
        scale_opts: PlotScaleParams,
        **kwargs,
    ):
        """
        Initialize the overlay 1D plotter.

        Parameters
        ----------
        scale_opts:
            Scaling options for axes.
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._base_opts = {
            'logx': scale_opts.x_scale == PlotScale.log,
            'logy': scale_opts.y_scale == PlotScale.log,
        }
        self._colors = hv.Cycle.default_cycles["default_colors"]

    @classmethod
    def from_params(cls, params: PlotParams1d):
        """Create Overlay1DPlotter from PlotParams1d."""
        from .plot_params import CombineMode

        return cls(
            grow_threshold=0.1,
            layout_params=LayoutParams(combine_mode=CombineMode.overlay),
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
        )

    def plot(
        self, data: sc.DataArray, data_key: ResultKey, **kwargs
    ) -> hv.Overlay | hv.Element:
        """
        Create overlaid curves from a 2D DataArray.

        Slices along the first dimension and creates a curve for each slice.
        """
        del kwargs  # Unused
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data, got {data.ndim}D")

        slice_dim = data.dims[0]
        slice_size = data.sizes[slice_dim]

        if slice_size == 0:
            return hv.Curve([]).opts(**self._base_opts)

        # Update autoscaler with full 2D data to establish global bounds
        framewise = self._update_autoscaler_and_get_framewise(data, data_key)

        # Get coordinate values for labels and colors
        if slice_dim in data.coords:
            coord_values = data.coords[slice_dim].values
        else:
            coord_values = np.arange(slice_size)

        # Pre-convert bin-edge coords to midpoints (shared across all slices)
        data = self._convert_bin_edges_to_midpoints(data, dim=data.dims[1])

        curves: list[hv.Element] = []
        for i in range(slice_size):
            slice_data = data[slice_dim, i]
            coord_val = coord_values[i]

            curve = to_holoviews(slice_data)

            # Assign color by coordinate value for stable identity
            color_idx = int(coord_val) % len(self._colors)
            color = self._colors[color_idx]

            # Label by coordinate value
            label = f"{slice_dim}={coord_val}"
            curve = curve.relabel(label).opts(
                color=color, framewise=framewise, **self._base_opts, **self._sizing_opts
            )
            curves.append(curve)

        if len(curves) == 1:
            return curves[0]
        return hv.Overlay(curves).opts(shared_axes=True)
