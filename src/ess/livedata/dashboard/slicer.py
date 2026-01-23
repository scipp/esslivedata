# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""SlicerPlotter and SlicerPresenter for interactive 3D data slicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import holoviews as hv
import numpy as np
import scipp as sc

from ess.livedata.config.workflow_spec import ResultKey

from .plot_params import PlotParams3d, PlotScale, PlotScaleParams2d, TickParams
from .plots import Plotter
from .scipp_to_holoviews import to_holoviews

if TYPE_CHECKING:
    pass


@dataclass
class SlicerState:
    """
    State for SlicerPresenter - the result of SlicerPlotter.compute().

    Contains 3D data and pre-computed color limits for consistent color
    scale across all slices.
    """

    data: dict[ResultKey, sc.DataArray]
    clim: tuple[float, float] | None = None


class SlicerPresenter:
    """
    Per-session presenter for SlicerPlotter.

    Handles interactive slicing with kdims for mode selection, dimension
    selection, and slice position sliders. Each browser session gets its
    own SlicerPresenter instance with session-bound components.
    """

    def __init__(self, plotter: SlicerPlotter) -> None:
        self._plotter = plotter
        self._kdims: list[hv.Dimension] | None = None

    def present(self, pipe: hv.streams.Pipe) -> hv.DynamicMap:
        """
        Create a DynamicMap with interactive slicing controls.

        Parameters
        ----------
        pipe:
            HoloViews Pipe that receives SlicerState updates.

        Returns
        -------
        :
            DynamicMap with kdims for mode/dimension/slice selection.
        """
        # Initialize kdims from the initial state in the pipe
        if self._kdims is None and pipe.data is not None:
            self._initialize_kdims(pipe.data)

        def render(mode: str, slice_dim: str, *, data: SlicerState = None, **kwargs):
            """Render a slice using the plotter's render_slice method.

            The `data` keyword argument is provided by the Pipe stream.
            The `mode`, `slice_dim`, and other slider kwargs come from kdims.
            """
            if data is None or not data.data:
                return hv.Text(0.5, 0.5, 'No data')

            # Get first data item (SlicerPlotter expects single data source)
            array_data = next(iter(data.data.values()))
            return self._plotter.render_slice(
                array_data, data.clim, mode=mode, slice_dim=slice_dim, **kwargs
            )

        return hv.DynamicMap(render, streams=[pipe], kdims=self._kdims, cache_size=1)

    def _initialize_kdims(self, state: SlicerState) -> None:
        """
        Initialize kdims from the SlicerState data.

        Creates interactive dimensions for mode selection, dimension
        selection, and per-dimension slice position sliders.
        """
        if not state.data:
            return

        first_data = next(iter(state.data.values()))

        if first_data.ndim != 3:
            raise ValueError(f"Expected 3D data, got {first_data.ndim}D")

        dim_names = list(first_data.dims)

        # Mode selector: slice or flatten
        mode_selector = hv.Dimension(
            'mode',
            values=['slice', 'flatten'],
            default='slice',
            label='Mode',
        )

        # Dimension selector
        dim_selector = hv.Dimension(
            'slice_dim',
            values=dim_names,
            default=dim_names[0],
            label='Slice Dimension',
        )

        # Create sliders for each dimension
        sliders = []
        for dim_name in dim_names:
            if (
                coord := first_data.coords.get(dim_name)
            ) is not None and coord.ndim == 1:
                # Use coordinate values for the slider
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


class SlicerPlotter(Plotter):
    """Plotter for 3D data with interactive slicing or flattening.

    Uses two-stage architecture:
    - compute(): Prepares 3D data with pre-computed color bounds (shared)
    - SlicerPresenter: Handles interactive slicing per-session
    """

    def __init__(
        self,
        scale_opts: PlotScaleParams2d,
        tick_params: TickParams | None = None,
        **kwargs,
    ):
        """
        Initialize the slicer plotter.

        Parameters
        ----------
        scale_opts:
            Scaling options for axes and color.
        tick_params:
            Tick configuration parameters.
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._scale_opts = scale_opts
        self._base_opts = self._make_2d_base_opts(scale_opts, tick_params)

    @classmethod
    def from_params(cls, params: PlotParams3d):
        """Create SlicerPlotter from PlotParams3d."""
        return cls(
            scale_opts=params.plot_scale,
            grow_threshold=0.1,
            tick_params=params.ticks,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
        )

    def compute(self, data: dict[ResultKey, sc.DataArray], **kwargs) -> SlicerState:
        """
        Prepare 3D data for slicing with pre-computed color bounds.

        This is Stage 1 of the two-stage architecture. The result is stored
        in PlotDataService and shared across all browser sessions.

        Parameters
        ----------
        data:
            Dictionary of 3D DataArrays.
        **kwargs:
            Unused (required by base class signature).

        Returns
        -------
        :
            SlicerState containing the data and pre-computed color limits.
        """
        del kwargs  # Unused for SlicerPlotter
        clim = self._compute_global_clim(data)
        return SlicerState(data=data, clim=clim)

    def _compute_global_clim(
        self, data: dict[ResultKey, sc.DataArray]
    ) -> tuple[float, float] | None:
        """Compute global color limits from all 3D data for consistent color scale."""
        use_log_scale = self._scale_opts.color_scale == PlotScale.log

        all_values = []
        for da in data.values():
            values = da.values.flatten()
            if use_log_scale:
                # For log scale, only consider positive values
                values = values[values > 0]
            else:
                values = values[np.isfinite(values)]
            if len(values) > 0:
                all_values.append(values)

        if not all_values:
            return None

        combined = np.concatenate(all_values)
        if len(combined) == 0:
            return None

        return (float(np.nanmin(combined)), float(np.nanmax(combined)))

    def create_presenter(self) -> SlicerPresenter:
        """Create a SlicerPresenter for per-session rendering."""
        return SlicerPresenter(self)

    def render_slice(
        self,
        data: sc.DataArray,
        clim: tuple[float, float] | None,
        *,
        mode: str = 'slice',
        slice_dim: str = '',
        **kwargs,
    ) -> hv.Image:
        """
        Render a single 2D slice from 3D data.

        This method is called by SlicerPresenter per-session. It does not
        modify any shared state (no autoscaler updates).

        Parameters
        ----------
        data:
            3D DataArray to process.
        clim:
            Pre-computed color limits for consistent color scale.
        mode:
            Either 'slice' to select a single slice, or 'flatten' to concatenate
            two dimensions into one.
        slice_dim:
            For 'slice' mode: dimension to slice along (removes this dimension).
            For 'flatten' mode: dimension to keep (the other two are flattened).
        **kwargs:
            For 'slice' mode: '{slice_dim}_value' (coordinate) or
            '{slice_dim}_index' (integer) for the slice position.

        Returns
        -------
        :
            A HoloViews Image element showing the 2D result.
        """
        if mode == 'flatten':
            plot_data = self._flatten_outer_dims(data, keep_dim=slice_dim)
        else:
            plot_data = self._slice_data(data, slice_dim, kwargs)

        # Prepare data with appropriate dtype and log scale masking
        use_log_scale = self._scale_opts.color_scale == PlotScale.log
        plot_data = self._prepare_2d_image_data(plot_data, use_log_scale)

        image = to_holoviews(plot_data)
        opts: dict[str, Any] = dict(self._base_opts)
        # Always use framewise=True for interactive slicing
        opts['framewise'] = True
        # Use pre-computed clim for consistent color scale across slices
        if clim is not None:
            opts['clim'] = clim
        return image.opts(**opts)

    def plot(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        **kwargs,
    ) -> hv.Image:
        """Not used - SlicerPlotter uses compute() + SlicerPresenter."""
        raise NotImplementedError(
            "SlicerPlotter uses two-stage architecture. "
            "Call compute() then use SlicerPresenter."
        )

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

    def _flatten_outer_dims(self, data: sc.DataArray, keep_dim: str) -> sc.DataArray:
        """Flatten two dimensions, keeping the specified dimension separate.

        Parameters
        ----------
        data:
            3D DataArray to flatten.
        keep_dim:
            Dimension to keep (not flatten). The other two dimensions will be
            flattened together.
        """
        dims = list(data.dims)
        flatten_dims = [d for d in dims if d != keep_dim]

        # Transpose so keep_dim is last (required for flatten to work on
        # adjacent dims)
        new_order = [*flatten_dims, keep_dim]
        if dims != new_order:
            data = data.transpose(new_order)

        # Conditionally use the inner of the flattened dims as output dim name. It might
        # seem natural to use something like flat_dim = '/'.join(flatten_dims) instead,
        # but in practice that causes more trouble, since we lose connection to
        # the relevant coords.
        if (
            coord := data.coords.get(flatten_dims[1])
        ) is not None and coord.dims == flatten_dims:
            flat_dim = flatten_dims[1]
        else:
            flat_dim = '/'.join(flatten_dims)
        return data.flatten(dims=flatten_dims, to=flat_dim)
