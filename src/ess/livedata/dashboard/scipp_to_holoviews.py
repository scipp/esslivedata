# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import holoviews as hv
import scipp as sc


def coord_to_dimension(var: sc.Variable) -> hv.Dimension:
    """Create a Holoviews Dimension for the coordinate."""
    dim = var.dim
    unit = str(var.unit) if var.unit is not None else None
    return hv.Dimension(dim, label=dim, unit=unit)


def create_value_dimension(data: sc.DataArray) -> hv.Dimension:
    """Create a Holoviews Dimension for the values."""
    label = data.name if data.name else 'values'
    unit = str(data.unit) if data.unit is not None else None
    return hv.Dimension('values', label=label, unit=unit)


def _create_dummy_coord(dim: str, size: int) -> sc.Variable:
    """Create a dummy coordinate for a missing dimension."""
    return sc.arange(dim, size, unit=None)


def _ensure_coords(da: sc.DataArray) -> sc.DataArray:
    """Ensure all dimensions have coordinates, creating dummy ones if needed."""
    for dim in da.dims:
        if dim not in da.coords:
            da = da.assign_coords({dim: _create_dummy_coord(dim, da.sizes[dim])})
    return da


def prepare_1d(
    data: sc.DataArray,
) -> tuple[sc.DataArray, list[hv.Dimension], list[hv.Dimension]]:
    """Prepare a 1D DataArray for HoloViews conversion.

    Ensures coordinates exist and returns the HoloViews dimension objects
    needed by all 1D element constructors. Call this once and pass the result
    to multiple element constructors to avoid redundant work.

    Parameters
    ----------
    data:
        Input 1D DataArray (may be missing dimension coordinates).

    Returns
    -------
    :
        Tuple of (prepared DataArray, kdims, vdims).
    """
    data = _ensure_coords(data)
    coord = data.coords[data.dim]
    return data, [coord_to_dimension(coord)], [create_value_dimension(data)]


def convert_histogram_1d(data: sc.DataArray, label: str = '') -> hv.Histogram:
    """Convert a 1D scipp DataArray to a Holoviews Histogram."""
    dim = data.dim
    coord = data.coords[dim]
    kdims = [coord_to_dimension(coord)]
    vdims = [create_value_dimension(data)]

    return hv.Histogram(
        data=(coord.values, data.values), kdims=kdims, vdims=vdims, label=label
    )


def convert_curve_1d(data: sc.DataArray, label: str = '') -> hv.Curve:
    """Convert a 1D scipp DataArray to a Holoviews Curve."""
    data, kdims, vdims = prepare_1d(data)
    return hv.Curve(
        data=(data.coords[data.dim].values, data.values),
        kdims=kdims,
        vdims=vdims,
        label=label,
    )


def convert_scatter_1d(data: sc.DataArray, label: str = '') -> hv.Scatter:
    """Convert a 1D scipp DataArray to a Holoviews Scatter."""
    data, kdims, vdims = prepare_1d(data)
    return hv.Scatter(
        data=(data.coords[data.dim].values, data.values),
        kdims=kdims,
        vdims=vdims,
        label=label,
    )


def convert_error_bars_1d(data: sc.DataArray, label: str = '') -> hv.ErrorBars:
    """Convert a 1D scipp DataArray to a Holoviews ErrorBars."""
    data, kdims, vdims = prepare_1d(data)
    return hv.ErrorBars(
        data=(data.coords[data.dim].values, data.values, sc.stddevs(data).values),
        kdims=kdims,
        vdims=[*vdims, 'yerr'],
        label=label,
    )


def convert_spread_1d(data: sc.DataArray, label: str = '') -> hv.Spread:
    """Convert a 1D scipp DataArray to a Holoviews Spread."""
    data, kdims, vdims = prepare_1d(data)
    return hv.Spread(
        data=(data.coords[data.dim].values, data.values, sc.stddevs(data).values),
        kdims=kdims,
        vdims=[*vdims, 'yerr'],
        label=label,
    )


def convert_quadmesh_2d(data: sc.DataArray, label: str = '') -> hv.QuadMesh:
    """
    Convert a 2D scipp DataArray to a Holoviews QuadMesh.

    This supports non-evenly spaced coordinates.

    Returns
    -------
    hv.QuadMesh
        A Holoviews QuadMesh object.
    """
    data = _ensure_coords(data)
    kdims = [coord_to_dimension(data.coords[dim]) for dim in reversed(data.dims)]
    vdims = [create_value_dimension(data)]
    coord_values = [data.coords[dim].values for dim in reversed(data.dims)]

    # QuadMesh expects (x, y, values) format
    return hv.QuadMesh(
        data=(*coord_values, data.values), kdims=kdims, vdims=vdims, label=label
    )


def _get_midpoints(data: sc.DataArray, dim: str) -> sc.Variable:
    coord = data.coords[dim]
    if data.coords.is_edges(dim):
        # See https://github.com/scipp/scipp/issues/3765 for why we convert to float64
        return sc.midpoints(coord.to(dtype='float64', copy=False), dim)
    return coord


def _has_degenerate_dimension(data: sc.DataArray) -> bool:
    """Check if any dimension has only one element."""
    return any(size == 1 for size in data.shape)


def _compute_coord_bounds(coord_values) -> tuple[float, float]:
    """
    Compute bounds with half-pixel extension for coordinate values.

    For Image plots, bounds should extend half a pixel beyond the outermost
    coordinate values to properly represent the pixel coverage.

    Parameters
    ----------
    coord_values:
        Array of coordinate values (e.g., bin centers).

    Returns
    -------
    :
        Tuple of (min_bound, max_bound) with half-pixel extensions.
    """
    if len(coord_values) == 1:
        # Single element: create artificial bounds around the value
        center = float(coord_values[0])
        return center - 0.5, center + 0.5
    else:
        # Multiple elements: compute pixel size and extend by half-pixel
        pixel_size = (coord_values.max() - coord_values.min()) / (len(coord_values) - 1)
        min_bound = float(coord_values.min() - pixel_size / 2)
        max_bound = float(coord_values.max() + pixel_size / 2)
        return min_bound, max_bound


def _compute_image_bounds_from_edges(
    data: sc.DataArray, x_midpoints, y_midpoints
) -> tuple[float, float, float, float]:
    """
    Compute exact bounds for hv.Image from edge coordinates.

    When coordinates are bin edges, we compute bounds from the edge values
    to avoid floating-point rounding errors from HoloViews' automatic inference.

    Parameters
    ----------
    data:
        DataArray with coordinates (may have bin edges).
    x_midpoints:
        Midpoint values for x dimension.
    y_midpoints:
        Midpoint values for y dimension.

    Returns
    -------
    :
        Tuple of (left, bottom, right, top) bounds.
    """
    x_dim = data.dims[1]
    y_dim = data.dims[0]

    # For x dimension
    if _is_edges(data, x_dim):
        # Use actual edge values
        x_edges = data.coords[x_dim].values
        left = float(x_edges[0])
        right = float(x_edges[-1])
    else:
        # Use midpoint-based bounds with half-pixel extension
        left, right = _compute_coord_bounds(x_midpoints)

    # For y dimension
    if _is_edges(data, y_dim):
        # Use actual edge values
        y_edges = data.coords[y_dim].values
        bottom = float(y_edges[0])
        top = float(y_edges[-1])
    else:
        # Use midpoint-based bounds with half-pixel extension
        bottom, top = _compute_coord_bounds(y_midpoints)

    return left, bottom, right, top


def convert_image_2d(data: sc.DataArray, label: str = '') -> hv.Image:
    """
    Convert a 2D scipp DataArray to a Holoviews Image.

    This is used when all coordinates are evenly spaced.

    Returns
    -------
    hv.Image
        A Holoviews Image object.
    """
    data = _ensure_coords(data)
    kdims = [coord_to_dimension(data.coords[dim]) for dim in reversed(data.dims)]
    vdims = [create_value_dimension(data)]

    x_coords = _get_midpoints(data, data.dims[1]).values
    y_coords = _get_midpoints(data, data.dims[0]).values

    # Check if we have bin edges - if so, compute exact bounds to avoid
    # floating-point rounding errors from HoloViews' automatic inference
    has_edges = any(_is_edges(data, dim) for dim in data.dims)

    if _has_degenerate_dimension(data) or has_edges:
        # Compute explicit bounds either for degenerate dimensions or bin edges
        left, bottom, right, top = _compute_image_bounds_from_edges(
            data, x_coords, y_coords
        )
        # Pass both coordinates and explicit bounds to maintain data structure
        # while ensuring exact bounds from edges (avoids floating-point errors)
        return hv.Image(
            data=(x_coords, y_coords, data.values),
            bounds=(left, bottom, right, top),
            kdims=kdims,
            vdims=vdims,
            label=label,
        )
    else:
        return hv.Image(
            data=(x_coords, y_coords, data.values),
            kdims=kdims,
            vdims=vdims,
            label=label,
        )


def _all_coords_evenly_spaced(data: sc.DataArray) -> bool:
    """Check if all coordinates in the DataArray are evenly spaced."""
    for dim in data.dims:
        coord = data.coords.get(dim)
        if coord is None:
            # Missing coordinates are treated as evenly spaced (dummy coords)
            continue
        # Empty or single-element coordinates are trivially evenly spaced
        # (sc.islinspace returns False for them)
        if len(coord) <= 1:
            continue
        if not sc.islinspace(coord):
            return False
    return True


def to_holoviews(
    data: sc.DataArray,
    preserve_edges: bool = False,
    label: str = '',
) -> hv.Histogram | hv.Curve | hv.ErrorBars | hv.QuadMesh | hv.Image:
    """
    Convert a scipp DataArray to a Holoviews object.

    Parameters
    ----------
    data:
        The input scipp DataArray to convert.
    preserve_edges:
        If True, use QuadMesh for 2D data with bin edges instead of Image.
        Default is False, which favors Image for better plotting performance. An Image
        can only be used with "midpoint" coords so this is slightly lossy. This option
        allows for preserving edges when needed.
        Edges are always preserved for 1D histogram data and this option is ignored.
    label:
        Label for the HoloViews element. Passed to the constructor to avoid
        the overhead of calling relabel() after creation.

    Returns
    -------
    hv.Histogram | hv.Curve | hv.QuadMesh | hv.Image
        A Holoviews Histogram, Curve, QuadMesh, or Image object.
    """
    if data.dims == ():
        raise ValueError("Input DataArray must have at least one dimension.")

    if len(data.dims) == 1:
        if _is_edges(data, data.dim):
            return convert_histogram_1d(data, label=label)
        elif data.variances is None:
            return convert_curve_1d(data, label=label)
        else:
            return convert_error_bars_1d(data, label=label)
    elif len(data.dims) == 2:
        # Check if we have bin edges and user favors QuadMesh
        has_bin_edges = any(_is_edges(data, dim) for dim in data.dims)
        if preserve_edges and has_bin_edges:
            return convert_quadmesh_2d(data, label=label)
        elif _all_coords_evenly_spaced(data):
            return convert_image_2d(data, label=label)
        else:
            return convert_quadmesh_2d(data, label=label)
    else:
        raise ValueError("Only 1D and 2D data are supported.")


def _is_edges(data: sc.DataArray, dim: str) -> bool:
    return dim in data.coords and data.coords.is_edges(dim)
