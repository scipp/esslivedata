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


def convert_histogram_1d(data: sc.DataArray) -> hv.Histogram:
    """
    Convert a 1D scipp DataArray to a Holoviews Histogram.

    Returns
    -------
    hv.Histogram
        A Holoviews Histogram object.
    """
    dim = data.dim
    coord = data.coords[dim]
    kdims = [coord_to_dimension(coord)]
    vdims = [create_value_dimension(data)]

    return hv.Histogram(data=(coord.values, data.values), kdims=kdims, vdims=vdims)


def convert_curve_1d(data: sc.DataArray) -> hv.Curve:
    """
    Convert a 1D scipp DataArray to a Holoviews Curve.

    Returns
    -------
    hv.Curve
        A Holoviews Curve object.
    """
    data = _ensure_coords(data)
    dim = data.dim
    coord = data.coords[dim]
    kdims = [coord_to_dimension(coord)]
    vdims = [create_value_dimension(data)]

    return hv.Curve(data=(coord.values, data.values), kdims=kdims, vdims=vdims)


def convert_error_bars_1d(data: sc.DataArray) -> hv.ErrorBars:
    """
    Convert a 1D scipp DataArray to a Holoviews Curve.

    Returns
    -------
    hv.ErrorBars
        A Holoviews ErrorBars object.
    """
    data = _ensure_coords(data)
    dim = data.dim
    coord = data.coords[dim]
    kdims = [coord_to_dimension(coord)]
    vdims = [create_value_dimension(data)]

    return hv.ErrorBars(
        data=(coord.values, data.values, sc.stddevs(data).values),
        kdims=kdims,
        vdims=[*vdims, 'yerr'],
    )


def convert_quadmesh_2d(data: sc.DataArray) -> hv.QuadMesh:
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
    return hv.QuadMesh(data=(*coord_values, data.values), kdims=kdims, vdims=vdims)


def _get_midpoints(data: sc.DataArray, dim: str) -> sc.Variable:
    coord = data.coords[dim]
    if data.coords.is_edges(dim):
        # See https://github.com/scipp/scipp/issues/3765 for why we convert to float64
        return sc.midpoints(coord.to(dtype='float64', copy=False), dim)
    return coord


def convert_image_2d(data: sc.DataArray) -> hv.Image:
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
    return hv.Image(
        data=(
            _get_midpoints(data, data.dims[1]).values,
            _get_midpoints(data, data.dims[0]).values,
            data.values,
        ),
        kdims=kdims,
        vdims=vdims,
    )


def _all_coords_evenly_spaced(data: sc.DataArray) -> bool:
    """Check if all coordinates in the DataArray are evenly spaced."""
    for dim in data.dims:
        coord = data.coords.get(dim)
        if coord is None:
            # Missing coordinates are treated as evenly spaced (dummy coords)
            continue
        if not sc.islinspace(coord):
            return False
    return True


def to_holoviews(
    data: sc.DataArray,
    preserve_edges: bool = False,
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

    Returns
    -------
    hv.Histogram | hv.Curve | hv.QuadMesh | hv.Image
        A Holoviews Histogram, Curve, QuadMesh, or Image object.
    """
    if data.dims == ():
        raise ValueError("Input DataArray must have at least one dimension.")

    if len(data.dims) == 1:
        if _is_edges(data, data.dim):
            return convert_histogram_1d(data)
        elif data.variances is None:
            return convert_curve_1d(data)
        else:
            return convert_error_bars_1d(data)
    elif len(data.dims) == 2:
        # Check if we have bin edges and user favors QuadMesh
        has_bin_edges = any(_is_edges(data, dim) for dim in data.dims)
        if preserve_edges and has_bin_edges:
            return convert_quadmesh_2d(data)
        elif _all_coords_evenly_spaced(data):
            return convert_image_2d(data)
        else:
            return convert_quadmesh_2d(data)
    else:
        raise ValueError("Only 1D and 2D data are supported.")


def _is_edges(data: sc.DataArray, dim: str) -> bool:
    return dim in data.coords and data.coords.is_edges(dim)
