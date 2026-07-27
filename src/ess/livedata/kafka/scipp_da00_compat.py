# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import operator
from functools import reduce

import numpy as np
import scipp as sc
from streaming_data_types import dataarray_da00

# Scipp only supports: bool, float32, float64, int32, int64, string, datetime64.
# Map unsupported integer types to compatible types.
_DTYPE_MAP = {
    np.dtype('uint8'): np.int32,
    np.dtype('int8'): np.int32,
    np.dtype('uint16'): np.int32,
    np.dtype('int16'): np.int32,
    np.dtype('uint32'): np.int64,
    np.dtype('uint64'): np.float64,  # May lose precision for large values
}


def scipp_to_da00(
    da: sc.DataArray, *, signal_name: str = 'signal'
) -> list[dataarray_da00.Variable]:
    # Encode DataArray.name in the 'label' field of the signal variable
    label = da.name if da.name is not None else None
    data = _masked_to_nan(da) if da.masks else da.data
    if data.variances is None:
        variables = [_to_da00_variable(signal_name, data, label=label)]
    else:
        variables = [
            _to_da00_variable(signal_name, sc.values(data), label=label),
            _to_da00_variable('errors', sc.stddevs(data)),
        ]
    variables.extend(
        [
            _to_da00_variable(name, var)
            for name, var in da.coords.items()
            if var.shape == var.values.shape  # vector3 etc. not supported currently
        ]
    )
    return variables


def da00_to_scipp(
    variables: list[dataarray_da00.Variable], *, signal_name: str = 'signal'
) -> sc.DataArray:
    # Extract label from signal variable to restore DataArray.name
    signal_var = next((v for v in variables if v.name == signal_name), None)
    # Use empty string if label is None (scipp convention for "no name")
    label = signal_var.label if signal_var and signal_var.label is not None else ''

    variables_dict = {var.name: _to_scipp_variable(var) for var in variables}
    data = variables_dict.pop(signal_name)
    if (errors := variables_dict.pop('errors', None)) is not None:
        data.variances = (errors**2).values

    # Filter coords to only those with compatible dimensions.
    # This is a workaround for EFU sending variables like `reference_time` and
    # `frame_total` with per-frame dimensions, while the signal data is integrated
    # over frames. Since DataArray requires compatible dimensions for all coords,
    # we drop coords with incompatible dimensions. See issue #679 for follow-up work.
    compatible_coords = {
        name: var
        for name, var in variables_dict.items()
        if set(var.dims).issubset(set(data.dims))
    }

    # scipp expects name to be a string (empty string for "no name")
    return sc.DataArray(data, coords=compatible_coords, name=label)


def _masked_to_nan(da: sc.DataArray) -> sc.Variable:
    """
    Return the data of ``da`` with masked elements (and their variances) set to NaN.

    Requires ``da`` to have at least one mask.

    Transporting masks is perfectly feasible: da00's variable list is generic enough
    to carry them by naming convention. The obstacle is the consumer side. Nothing in
    the dashboard reads ``DataArray.masks`` -- neither the buffering and extraction
    chain nor the plotters -- so a transported mask would be dropped before it could
    affect what is drawn. NaN is honored end to end instead: transparent image pixels,
    gaps in curves, autoscaling and aggregation over finite values only. That is what
    a mask is meant to convey.

    Applying masks here thus discards the mask/data distinction deliberately. Adding
    mask support across transport, dashboard data chain, and plotters is a viable
    option should that distinction be needed.

    Integer data is promoted to float64 since it cannot represent NaN.
    """
    mask = reduce(operator.or_, da.masks.values())
    data = da.data
    if data.dtype not in (sc.DType.float64, sc.DType.float32):
        data = data.to(dtype=sc.DType.float64)
    nan = sc.scalar(
        np.nan,
        unit=data.unit,
        dtype=data.dtype,
        variance=None if data.variances is None else np.nan,
    )
    return sc.where(mask, nan, data)


def _to_da00_variable(
    name: str, var: sc.Variable, *, label: str | None = None
) -> dataarray_da00.Variable:
    if var.dtype == sc.DType.datetime64:
        timedelta = var - sc.epoch(unit=var.unit)
        return dataarray_da00.Variable(
            name=name,
            data=timedelta.values,
            axes=list(var.dims),
            shape=var.shape,
            unit=f'datetime64[{var.unit}]',
            label=label,
        )
    return dataarray_da00.Variable(
        name=name,
        data=var.values,
        axes=list(var.dims),
        shape=var.shape,
        unit=None if var.unit is None else str(var.unit),
        label=label,
    )


def _to_scipp_variable(var: dataarray_da00.Variable) -> sc.Variable:
    data = np.asarray(var.data)
    if data.dtype in _DTYPE_MAP:
        data = data.astype(_DTYPE_MAP[data.dtype])
    if var.unit is not None and var.unit.startswith('datetime64'):
        unit = var.unit.split('[')[1].rstrip(']')
        return sc.epoch(unit=unit) + sc.array(dims=var.axes, values=data, unit=unit)
    return sc.array(dims=var.axes, values=data, unit=var.unit)
