# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc
from streaming_data_types import dataarray_da00


def scipp_to_da00(
    da: sc.DataArray, *, signal_name: str = 'signal'
) -> list[dataarray_da00.Variable]:
    # Encode DataArray.name in the 'label' field of the signal variable
    label = da.name if da.name is not None else None
    if da.variances is None:
        variables = [_to_da00_variable(signal_name, da.data, label=label)]
    else:
        variables = [
            _to_da00_variable(signal_name, sc.values(da.data), label=label),
            _to_da00_variable('errors', sc.stddevs(da.data)),
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

    # scipp expects name to be a string (empty string for "no name")
    return sc.DataArray(data, coords=variables_dict, name=label)


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
    if var.unit is not None and var.unit.startswith('datetime64'):
        unit = var.unit.split('[')[1].rstrip(']')
        return sc.epoch(unit=unit) + sc.array(dims=var.axes, values=var.data, unit=unit)
    return sc.array(dims=var.axes, values=var.data, unit=var.unit)
