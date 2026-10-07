# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Write scipp DataArrays to an in-memory NeXus file.

scippnexus reads ``NXdata`` into DataArrays but does not write them, so this
module writes the groups with h5py. The layout follows the NeXus ``NXdata``
conventions, so the file opens in generic NeXus tools, and scippnexus loads each
group back into the DataArray it was written from::

    /entry               NXentry
        title            the ``title`` argument
        program_name     "ess.livedata", with a ``version`` attribute
        <name>           NXdata, one per DataArray
            data         the values, ``signal``
            data_errors  standard deviations, if the data has variances
            <coord>      one field per coord

Coordinates are written as follows:

- A 1-D coord along the dim of the same name (bin edges included) is listed in
  the ``axes`` attribute. A dim without such a coord is listed as ``"."``.
- Other coords get a ``<coord>_indices`` attribute naming the data dims they
  span, unless they span a dim listed as ``"."``, which ``_indices`` cannot
  refer to. Readers then match them to the data by shape. 0-D coords need none.
- Every field carries HDF5 dimension labels with the scipp dim names. scippnexus
  does not use them for the signal, so it names a dim without a coord
  ``dim_<i>`` on load.
- datetime64 values are stored as integer offsets from the Unix epoch, with the
  epoch in the NeXus ``start`` attribute.

Masks are written like coords. NeXus has no notion of masks, so they load back
as boolean coords.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping

import h5py
import numpy as np
import scipp as sc

from .. import __version__

_EPOCH = '1970-01-01T00:00:00Z'
_SIGNAL = 'data'
_ERRORS = f'{_SIGNAL}_errors'


def write_nexus(
    data: Mapping[str, sc.DataArray],
    *,
    title: str,
    group_attrs: Mapping[str, Mapping[str, str]] | None = None,
) -> bytes:
    """Write DataArrays to a NeXus file, one ``NXdata`` group each.

    Parameters
    ----------
    data:
        DataArrays keyed by the name of their ``NXdata`` group.
    title:
        Written to ``/entry/title``.
    group_attrs:
        Extra string attributes per group, keyed like ``data``, e.g. to record
        where the data came from.

    Returns
    -------
    :
        The content of the file.

    Raises
    ------
    ValueError
        If a DataArray is binned, a name contains ``/`` (the HDF5 path
        separator), or a coord or mask name clashes with the signal fields.
    """
    group_attrs = group_attrs or {}
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        entry['title'] = title
        entry['program_name'] = 'ess.livedata'
        entry['program_name'].attrs['version'] = __version__
        _check_names(data)
        for name, da in data.items():
            group = entry.create_group(name)
            _write_nxdata(group, da)
            group.attrs.update(group_attrs.get(name, {}))
    return buffer.getvalue()


def _write_nxdata(group: h5py.Group, da: sc.DataArray) -> None:
    if da.bins is not None:
        raise ValueError('Binned data cannot be written to NXdata.')
    fields = {**da.coords, **da.masks}
    _check_names(fields)
    if clash := {_SIGNAL, _ERRORS} & fields.keys():
        raise ValueError(f'Coord or mask names clash with NXdata signal: {clash}')

    axes = [dim for dim in da.dims if _is_axis(da, dim)]
    group.attrs['NX_class'] = 'NXdata'
    group.attrs['signal'] = _SIGNAL
    if da.ndim > 0:
        group.attrs['axes'] = [dim if dim in axes else '.' for dim in da.dims]
    _write_field(group, _SIGNAL, da.data)
    if da.variances is not None:
        _write_field(group, _ERRORS, sc.stddevs(da.data))
    for name, var in fields.items():
        _write_field(group, name, var)
        if name not in axes and var.ndim > 0 and set(var.dims) <= set(axes):
            group.attrs[f'{name}_indices'] = [da.dims.index(dim) for dim in var.dims]


def _check_names(names: Iterable[str]) -> None:
    if invalid := [name for name in names if '/' in name]:
        raise ValueError(f"Names must not contain '/': {invalid}")


def _is_axis(da: sc.DataArray, name: str) -> bool:
    return name in da.dims and name in da.coords and da.coords[name].dims == (name,)


def _write_field(group: h5py.Group, name: str, var: sc.Variable) -> None:
    if var.dtype == sc.DType.datetime64:
        dataset = group.create_dataset(
            name, data=(var - sc.epoch(unit=var.unit)).values
        )
        dataset.attrs['start'] = _EPOCH
    elif var.dtype == sc.DType.string:
        dataset = group.create_dataset(
            name, data=np.asarray(var.values, dtype=object), dtype=h5py.string_dtype()
        )
    else:
        dataset = group.create_dataset(name, data=var.values)
    if var.unit is not None:
        dataset.attrs['units'] = str(var.unit)
    for label, dim in zip(dataset.dims, var.dims, strict=True):
        label.label = dim
