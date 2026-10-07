# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import io

import h5py
import numpy as np
import pytest
import scipp as sc
import scipp.testing
import scippnexus as snx

from ess.livedata.dashboard.nexus_writer import write_nexus

_TIMES = sc.datetimes(
    dims=['time'],
    values=np.array(
        ['2026-10-07T10:00', '2026-10-07T10:01', '2026-10-07T10:02'],
        dtype='datetime64[ns]',
    ),
    unit='ns',
)


def _load(content: bytes) -> sc.DataGroup:
    return snx.load(io.BytesIO(content))['entry']


def _assert_round_trip(da: sc.DataArray, loaded: sc.DataArray) -> None:
    # A dim without a coord loads under a generic name.
    loaded = loaded.rename_dims(dict(zip(loaded.dims, da.dims, strict=True)))
    # Errors are stored as standard deviations, so variances come back squared.
    if da.variances is None:
        sc.testing.assert_identical(loaded, da)
    else:
        sc.testing.assert_identical(sc.values(loaded), sc.values(da))
        np.testing.assert_allclose(loaded.variances, da.variances)


CASES = {
    'time_series': sc.DataArray(
        sc.array(
            dims=['time'],
            values=[1.0, 2.0, 3.0],
            variances=[1.0, 4.0, 9.0],
            unit='counts',
        ),
        coords={
            'time': _TIMES,
            'motor_x': sc.array(dims=['time'], values=[0.1, 0.2, 0.3], unit='mm'),
            'start_time': _TIMES[0],
            'end_time': _TIMES[-1],
        },
    ),
    'bin_edges': sc.DataArray(
        sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
        coords={'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0], unit='m')},
    ),
    'image_with_string_coord': sc.DataArray(
        sc.ones(dims=['y', 'x'], shape=[2, 3], unit='dimensionless'),
        coords={
            'x': sc.arange('x', 3.0, unit='m'),
            'y': sc.arange('y', 2),
            'label': sc.array(dims=['y'], values=['a', 'b']),
        },
    ),
    'two_d_coord': sc.DataArray(
        sc.ones(dims=['y', 'x'], shape=[2, 3]),
        coords={
            'x': sc.arange('x', 3.0, unit='m'),
            'position': sc.ones(dims=['y', 'x'], shape=[2, 3], unit='m'),
        },
    ),
    'coord_over_dim_with_and_dim_without_coord': sc.DataArray(
        sc.ones(dims=['y', 'x'], shape=[2, 3]),
        coords={
            'x': sc.arange('x', 3.0, unit='m'),
            'label': sc.array(dims=['y'], values=['a', 'b']),
            'position': sc.ones(dims=['y', 'x'], shape=[2, 3], unit='m'),
        },
    ),
    'two_d_coord_over_dims_without_coords': sc.DataArray(
        sc.ones(dims=['y', 'x'], shape=[2, 3]),
        coords={'position': sc.ones(dims=['y', 'x'], shape=[2, 3], unit='m')},
    ),
    'no_units': sc.DataArray(
        sc.array(dims=['x'], values=[1, 2], unit=None),
        coords={'x': sc.array(dims=['x'], values=[0.0, 1.0], unit=None)},
    ),
    'dim_without_coord': sc.DataArray(
        sc.array(dims=['x'], values=[1.0, 2.0], unit='counts')
    ),
    'scalar': sc.DataArray(
        sc.scalar(5.0, variance=1.0, unit='counts'),
        coords={'start_time': _TIMES[0], 'end_time': _TIMES[1]},
    ),
}


@pytest.mark.parametrize('name', CASES)
def test_scippnexus_loads_the_written_data_array(name: str) -> None:
    da = CASES[name]
    loaded = _load(write_nexus({name: da}, title='t'))
    _assert_round_trip(da, loaded[name])


def test_mask_loads_as_coord() -> None:
    mask = sc.array(dims=['x'], values=[True, False])
    da = sc.DataArray(
        sc.array(dims=['x'], values=[1.0, 2.0]),
        coords={'x': sc.arange('x', 2.0)},
        masks={'bad': mask},
    )
    loaded = _load(write_nexus({'d': da}, title='t'))['d']
    sc.testing.assert_identical(loaded.coords['bad'], mask)


def test_writes_one_nxdata_per_array_in_an_nxentry() -> None:
    data = {'a': CASES['bin_edges'], 'b': CASES['scalar']}
    with h5py.File(io.BytesIO(write_nexus(data, title='My scan')), 'r') as f:
        entry = f['entry']
        assert entry.attrs['NX_class'] == 'NXentry'
        assert entry['title'][()].decode() == 'My scan'
        assert entry['program_name'][()].decode() == 'ess.livedata'
        assert {name: entry[name].attrs['NX_class'] for name in ('a', 'b')} == {
            'a': 'NXdata',
            'b': 'NXdata',
        }


def test_group_attrs_are_written() -> None:
    content = write_nexus(
        {'a': CASES['bin_edges']},
        title='t',
        group_attrs={'a': {'source_name': 'monitor_1'}},
    )
    with h5py.File(io.BytesIO(content), 'r') as f:
        assert f['entry/a'].attrs['source_name'] == 'monitor_1'


def test_datetimes_are_offsets_from_epoch() -> None:
    content = write_nexus({'a': CASES['time_series']}, title='t')
    with h5py.File(io.BytesIO(content), 'r') as f:
        time = f['entry/a/time']
        assert time.attrs['start'] == '1970-01-01T00:00:00Z'
        assert time.attrs['units'] == 'ns'
        assert time[0] == _TIMES[0].value.astype('int64')


def test_binned_data_raises() -> None:
    table = sc.data.table_xyz(10)
    with pytest.raises(ValueError, match='Binned'):
        write_nexus({'a': table.bin(x=2)}, title='t')


def test_coord_named_like_signal_raises() -> None:
    da = sc.DataArray(sc.arange('x', 2.0), coords={'data': sc.arange('x', 2.0)})
    with pytest.raises(ValueError, match='clash'):
        write_nexus({'a': da}, title='t')
