# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from streaming_data_types import dataarray_da00

from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00


@pytest.mark.parametrize("unit", [None, 'm', 's', 'counts'])
def test_scipp_to_da00_basic(unit: str | None):
    # Create simple DataArray
    da = sc.DataArray(
        data=sc.array(dims=['x'], values=[1, 2, 3], unit=unit),
        coords={'x': sc.array(dims=['x'], values=[10, 20, 30], unit='m')},
    )

    # Convert to da00
    variables = scipp_to_da00(da)

    # Check results
    assert len(variables) == 2  # data and x coord
    data_var = next(var for var in variables if var.name == 'signal')
    x_var = next(var for var in variables if var.name == 'x')

    assert data_var.unit == unit
    assert np.array_equal(data_var.data, [1, 2, 3])
    assert data_var.axes == ['x']

    assert x_var.unit == 'm'
    assert np.array_equal(x_var.data, [10, 20, 30])
    assert x_var.axes == ['x']


def test_scipp_to_da00_with_variances():
    # Create DataArray with variances
    da = sc.DataArray(
        data=sc.array(
            dims=['x'], values=[1.0, 2.0, 3.0], variances=[0.1, 0.2, 0.3], unit='counts'
        )
    )

    variables = scipp_to_da00(da)

    assert len(variables) == 2  # data and errors
    errors_var = next(var for var in variables if var.name == 'errors')

    # Check that errors are standard deviations (sqrt of variances)
    expected_errors = np.sqrt([0.1, 0.2, 0.3])
    assert np.allclose(errors_var.data, expected_errors)


def test_da00_to_scipp():
    # Create da00 variables
    variables = [
        dataarray_da00.Variable(
            name='signal', data=[1, 2, 3], axes=['x'], shape=(3,), unit='counts'
        ),
        dataarray_da00.Variable(
            name='x', data=[10, 20, 30], axes=['x'], shape=(3,), unit='m'
        ),
    ]

    da = da00_to_scipp(variables)

    assert sc.identical(
        da,
        sc.DataArray(
            sc.array(dims=['x'], values=[1, 2, 3], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[10, 20, 30], unit='m')},
        ),
    )


def test_da00_to_scipp_with_none_unit():
    """Test that variables with unit=None are handled correctly."""
    variables = [
        dataarray_da00.Variable(
            name='signal', data=[1, 2, 3], axes=['x'], shape=(3,), unit=None
        ),
        dataarray_da00.Variable(
            name='x', data=[10, 20, 30], axes=['x'], shape=(3,), unit=None
        ),
    ]

    da = da00_to_scipp(variables)

    assert sc.identical(
        da,
        sc.DataArray(
            sc.array(dims=['x'], values=[1, 2, 3], unit=None),
            coords={'x': sc.array(dims=['x'], values=[10, 20, 30], unit=None)},
        ),
    )


def test_da00_to_scipp_with_errors_and_none_unit():
    """Test conversion with errors when unit is None."""
    variables = [
        dataarray_da00.Variable(
            name='signal', data=[1.0, 2.0, 3.0], axes=['x'], shape=(3,), unit=None
        ),
        dataarray_da00.Variable(
            name='errors', data=[0.1, 0.2, 0.3], axes=['x'], shape=(3,), unit=None
        ),
    ]

    da = da00_to_scipp(variables)

    expected_variances = np.array([0.1, 0.2, 0.3]) ** 2
    assert sc.identical(
        da,
        sc.DataArray(
            sc.array(
                dims=['x'],
                values=[1.0, 2.0, 3.0],
                variances=expected_variances,
                unit=None,
            )
        ),
    )


def test_scipp_to_da00_datetime64():
    """Test conversion of datetime64 variables."""
    times = sc.array(
        dims=['time'],
        values=np.array(['2024-01-01', '2024-01-02'], dtype='datetime64[D]'),
    )
    da = sc.DataArray(
        data=sc.array(dims=['time'], values=[1, 2], unit='counts'),
        coords={'time': times},
    )

    variables = scipp_to_da00(da)

    time_var = next(var for var in variables if var.name == 'time')
    assert time_var.unit == 'datetime64[D]'
    assert time_var.axes == ['time']


def test_da00_to_scipp_datetime64():
    """Test conversion from da00 datetime64 variables."""
    # Days since epoch for 2024-01-01 and 2024-01-02
    days_since_epoch = [19723, 19724]

    variables = [
        dataarray_da00.Variable(
            name='signal', data=[1, 2], axes=['time'], shape=(2,), unit='counts'
        ),
        dataarray_da00.Variable(
            name='time',
            data=days_since_epoch,
            axes=['time'],
            shape=(2,),
            unit='datetime64[D]',
        ),
    ]

    da = da00_to_scipp(variables)

    expected_times = sc.epoch(unit='D') + sc.array(
        dims=['time'], values=days_since_epoch, unit='D'
    )
    assert sc.identical(da.coords['time'], expected_times)


def test_roundtrip_datetime64():
    """Test roundtrip conversion with datetime64 coordinates."""
    times = sc.array(
        dims=['time'],
        values=np.array(['2024-01-01', '2024-01-02'], dtype='datetime64[D]'),
    )
    original = sc.DataArray(
        data=sc.array(dims=['time'], values=[1, 2], unit='counts'),
        coords={'time': times},
    )

    da00 = scipp_to_da00(original)
    converted = da00_to_scipp(da00)

    assert sc.identical(original, converted)


def test_scipp_to_da00_empty_data():
    """Test conversion with empty arrays."""
    da = sc.DataArray(
        data=sc.array(dims=['x'], values=[], unit='counts'),
        coords={'x': sc.array(dims=['x'], values=[], unit='m')},
    )

    variables = scipp_to_da00(da)

    assert len(variables) == 2
    data_var = next(var for var in variables if var.name == 'signal')
    assert len(data_var.data) == 0
    assert data_var.shape == (0,)


def test_da00_to_scipp_empty_data():
    """Test conversion from da00 with empty arrays."""
    variables = [
        dataarray_da00.Variable(
            name='signal', data=[], axes=['x'], shape=(0,), unit='counts'
        ),
    ]

    da = da00_to_scipp(variables)

    assert da.data.shape == (0,)
    assert da.data.unit == 'counts'


def test_roundtrip_conversion():
    original = sc.DataArray(
        data=sc.array(
            dims=['x'], values=[1.0, 2.0, 3.0], variances=[1.0, 4.0, 9.0], unit='counts'
        ),
        coords={'x': sc.array(dims=['x'], values=[10, 20, 30], unit='m')},
    )

    da00 = scipp_to_da00(original)
    converted = da00_to_scipp(da00)

    assert sc.identical(original, converted)


def test_scipp_to_da00_with_name():
    """Test that DataArray.name is preserved in the label field."""
    da = sc.DataArray(
        data=sc.array(dims=['x'], values=[1, 2, 3], unit='counts'),
        coords={'x': sc.array(dims=['x'], values=[10, 20, 30], unit='m')},
        name='my_data',
    )

    variables = scipp_to_da00(da)

    signal_var = next(var for var in variables if var.name == 'signal')
    assert signal_var.label == 'my_data'


def test_scipp_to_da00_without_name():
    """Test that DataArray without name has empty string label."""
    da = sc.DataArray(
        data=sc.array(dims=['x'], values=[1, 2, 3], unit='counts'),
        coords={'x': sc.array(dims=['x'], values=[10, 20, 30], unit='m')},
    )

    variables = scipp_to_da00(da)

    signal_var = next(var for var in variables if var.name == 'signal')
    # scipp uses empty string for "no name"
    assert signal_var.label == ''


def test_da00_to_scipp_with_label():
    """Test that label is restored as DataArray.name."""
    variables = [
        dataarray_da00.Variable(
            name='signal',
            data=[1, 2, 3],
            axes=['x'],
            shape=(3,),
            unit='counts',
            label='my_data',
        ),
        dataarray_da00.Variable(
            name='x', data=[10, 20, 30], axes=['x'], shape=(3,), unit='m'
        ),
    ]

    da = da00_to_scipp(variables)

    assert da.name == 'my_data'


def test_da00_to_scipp_without_label():
    """Test that missing label results in DataArray with empty string name."""
    variables = [
        dataarray_da00.Variable(
            name='signal', data=[1, 2, 3], axes=['x'], shape=(3,), unit='counts'
        ),
        dataarray_da00.Variable(
            name='x', data=[10, 20, 30], axes=['x'], shape=(3,), unit='m'
        ),
    ]

    da = da00_to_scipp(variables)

    # scipp uses empty string for "no name"
    assert da.name == ''


def test_roundtrip_with_name():
    """Test roundtrip conversion preserves DataArray.name."""
    original = sc.DataArray(
        data=sc.array(dims=['x'], values=[1, 2, 3], unit='counts'),
        coords={'x': sc.array(dims=['x'], values=[10, 20, 30], unit='m')},
        name='test_name',
    )

    da00 = scipp_to_da00(original)
    converted = da00_to_scipp(da00)

    assert converted.name == original.name
    assert sc.identical(original, converted)


@pytest.mark.parametrize(
    ('dtype', 'expected_dtype'),
    [
        (np.uint8, np.int32),
        (np.int8, np.int32),
        (np.uint16, np.int32),
        (np.int16, np.int32),
        (np.uint32, np.int64),
        (np.uint64, np.float64),
    ],
)
def test_da00_to_scipp_converts_unsupported_integer_dtypes(
    dtype: np.dtype, expected_dtype: np.dtype
):
    """Test that unsupported integer dtypes are converted to compatible types."""
    variables = [
        dataarray_da00.Variable(
            name='signal',
            data=np.array([1, 2, 3], dtype=dtype),
            axes=['x'],
            shape=(3,),
            unit='counts',
        ),
    ]

    da = da00_to_scipp(variables)

    assert da.data.values.dtype == expected_dtype


def test_da00_to_scipp_preserves_supported_dtypes():
    """Test that supported dtypes are not modified."""
    for dtype in [np.int32, np.int64, np.float32, np.float64]:
        variables = [
            dataarray_da00.Variable(
                name='signal',
                data=np.array([1, 2, 3], dtype=dtype),
                axes=['x'],
                shape=(3,),
                unit='counts',
            ),
        ]

        da = da00_to_scipp(variables)

        assert da.data.values.dtype == dtype


def test_da00_to_scipp_drops_coords_with_incompatible_dimensions():
    """Test that coords with incompatible dimensions are filtered out.

    This handles cases like EFU sending `reference_time` and `frame_total` with
    per-frame dimensions while the signal data is integrated over frames.
    See issue #679 for follow-up work.
    """
    variables = [
        dataarray_da00.Variable(
            name='signal',
            data=np.array([[1, 2], [3, 4]]),
            axes=['x', 'y'],
            shape=(2, 2),
            unit='counts',
        ),
        dataarray_da00.Variable(
            name='x', data=np.array([10, 20]), axes=['x'], shape=(2,), unit='m'
        ),
        # This coord has an incompatible dimension 'frame'
        dataarray_da00.Variable(
            name='reference_time',
            data=np.array([100, 200, 300]),
            axes=['frame'],
            shape=(3,),
            unit='ns',
        ),
        # Another coord with incompatible dimension
        dataarray_da00.Variable(
            name='frame_total',
            data=np.array([10, 20, 30]),
            axes=['frame'],
            shape=(3,),
            unit='counts',
        ),
    ]

    da = da00_to_scipp(variables)

    # Should keep 'x' coord but drop 'reference_time' and 'frame_total'
    assert 'x' in da.coords
    assert 'reference_time' not in da.coords
    assert 'frame_total' not in da.coords


def test_roundtrip_with_inf_bin_edges():
    """Test roundtrip conversion with -inf/+inf overflow bin edges.

    Detector view histograms use overflow bins with -inf/+inf edges to
    capture events outside the user-configured bin range.
    """
    user_edges = sc.linspace('tof', 0.0, 71000.0, 11, unit='ns')
    extended_edges = sc.concat(
        [
            sc.scalar(float('-inf'), unit='ns'),
            user_edges,
            sc.scalar(float('+inf'), unit='ns'),
        ],
        'tof',
    )

    original = sc.DataArray(
        data=sc.ones(dims=['tof'], shape=[12], unit='counts'),
        coords={'tof': extended_edges},
    )

    da00 = scipp_to_da00(original)
    converted = da00_to_scipp(da00)

    assert sc.identical(original, converted)
    # Verify inf values survived the roundtrip
    assert np.isneginf(converted.coords['tof'].values[0])
    assert np.isposinf(converted.coords['tof'].values[-1])


def test_scipp_to_da00_preserves_inf_in_coord():
    """Test that inf values in coordinates are preserved during serialization."""
    edges = sc.array(
        dims=['x'],
        values=[float('-inf'), 0.0, 1.0, float('+inf')],
        unit='m',
    )
    da = sc.DataArray(
        data=sc.array(dims=['x'], values=[1, 2, 3], unit='counts'),
        coords={'x': edges},
    )

    variables = scipp_to_da00(da)
    x_var = next(var for var in variables if var.name == 'x')

    assert np.isneginf(x_var.data[0])
    assert np.isposinf(x_var.data[-1])


def test_da00_to_scipp_keeps_scalar_coords():
    """Test that scalar coords (0-dimensional) are kept."""
    variables = [
        dataarray_da00.Variable(
            name='signal',
            data=np.array([1, 2, 3]),
            axes=['x'],
            shape=(3,),
            unit='counts',
        ),
        dataarray_da00.Variable(
            name='x', data=np.array([10, 20, 30]), axes=['x'], shape=(3,), unit='m'
        ),
        # Scalar coord (no dimensions) should be compatible with any data
        dataarray_da00.Variable(
            name='temperature', data=np.array(300.0), axes=[], shape=(), unit='K'
        ),
    ]

    da = da00_to_scipp(variables)

    assert 'x' in da.coords
    assert 'temperature' in da.coords
    assert da.coords['temperature'].dims == ()
