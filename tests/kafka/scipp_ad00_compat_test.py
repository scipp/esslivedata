# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from streaming_data_types import area_detector_ad00

from ess.livedata.kafka.scipp_ad00_compat import ad00_to_scipp


def test_ad00_to_scipp_2d():
    """Test conversion of 2D image data."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    ad00 = area_detector_ad00.ADArray(
        source_name="detector",
        unique_id=1,
        timestamp_ns=12345,
        dimensions=np.array([2, 2]),
        data=data.flatten(),
        attributes=[],
    )

    da = ad00_to_scipp(ad00)

    assert isinstance(da, sc.DataArray)
    assert da.dims == ('dim_0', 'dim_1')
    assert da.shape == (2, 2)
    np.testing.assert_array_equal(da.values, [[1.0, 2.0], [3.0, 4.0]])


def test_ad00_to_scipp_3d():
    """Test conversion of 3D image data."""
    data = np.arange(24.0).reshape((2, 3, 4))
    ad00 = area_detector_ad00.ADArray(
        source_name="detector",
        unique_id=1,
        timestamp_ns=12345,
        dimensions=np.array([2, 3, 4]),
        data=data.flatten(),
        attributes=[],
    )

    da = ad00_to_scipp(ad00)

    assert isinstance(da, sc.DataArray)
    assert da.dims == ('dim_0', 'dim_1', 'dim_2')
    assert da.shape == (2, 3, 4)
    np.testing.assert_array_equal(da.values, data)


def test_ad00_to_scipp_1d():
    """Test conversion of 1D data."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ad00 = area_detector_ad00.ADArray(
        source_name="detector",
        unique_id=1,
        timestamp_ns=12345,
        dimensions=np.array([5]),
        data=data,
        attributes=[],
    )

    da = ad00_to_scipp(ad00)

    assert isinstance(da, sc.DataArray)
    assert da.dims == ('dim_0',)
    assert da.shape == (5,)
    np.testing.assert_array_equal(da.values, data)


def test_ad00_to_scipp_preserves_int32_dtype():
    """Test that int32 data type is preserved during conversion."""
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    ad00 = area_detector_ad00.ADArray(
        source_name="detector",
        unique_id=1,
        timestamp_ns=12345,
        dimensions=np.array([2, 2]),
        data=data.flatten(),
        attributes=[],
    )

    da = ad00_to_scipp(ad00)

    assert da.dtype == sc.DType.int32


def test_ad00_to_scipp_converts_uint16_to_int32():
    """Test that uint16 (common for area detectors) is converted to int32."""
    data = np.array([[100, 200], [300, 65535]], dtype=np.uint16)
    ad00 = area_detector_ad00.ADArray(
        source_name="detector",
        unique_id=1,
        timestamp_ns=12345,
        dimensions=np.array([2, 2]),
        data=data.flatten(),
        attributes=[],
    )

    da = ad00_to_scipp(ad00)

    assert da.dtype == sc.DType.int32
    np.testing.assert_array_equal(da.values, [[100, 200], [300, 65535]])


def test_ad00_to_scipp_converts_int16_to_int32():
    """Test that int16 is converted to int32."""
    data = np.array([[1, -2], [3, -4]], dtype=np.int16)
    ad00 = area_detector_ad00.ADArray(
        source_name="detector",
        unique_id=1,
        timestamp_ns=12345,
        dimensions=np.array([2, 2]),
        data=data.flatten(),
        attributes=[],
    )

    da = ad00_to_scipp(ad00)

    assert da.dtype == sc.DType.int32
    np.testing.assert_array_equal(da.values, [[1, -2], [3, -4]])


def test_ad00_to_scipp_converts_uint8_to_int32():
    """Test that uint8 is converted to int32."""
    data = np.array([[0, 127], [128, 255]], dtype=np.uint8)
    ad00 = area_detector_ad00.ADArray(
        source_name="detector",
        unique_id=1,
        timestamp_ns=12345,
        dimensions=np.array([2, 2]),
        data=data.flatten(),
        attributes=[],
    )

    da = ad00_to_scipp(ad00)

    assert da.dtype == sc.DType.int32
    np.testing.assert_array_equal(da.values, [[0, 127], [128, 255]])


def test_ad00_to_scipp_large_image():
    """Test conversion of a larger image."""
    shape = (512, 512)
    data = np.random.rand(*shape).astype(np.float32)
    ad00 = area_detector_ad00.ADArray(
        source_name="detector",
        unique_id=1,
        timestamp_ns=12345,
        dimensions=np.array(shape),
        data=data.flatten(),
        attributes=[],
    )

    da = ad00_to_scipp(ad00)

    assert da.shape == shape
    np.testing.assert_array_equal(da.values, data)
