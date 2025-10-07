# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from pydantic import ValidationError

from ess.livedata.config import models


def test_weighting_method_values():
    assert models.WeightingMethod.PIXEL_NUMBER == "pixel_number"


def test_pixel_weighting_defaults():
    weight = models.PixelWeighting()
    assert not weight.enabled
    assert weight.method == models.WeightingMethod.PIXEL_NUMBER


def test_pixel_weighting_custom():
    weight = models.PixelWeighting(enabled=True)
    assert weight.enabled


def test_pixel_weighting_invalid_method():
    with pytest.raises(ValidationError):
        models.PixelWeighting(method="invalid")


def test_update_every_defaults():
    model = models.UpdateEvery()
    assert model.value == 1.0
    assert model.unit == "s"


@pytest.mark.parametrize(
    ("value", "unit", "expected_ns"),
    [
        (1, "ns", 1),
        (1, "us", 1000),
        (1, "ms", 1_000_000),
        (1, "s", 1_000_000_000),
    ],
)
def test_time_model_conversions(value, unit, expected_ns):
    model = models.TimeModel(value=value, unit=unit)
    assert model.value_ns == expected_ns


def test_update_every_validation():
    with pytest.raises(ValidationError):
        models.UpdateEvery(value=0.05)  # Below minimum of 0.1


class TestConfigKey:
    def test_defaults(self):
        key = models.ConfigKey(key="test_key")
        assert key.source_name is None
        assert key.service_name is None
        assert key.key == "test_key"

    def test_custom_values(self):
        key = models.ConfigKey(
            source_name="source1", service_name="service1", key="test_key"
        )
        assert key.source_name == "source1"
        assert key.service_name == "service1"
        assert key.key == "test_key"

    def test_str_all_values(self):
        key = models.ConfigKey(
            source_name="source1", service_name="service1", key="test_key"
        )
        assert str(key) == "source1/service1/test_key"

    def test_str_with_wildcards(self):
        key1 = models.ConfigKey(
            source_name=None, service_name="service1", key="test_key"
        )
        assert str(key1) == "*/service1/test_key"

        key2 = models.ConfigKey(
            source_name="source1", service_name=None, key="test_key"
        )
        assert str(key2) == "source1/*/test_key"

        key3 = models.ConfigKey(source_name=None, service_name=None, key="test_key")
        assert str(key3) == "*/*/test_key"

    def test_from_string_all_values(self):
        key = models.ConfigKey.from_string("source1/service1/test_key")
        assert key.source_name == "source1"
        assert key.service_name == "service1"
        assert key.key == "test_key"

    def test_from_string_with_wildcards(self):
        key1 = models.ConfigKey.from_string("*/service1/test_key")
        assert key1.source_name is None
        assert key1.service_name == "service1"
        assert key1.key == "test_key"

        key2 = models.ConfigKey.from_string("source1/*/test_key")
        assert key2.source_name == "source1"
        assert key2.service_name is None
        assert key2.key == "test_key"

        key3 = models.ConfigKey.from_string("*/*/test_key")
        assert key3.source_name is None
        assert key3.service_name is None
        assert key3.key == "test_key"

    def test_from_string_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid key format"):
            models.ConfigKey.from_string("invalid_key")

        with pytest.raises(ValueError, match="Invalid key format"):
            models.ConfigKey.from_string("source1/service1")

        with pytest.raises(ValueError, match="Invalid key format"):
            models.ConfigKey.from_string("source1/service1/key1/extra")

    def test_roundtrip_conversion(self):
        original = models.ConfigKey(
            source_name="source1", service_name="service1", key="test_key"
        )
        string_repr = str(original)
        parsed = models.ConfigKey.from_string(string_repr)
        assert parsed.source_name == original.source_name
        assert parsed.service_name == original.service_name
        assert parsed.key == original.key

        # With wildcards
        original = models.ConfigKey(source_name=None, service_name=None, key="test_key")
        string_repr = str(original)
        parsed = models.ConfigKey.from_string(string_repr)
        assert parsed.source_name is None
        assert parsed.service_name is None
        assert parsed.key == original.key


class TestRectangleROI:
    def test_creation(self):
        roi = models.RectangleROI(
            x_min=10.0, x_max=20.0, y_min=5.0, y_max=15.0, x_unit='mm', y_unit='mm'
        )
        assert roi.x_min == 10.0
        assert roi.x_max == 20.0
        assert roi.y_min == 5.0
        assert roi.y_max == 15.0
        assert roi.x_unit == 'mm'
        assert roi.y_unit == 'mm'

    def test_validation_x_bounds(self):
        with pytest.raises(ValidationError, match="x_min .* must be < x_max"):
            models.RectangleROI(
                x_min=20.0, x_max=10.0, y_min=5.0, y_max=15.0, x_unit='mm', y_unit='mm'
            )

    def test_validation_y_bounds(self):
        with pytest.raises(ValidationError, match="y_min .* must be < y_max"):
            models.RectangleROI(
                x_min=10.0, x_max=20.0, y_min=15.0, y_max=5.0, x_unit='mm', y_unit='mm'
            )

    def test_to_data_array(self):
        roi = models.RectangleROI(
            x_min=10.0, x_max=20.0, y_min=5.0, y_max=15.0, x_unit='mm', y_unit='mm'
        )
        da = roi.to_data_array()

        assert da.name == models.ROIType.RECTANGLE
        assert list(da.dims) == ['bounds']
        assert da.shape == (2,)
        assert 'x' in da.coords
        assert 'y' in da.coords
        np.testing.assert_array_equal(da.coords['x'].values, [10.0, 20.0])
        np.testing.assert_array_equal(da.coords['y'].values, [5.0, 15.0])
        assert str(da.coords['x'].unit) == 'mm'
        assert str(da.coords['y'].unit) == 'mm'

    def test_from_data_array(self):
        # Create a DataArray
        data = sc.array(dims=['bounds'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'x': sc.array(dims=['bounds'], values=[10.0, 20.0], unit='mm'),
            'y': sc.array(dims=['bounds'], values=[5.0, 15.0], unit='mm'),
        }
        da = sc.DataArray(data, coords=coords, name=models.ROIType.RECTANGLE)

        # Convert back to ROI
        roi = models.ROI.from_data_array(da)

        assert isinstance(roi, models.RectangleROI)
        assert roi.x_min == 10.0
        assert roi.x_max == 20.0
        assert roi.y_min == 5.0
        assert roi.y_max == 15.0
        assert roi.x_unit == 'mm'
        assert roi.y_unit == 'mm'

    def test_roundtrip_conversion(self):
        original = models.RectangleROI(
            x_min=10.0, x_max=20.0, y_min=5.0, y_max=15.0, x_unit='mm', y_unit='mm'
        )
        da = original.to_data_array()
        restored = models.ROI.from_data_array(da)
        assert original == restored

    def test_different_units(self):
        roi = models.RectangleROI(
            x_min=10.0, x_max=20.0, y_min=5.0, y_max=15.0, x_unit='mm', y_unit='m'
        )
        da = roi.to_data_array()
        restored = models.ROI.from_data_array(da)
        assert restored.x_unit == 'mm'
        assert restored.y_unit == 'm'


class TestPolygonROI:
    def test_creation(self):
        roi = models.PolygonROI(
            x=[0.0, 10.0, 5.0], y=[0.0, 0.0, 10.0], x_unit='mm', y_unit='mm'
        )
        assert roi.x == [0.0, 10.0, 5.0]
        assert roi.y == [0.0, 0.0, 10.0]
        assert roi.x_unit == 'mm'
        assert roi.y_unit == 'mm'

    def test_validation_length_mismatch(self):
        with pytest.raises(ValidationError, match="x and y must have the same length"):
            models.PolygonROI(x=[0.0, 10.0], y=[0.0], x_unit='mm', y_unit='mm')

    def test_validation_min_vertices(self):
        with pytest.raises(
            ValidationError, match="Polygon must have at least 3 vertices"
        ):
            models.PolygonROI(x=[0.0, 10.0], y=[0.0, 0.0], x_unit='mm', y_unit='mm')

    def test_to_data_array(self):
        roi = models.PolygonROI(
            x=[0.0, 10.0, 5.0], y=[0.0, 0.0, 10.0], x_unit='mm', y_unit='mm'
        )
        da = roi.to_data_array()

        assert da.name == models.ROIType.POLYGON
        assert list(da.dims) == ['vertex']
        assert da.shape == (3,)
        assert 'x' in da.coords
        assert 'y' in da.coords
        np.testing.assert_array_equal(da.coords['x'].values, [0.0, 10.0, 5.0])
        np.testing.assert_array_equal(da.coords['y'].values, [0.0, 0.0, 10.0])

    def test_from_data_array(self):
        # Create a DataArray
        n = 4
        data = sc.array(dims=['vertex'], values=np.ones(n, dtype=np.int32), unit='')
        coords = {
            'x': sc.array(dims=['vertex'], values=[0.0, 10.0, 10.0, 0.0], unit='mm'),
            'y': sc.array(dims=['vertex'], values=[0.0, 0.0, 10.0, 10.0], unit='mm'),
        }
        da = sc.DataArray(data, coords=coords, name=models.ROIType.POLYGON)

        # Convert back to ROI
        roi = models.ROI.from_data_array(da)

        assert isinstance(roi, models.PolygonROI)
        assert roi.x == [0.0, 10.0, 10.0, 0.0]
        assert roi.y == [0.0, 0.0, 10.0, 10.0]

    def test_roundtrip_conversion(self):
        original = models.PolygonROI(
            x=[0.0, 10.0, 5.0, 2.0], y=[0.0, 0.0, 10.0, 5.0], x_unit='mm', y_unit='mm'
        )
        da = original.to_data_array()
        restored = models.ROI.from_data_array(da)
        assert original == restored


class TestEllipseROI:
    def test_creation(self):
        roi = models.EllipseROI(
            center_x=10.0,
            center_y=20.0,
            radius_x=5.0,
            radius_y=3.0,
            rotation=45.0,
            unit='mm',
        )
        assert roi.center_x == 10.0
        assert roi.center_y == 20.0
        assert roi.radius_x == 5.0
        assert roi.radius_y == 3.0
        assert roi.rotation == 45.0
        assert roi.unit == 'mm'

    def test_default_rotation(self):
        roi = models.EllipseROI(
            center_x=10.0, center_y=20.0, radius_x=5.0, radius_y=3.0, unit='mm'
        )
        assert roi.rotation == 0.0

    def test_validation_positive_radii(self):
        with pytest.raises(ValidationError):
            models.EllipseROI(
                center_x=10.0, center_y=20.0, radius_x=0.0, radius_y=3.0, unit='mm'
            )

        with pytest.raises(ValidationError):
            models.EllipseROI(
                center_x=10.0, center_y=20.0, radius_x=5.0, radius_y=-1.0, unit='mm'
            )

    def test_to_data_array(self):
        roi = models.EllipseROI(
            center_x=10.0,
            center_y=20.0,
            radius_x=5.0,
            radius_y=3.0,
            rotation=45.0,
            unit='mm',
        )
        da = roi.to_data_array()

        assert da.name == models.ROIType.ELLIPSE
        assert list(da.dims) == ['dim']
        assert da.shape == (2,)
        assert 'center' in da.coords
        assert 'radius' in da.coords
        assert 'rotation' in da.coords
        np.testing.assert_array_equal(da.coords['center'].values, [10.0, 20.0])
        np.testing.assert_array_equal(da.coords['radius'].values, [5.0, 3.0])
        assert da.coords['rotation'].value == 45.0
        assert str(da.coords['rotation'].unit) == 'deg'

    def test_from_data_array(self):
        # Create a DataArray
        data = sc.array(dims=['dim'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'center': sc.array(dims=['dim'], values=[10.0, 20.0], unit='mm'),
            'radius': sc.array(dims=['dim'], values=[5.0, 3.0], unit='mm'),
        }
        da = sc.DataArray(data, coords=coords, name=models.ROIType.ELLIPSE)
        da.coords['rotation'] = sc.scalar(45.0, unit='deg')

        # Convert back to ROI
        roi = models.ROI.from_data_array(da)

        assert isinstance(roi, models.EllipseROI)
        assert roi.center_x == 10.0
        assert roi.center_y == 20.0
        assert roi.radius_x == 5.0
        assert roi.radius_y == 3.0
        assert roi.rotation == 45.0
        assert roi.unit == 'mm'

    def test_from_data_array_no_rotation(self):
        # Create a DataArray without rotation coordinate
        data = sc.array(dims=['dim'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'center': sc.array(dims=['dim'], values=[10.0, 20.0], unit='mm'),
            'radius': sc.array(dims=['dim'], values=[5.0, 3.0], unit='mm'),
        }
        da = sc.DataArray(data, coords=coords, name=models.ROIType.ELLIPSE)

        # Convert back to ROI - should default to 0.0 rotation
        roi = models.ROI.from_data_array(da)
        assert roi.rotation == 0.0

    def test_roundtrip_conversion(self):
        original = models.EllipseROI(
            center_x=10.0,
            center_y=20.0,
            radius_x=5.0,
            radius_y=3.0,
            rotation=45.0,
            unit='mm',
        )
        da = original.to_data_array()
        restored = models.ROI.from_data_array(da)
        assert original == restored


class TestROIDispatch:
    def test_from_data_array_missing_name(self):
        data = sc.array(dims=['bounds'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'x': sc.array(dims=['bounds'], values=[10.0, 20.0], unit='mm'),
            'y': sc.array(dims=['bounds'], values=[5.0, 15.0], unit='mm'),
        }
        da = sc.DataArray(data, coords=coords)  # No name

        with pytest.raises(ValueError, match="DataArray missing name"):
            models.ROI.from_data_array(da)

    def test_from_data_array_unknown_type(self):
        data = sc.array(dims=['bounds'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'x': sc.array(dims=['bounds'], values=[10.0, 20.0], unit='mm'),
            'y': sc.array(dims=['bounds'], values=[5.0, 15.0], unit='mm'),
        }
        da = sc.DataArray(data, coords=coords, name='unknown_type')

        with pytest.raises(ValueError, match="Unknown ROI type: unknown_type"):
            models.ROI.from_data_array(da)


class TestROIWithDa00:
    """Test ROI roundtrip through da00 serialization."""

    def test_rectangle_through_da00(self):
        from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00

        # Create ROI
        original_roi = models.RectangleROI(
            x_min=10.0, x_max=20.0, y_min=5.0, y_max=15.0, x_unit='mm', y_unit='mm'
        )

        # Convert to DataArray
        da = original_roi.to_data_array()

        # Serialize through da00
        da00_vars = scipp_to_da00(da)
        da_restored = da00_to_scipp(da00_vars)

        # Convert back to ROI
        restored_roi = models.ROI.from_data_array(da_restored)

        assert original_roi == restored_roi

    def test_polygon_through_da00(self):
        from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00

        original_roi = models.PolygonROI(
            x=[0.0, 10.0, 5.0, 2.0], y=[0.0, 0.0, 10.0, 5.0], x_unit='mm', y_unit='mm'
        )

        da = original_roi.to_data_array()
        da00_vars = scipp_to_da00(da)
        da_restored = da00_to_scipp(da00_vars)
        restored_roi = models.ROI.from_data_array(da_restored)

        assert original_roi == restored_roi

    def test_ellipse_through_da00(self):
        from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00

        original_roi = models.EllipseROI(
            center_x=10.0,
            center_y=20.0,
            radius_x=5.0,
            radius_y=3.0,
            rotation=45.0,
            unit='mm',
        )

        da = original_roi.to_data_array()
        da00_vars = scipp_to_da00(da)
        da_restored = da00_to_scipp(da00_vars)
        restored_roi = models.ROI.from_data_array(da_restored)

        assert original_roi == restored_roi
