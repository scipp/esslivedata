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

    def test_encoding_special_characters_in_source_name(self):
        """Test that special characters in source_name are properly encoded."""
        # JobId.str() produces "source_name/job_number" with a slash
        key = models.ConfigKey(
            source_name="mantle_detector/87529091-604f-402a-b983-7ef190661bf5",
            service_name=None,
            key="job_command",
        )
        string_repr = str(key)
        # Should have exactly 3 parts when split by /
        assert string_repr.count('/') == 2
        # Should be able to parse it back
        parsed = models.ConfigKey.from_string(string_repr)
        assert (
            parsed.source_name == "mantle_detector/87529091-604f-402a-b983-7ef190661bf5"
        )
        assert parsed.service_name is None
        assert parsed.key == "job_command"

    def test_encoding_special_characters_in_service_name(self):
        """Test that special characters in service_name are properly encoded."""
        key = models.ConfigKey(
            source_name="source1",
            service_name="service/with/slashes",
            key="test_key",
        )
        string_repr = str(key)
        # Should have exactly 3 parts when split by /
        assert string_repr.count('/') == 2
        parsed = models.ConfigKey.from_string(string_repr)
        assert parsed.source_name == "source1"
        assert parsed.service_name == "service/with/slashes"
        assert parsed.key == "test_key"

    def test_encoding_special_characters_in_key(self):
        """Test that special characters in key are properly encoded."""
        key = models.ConfigKey(
            source_name="source1",
            service_name="service1",
            key="key/with/slashes",
        )
        string_repr = str(key)
        # Should have exactly 3 parts when split by /
        assert string_repr.count('/') == 2
        parsed = models.ConfigKey.from_string(string_repr)
        assert parsed.source_name == "source1"
        assert parsed.service_name == "service1"
        assert parsed.key == "key/with/slashes"

    def test_encoding_url_special_characters(self):
        """Test URL encoding of various special characters."""
        # Test various URL special characters that should be encoded
        key = models.ConfigKey(
            source_name="source with spaces",
            service_name="service?query=value",
            key="key#fragment",
        )
        string_repr = str(key)
        parsed = models.ConfigKey.from_string(string_repr)
        assert parsed.source_name == "source with spaces"
        assert parsed.service_name == "service?query=value"
        assert parsed.key == "key#fragment"

    def test_roundtrip_with_special_characters(self):
        """Test complete roundtrip with all special characters."""
        test_cases = [
            ("source/name", "service/name", "key/name"),
            ("a/b/c", "d/e/f", "g/h/i"),
            ("source with spaces", "service?query", "key#frag"),
            ("source%encoded", "service&param", "key=value"),
        ]
        for source, service, key_val in test_cases:
            original = models.ConfigKey(
                source_name=source, service_name=service, key=key_val
            )
            string_repr = str(original)
            # Ensure exactly 3 parts
            assert string_repr.count('/') == 2
            parsed = models.ConfigKey.from_string(string_repr)
            assert parsed.source_name == source
            assert parsed.service_name == service
            assert parsed.key == key_val


class TestInterval:
    def test_creation(self):
        interval = models.Interval(min=10.0, max=20.0, unit='mm')
        assert interval.min == 10.0
        assert interval.max == 20.0
        assert interval.unit == 'mm'

    def test_validation_bounds(self):
        with pytest.raises(ValidationError, match=r"min .* must be < max"):
            models.Interval(min=20.0, max=10.0, unit='mm')

    def test_to_bounds_with_unit(self):
        interval = models.Interval(min=10.0, max=20.0, unit='mm')
        bounds = interval.to_bounds()
        assert len(bounds) == 2
        assert isinstance(bounds[0], sc.Variable)
        assert isinstance(bounds[1], sc.Variable)
        assert bounds[0].value == 10.0
        assert bounds[1].value == 20.0
        assert str(bounds[0].unit) == 'mm'
        assert str(bounds[1].unit) == 'mm'

    def test_to_bounds_without_unit(self):
        interval = models.Interval(min=1.0, max=5.0, unit=None)
        bounds = interval.to_bounds()
        assert bounds == (1, 5)
        assert isinstance(bounds[0], int)
        assert isinstance(bounds[1], int)

    def test_default_unit_is_none(self):
        interval = models.Interval(min=1.0, max=5.0)
        assert interval.unit is None


class TestRectangleROI:
    def test_creation(self):
        roi = models.RectangleROI(
            x=models.Interval(min=10.0, max=20.0, unit='mm'),
            y=models.Interval(min=5.0, max=15.0, unit='mm'),
        )
        assert roi.x.min == 10.0
        assert roi.x.max == 20.0
        assert roi.y.min == 5.0
        assert roi.y.max == 15.0
        assert roi.x.unit == 'mm'
        assert roi.y.unit == 'mm'

    def test_validation_x_bounds(self):
        with pytest.raises(ValidationError, match=r"min .* must be < max"):
            models.RectangleROI(
                x=models.Interval(min=20.0, max=10.0, unit='mm'),
                y=models.Interval(min=5.0, max=15.0, unit='mm'),
            )

    def test_validation_y_bounds(self):
        with pytest.raises(ValidationError, match=r"min .* must be < max"):
            models.RectangleROI(
                x=models.Interval(min=10.0, max=20.0, unit='mm'),
                y=models.Interval(min=15.0, max=5.0, unit='mm'),
            )

    def test_to_data_array(self):
        roi = models.RectangleROI(
            x=models.Interval(min=10.0, max=20.0, unit='mm'),
            y=models.Interval(min=5.0, max=15.0, unit='mm'),
        )
        da = roi.to_data_array()

        assert list(da.dims) == ['bounds']
        assert da.shape == (2,)
        assert 'x' in da.coords
        assert 'y' in da.coords
        np.testing.assert_array_equal(da.coords['x'].values, [10.0, 20.0])
        np.testing.assert_array_equal(da.coords['y'].values, [5.0, 15.0])
        assert str(da.coords['x'].unit) == 'mm'
        assert str(da.coords['y'].unit) == 'mm'

    def test_from_data_array(self):
        # Create a DataArray with 'bounds' dimension (identifies rectangle type)
        data = sc.array(dims=['bounds'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'x': sc.array(dims=['bounds'], values=[10.0, 20.0], unit='mm'),
            'y': sc.array(dims=['bounds'], values=[5.0, 15.0], unit='mm'),
        }
        da = sc.DataArray(data, coords=coords)

        # Convert back to ROI
        roi = models.ROI.from_data_array(da)

        assert isinstance(roi, models.RectangleROI)
        assert roi.x.min == 10.0
        assert roi.x.max == 20.0
        assert roi.y.min == 5.0
        assert roi.y.max == 15.0
        assert roi.x.unit == 'mm'
        assert roi.y.unit == 'mm'

    def test_roundtrip_conversion(self):
        original = models.RectangleROI(
            x=models.Interval(min=10.0, max=20.0, unit='mm'),
            y=models.Interval(min=5.0, max=15.0, unit='mm'),
        )
        da = original.to_data_array()
        restored = models.ROI.from_data_array(da)
        assert original == restored

    def test_different_units(self):
        roi = models.RectangleROI(
            x=models.Interval(min=10.0, max=20.0, unit='mm'),
            y=models.Interval(min=5.0, max=15.0, unit='m'),
        )
        da = roi.to_data_array()
        restored = models.ROI.from_data_array(da)
        assert restored.x.unit == 'mm'
        assert restored.y.unit == 'm'

    def test_none_units_with_float_values(self):
        # None units allow floats for sub-pixel precision
        roi = models.RectangleROI(
            x=models.Interval(min=10.5, max=20.5, unit=None),
            y=models.Interval(min=5.0, max=15.0, unit=None),
        )
        assert roi.x.unit is None
        assert roi.y.unit is None
        assert roi.x.min == 10.5
        assert roi.x.max == 20.5

    def test_none_units_roundtrip(self):
        original = models.RectangleROI(
            x=models.Interval(min=10.0, max=20.0, unit=None),
            y=models.Interval(min=5.0, max=15.0, unit=None),
        )
        da = original.to_data_array()

        # Check that coordinates have None unit
        assert da.coords['x'].unit is None
        assert da.coords['y'].unit is None

        # Roundtrip conversion
        restored = models.ROI.from_data_array(da)
        assert restored.x.unit is None
        assert restored.y.unit is None
        assert original == restored

    def test_mixed_units(self):
        # One dimension with unit, one without
        roi = models.RectangleROI(
            x=models.Interval(min=10.0, max=20.0, unit='mm'),
            y=models.Interval(min=5.0, max=15.0, unit=None),
        )
        da = roi.to_data_array()
        restored = models.ROI.from_data_array(da)
        assert restored.x.unit == 'mm'
        assert restored.y.unit is None

    def test_get_bounds_with_units(self):
        roi = models.RectangleROI(
            x=models.Interval(min=10.0, max=20.0, unit='mm'),
            y=models.Interval(min=5.0, max=15.0, unit='mm'),
        )
        bounds = roi.get_bounds(x_dim='x', y_dim='y')
        assert 'x' in bounds
        assert 'y' in bounds
        assert isinstance(bounds['x'][0], sc.Variable)
        assert isinstance(bounds['y'][0], sc.Variable)

    def test_get_bounds_without_units(self):
        roi = models.RectangleROI(
            x=models.Interval(min=1.0, max=3.0, unit=None),
            y=models.Interval(min=2.0, max=4.0, unit=None),
        )
        bounds = roi.get_bounds(x_dim='x', y_dim='y')
        assert bounds['x'] == (1, 3)
        assert bounds['y'] == (2, 4)


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

        assert list(da.dims) == ['vertex']
        assert da.shape == (3,)
        assert 'x' in da.coords
        assert 'y' in da.coords
        np.testing.assert_array_equal(da.coords['x'].values, [0.0, 10.0, 5.0])
        np.testing.assert_array_equal(da.coords['y'].values, [0.0, 0.0, 10.0])

    def test_from_data_array(self):
        # Create a DataArray with 'vertex' dimension (identifies polygon type)
        n = 4
        data = sc.array(dims=['vertex'], values=np.ones(n, dtype=np.int32), unit='')
        coords = {
            'x': sc.array(dims=['vertex'], values=[0.0, 10.0, 10.0, 0.0], unit='mm'),
            'y': sc.array(dims=['vertex'], values=[0.0, 0.0, 10.0, 10.0], unit='mm'),
        }
        da = sc.DataArray(data, coords=coords)

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

    def test_none_units_with_float_vertices(self):
        # None units allow floats for sub-pixel precision
        roi = models.PolygonROI(
            x=[0.5, 10.5, 5.0], y=[0.0, 0.0, 10.5], x_unit=None, y_unit=None
        )
        assert roi.x_unit is None
        assert roi.y_unit is None
        assert roi.x == [0.5, 10.5, 5.0]
        assert roi.y == [0.0, 0.0, 10.5]

    def test_none_units_roundtrip(self):
        original = models.PolygonROI(
            x=[0.0, 10.0, 5.0, 2.0], y=[0.0, 0.0, 10.0, 5.0], x_unit=None, y_unit=None
        )
        da = original.to_data_array()

        # Check that coordinates have None unit
        assert da.coords['x'].unit is None
        assert da.coords['y'].unit is None

        # Roundtrip conversion
        restored = models.ROI.from_data_array(da)
        assert restored.x_unit is None
        assert restored.y_unit is None
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

        assert list(da.dims) == ['ellipse']
        assert da.shape == (2,)
        assert 'center' in da.coords
        assert 'radius' in da.coords
        assert 'rotation' in da.coords
        np.testing.assert_array_equal(da.coords['center'].values, [10.0, 20.0])
        np.testing.assert_array_equal(da.coords['radius'].values, [5.0, 3.0])
        assert da.coords['rotation'].value == 45.0
        assert str(da.coords['rotation'].unit) == 'deg'

    def test_from_data_array(self):
        # Create a DataArray with 'ellipse' dimension (identifies ellipse type)
        data = sc.array(dims=['ellipse'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'center': sc.array(dims=['ellipse'], values=[10.0, 20.0], unit='mm'),
            'radius': sc.array(dims=['ellipse'], values=[5.0, 3.0], unit='mm'),
        }
        da = sc.DataArray(data, coords=coords)
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
        data = sc.array(dims=['ellipse'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'center': sc.array(dims=['ellipse'], values=[10.0, 20.0], unit='mm'),
            'radius': sc.array(dims=['ellipse'], values=[5.0, 3.0], unit='mm'),
        }
        da = sc.DataArray(data, coords=coords)

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

    def test_none_unit_with_float_values(self):
        # None unit allows floats for sub-pixel precision
        roi = models.EllipseROI(
            center_x=10.5,
            center_y=20.5,
            radius_x=5.5,
            radius_y=3.5,
            rotation=45.0,
            unit=None,
        )
        assert roi.unit is None
        assert roi.center_x == 10.5
        assert roi.center_y == 20.5
        assert roi.radius_x == 5.5
        assert roi.radius_y == 3.5

    def test_none_unit_roundtrip(self):
        original = models.EllipseROI(
            center_x=10.0,
            center_y=20.0,
            radius_x=5.0,
            radius_y=3.0,
            rotation=45.0,
            unit=None,
        )
        da = original.to_data_array()

        # Check that coordinates have None unit
        assert da.coords['center'].unit is None
        assert da.coords['radius'].unit is None

        # Roundtrip conversion
        restored = models.ROI.from_data_array(da)
        assert restored.unit is None
        assert original == restored


class TestROIDispatch:
    def test_from_data_array_unknown_dimension(self):
        """Unknown dimension name should raise ValueError."""
        data = sc.array(dims=['unknown'], values=[1, 1], dtype='int32', unit='')
        coords = {
            'x': sc.array(dims=['unknown'], values=[10.0, 20.0], unit='mm'),
            'y': sc.array(dims=['unknown'], values=[5.0, 15.0], unit='mm'),
        }
        da = sc.DataArray(data, coords=coords)

        with pytest.raises(
            ValueError, match="Cannot determine ROI type from dimension"
        ):
            models.ROI.from_data_array(da)


class TestMultipleRectangleROI:
    """Test serialization of multiple rectangles into single DataArray."""

    def test_concatenate_multiple_rectangles(self):
        """Multiple rectangles should be concatenated along bounds dimension."""
        rois = {
            0: models.RectangleROI(
                x=models.Interval(min=10.0, max=20.0, unit='mm'),
                y=models.Interval(min=30.0, max=40.0, unit='mm'),
            ),
            1: models.RectangleROI(
                x=models.Interval(min=50.0, max=60.0, unit='mm'),
                y=models.Interval(min=70.0, max=80.0, unit='mm'),
            ),
            2: models.RectangleROI(
                x=models.Interval(min=100.0, max=110.0, unit='mm'),
                y=models.Interval(min=120.0, max=130.0, unit='mm'),
            ),
        }

        da = models.RectangleROI.to_concatenated_data_array(rois)

        # Should have bounds dimension with 6 elements (3 ROIs x 2 bounds each)
        assert list(da.dims) == ['bounds']
        assert da.shape == (6,)

        # Should have roi_index coordinate mapping bounds to ROIs
        assert 'roi_index' in da.coords
        np.testing.assert_array_equal(da.coords['roi_index'].values, [0, 0, 1, 1, 2, 2])

        # x and y coordinates should be concatenated
        np.testing.assert_array_equal(
            da.coords['x'].values, [10.0, 20.0, 50.0, 60.0, 100.0, 110.0]
        )
        np.testing.assert_array_equal(
            da.coords['y'].values, [30.0, 40.0, 70.0, 80.0, 120.0, 130.0]
        )

    def test_concatenate_empty_dict(self):
        """Empty dict should produce empty DataArray."""
        rois = {}
        da = models.RectangleROI.to_concatenated_data_array(rois)

        assert list(da.dims) == ['bounds']
        assert da.shape == (0,)
        assert 'roi_index' in da.coords
        assert len(da.coords['roi_index']) == 0

    def test_concatenate_single_rectangle(self):
        """Single rectangle should work (edge case)."""
        rois = {
            0: models.RectangleROI(
                x=models.Interval(min=10.0, max=20.0, unit='mm'),
                y=models.Interval(min=30.0, max=40.0, unit='mm'),
            ),
        }

        da = models.RectangleROI.to_concatenated_data_array(rois)

        assert da.shape == (2,)
        np.testing.assert_array_equal(da.coords['roi_index'].values, [0, 0])
        np.testing.assert_array_equal(da.coords['x'].values, [10.0, 20.0])

    def test_from_concatenated_data_array(self):
        """Should reconstruct dict of ROIs from concatenated DataArray."""
        # Create concatenated DataArray ('bounds' dimension identifies rectangles)
        data = sc.ones(dims=['bounds'], shape=[4], dtype='int32', unit='')
        coords = {
            'x': sc.array(dims=['bounds'], values=[10.0, 20.0, 50.0, 60.0], unit='mm'),
            'y': sc.array(dims=['bounds'], values=[30.0, 40.0, 70.0, 80.0], unit='mm'),
            'roi_index': sc.array(dims=['bounds'], values=[0, 0, 1, 1], dtype='int32'),
        }
        da = sc.DataArray(data, coords=coords)

        rois = models.RectangleROI.from_concatenated_data_array(da)

        assert len(rois) == 2
        assert 0 in rois
        assert 1 in rois

        assert rois[0].x.min == 10.0
        assert rois[0].x.max == 20.0
        assert rois[0].y.min == 30.0
        assert rois[0].y.max == 40.0

        assert rois[1].x.min == 50.0
        assert rois[1].x.max == 60.0
        assert rois[1].y.min == 70.0
        assert rois[1].y.max == 80.0

    def test_from_concatenated_empty_data_array(self):
        """Empty DataArray should produce empty dict."""
        data = sc.empty(dims=['bounds'], shape=[0], dtype='int32', unit='')
        coords = {
            'x': sc.empty(dims=['bounds'], shape=[0], unit='mm'),
            'y': sc.empty(dims=['bounds'], shape=[0], unit='mm'),
            'roi_index': sc.empty(dims=['bounds'], shape=[0], dtype='int32'),
        }
        da = sc.DataArray(data, coords=coords, name='rectangles')

        rois = models.RectangleROI.from_concatenated_data_array(da)
        assert rois == {}

    def test_roundtrip_concatenated_rectangles(self):
        """Roundtrip: dict → concatenated DataArray → dict."""
        original = {
            0: models.RectangleROI(
                x=models.Interval(min=10.0, max=20.0, unit='mm'),
                y=models.Interval(min=30.0, max=40.0, unit='mm'),
            ),
            2: models.RectangleROI(
                x=models.Interval(min=50.0, max=60.0, unit='mm'),
                y=models.Interval(min=70.0, max=80.0, unit='mm'),
            ),
        }

        da = models.RectangleROI.to_concatenated_data_array(original)
        restored = models.RectangleROI.from_concatenated_data_array(da)

        assert original == restored

    def test_concatenated_rectangles_none_units(self):
        """Should handle None units (pixel coordinates)."""
        rois = {
            0: models.RectangleROI(
                x=models.Interval(min=10.0, max=20.0, unit=None),
                y=models.Interval(min=30.0, max=40.0, unit=None),
            ),
            1: models.RectangleROI(
                x=models.Interval(min=50.0, max=60.0, unit=None),
                y=models.Interval(min=70.0, max=80.0, unit=None),
            ),
        }

        da = models.RectangleROI.to_concatenated_data_array(rois)
        restored = models.RectangleROI.from_concatenated_data_array(da)

        assert restored[0].x.unit is None
        assert restored[0].y.unit is None
        assert rois == restored


class TestMultiplePolygonROI:
    """Test serialization of multiple polygons into single DataArray."""

    def test_concatenate_multiple_polygons(self):
        """Multiple polygons should be concatenated along vertex dimension."""
        rois = {
            0: models.PolygonROI(
                x=[0.0, 10.0, 5.0], y=[0.0, 0.0, 10.0], x_unit='mm', y_unit='mm'
            ),
            1: models.PolygonROI(
                x=[20.0, 30.0, 25.0, 20.0],
                y=[20.0, 20.0, 30.0, 25.0],
                x_unit='mm',
                y_unit='mm',
            ),
        }

        da = models.PolygonROI.to_concatenated_data_array(rois)

        # Should have vertex dimension with 7 elements (3 + 4 vertices)
        assert list(da.dims) == ['vertex']
        assert da.shape == (7,)

        # Should have roi_index coordinate mapping vertices to polygons
        assert 'roi_index' in da.coords
        np.testing.assert_array_equal(
            da.coords['roi_index'].values, [0, 0, 0, 1, 1, 1, 1]
        )

    def test_concatenate_empty_dict(self):
        """Empty dict should produce empty DataArray."""
        rois = {}
        da = models.PolygonROI.to_concatenated_data_array(rois)

        assert list(da.dims) == ['vertex']
        assert da.shape == (0,)
        assert 'roi_index' in da.coords
        assert len(da.coords['roi_index']) == 0

    def test_from_concatenated_data_array(self):
        """Should reconstruct dict of polygons from concatenated DataArray."""
        # The 'vertex' dimension identifies this as polygons
        data = sc.ones(dims=['vertex'], shape=[7], dtype='int32', unit='')
        coords = {
            'x': sc.array(
                dims=['vertex'],
                values=[0.0, 10.0, 5.0, 20.0, 30.0, 25.0, 20.0],
                unit='mm',
            ),
            'y': sc.array(
                dims=['vertex'],
                values=[0.0, 0.0, 10.0, 20.0, 20.0, 30.0, 25.0],
                unit='mm',
            ),
            'roi_index': sc.array(
                dims=['vertex'], values=[0, 0, 0, 1, 1, 1, 1], dtype='int32'
            ),
        }
        da = sc.DataArray(data, coords=coords)

        rois = models.PolygonROI.from_concatenated_data_array(da)

        assert len(rois) == 2
        assert len(rois[0].x) == 3
        assert len(rois[1].x) == 4

    def test_roundtrip_concatenated_polygons(self):
        """Roundtrip: dict → concatenated DataArray → dict."""
        original = {
            0: models.PolygonROI(
                x=[0.0, 10.0, 5.0], y=[0.0, 0.0, 10.0], x_unit='mm', y_unit='mm'
            ),
            2: models.PolygonROI(
                x=[20.0, 30.0, 25.0, 20.0],
                y=[20.0, 20.0, 30.0, 25.0],
                x_unit='mm',
                y_unit='mm',
            ),
        }

        da = models.PolygonROI.to_concatenated_data_array(original)
        restored = models.PolygonROI.from_concatenated_data_array(da)

        assert original == restored


class TestMultipleEllipseROI:
    """Test serialization of multiple ellipses into single DataArray."""

    def test_concatenate_multiple_ellipses(self):
        """Multiple ellipses should be concatenated along dim dimension."""
        rois = {
            0: models.EllipseROI(
                center_x=10.0,
                center_y=20.0,
                radius_x=5.0,
                radius_y=3.0,
                rotation=45.0,
                unit='mm',
            ),
            1: models.EllipseROI(
                center_x=50.0,
                center_y=60.0,
                radius_x=8.0,
                radius_y=4.0,
                rotation=90.0,
                unit='mm',
            ),
        }

        da = models.EllipseROI.to_concatenated_data_array(rois)

        # Should have ellipse dimension with 4 elements (2 ROIs x 2 dims each)
        assert list(da.dims) == ['ellipse']
        assert da.shape == (4,)

        # Should have roi_index coordinate mapping dims to ellipses
        assert 'roi_index' in da.coords
        np.testing.assert_array_equal(da.coords['roi_index'].values, [0, 0, 1, 1])

    def test_concatenate_empty_dict(self):
        """Empty dict should produce empty DataArray."""
        rois = {}
        da = models.EllipseROI.to_concatenated_data_array(rois)

        assert list(da.dims) == ['ellipse']
        assert da.shape == (0,)
        assert 'roi_index' in da.coords
        assert len(da.coords['roi_index']) == 0

    def test_from_concatenated_data_array(self):
        """Should reconstruct dict of ellipses from concatenated DataArray."""
        # The 'ellipse' dimension identifies this as ellipses
        data = sc.ones(dims=['ellipse'], shape=[4], dtype='int32', unit='')
        coords = {
            'center': sc.array(
                dims=['ellipse'], values=[10.0, 20.0, 50.0, 60.0], unit='mm'
            ),
            'radius': sc.array(
                dims=['ellipse'], values=[5.0, 3.0, 8.0, 4.0], unit='mm'
            ),
            'rotation': sc.array(
                dims=['ellipse'], values=[45.0, 45.0, 90.0, 90.0], unit='deg'
            ),
            'roi_index': sc.array(dims=['ellipse'], values=[0, 0, 1, 1], dtype='int32'),
        }
        da = sc.DataArray(data, coords=coords)

        rois = models.EllipseROI.from_concatenated_data_array(da)

        assert len(rois) == 2
        assert rois[0].center_x == 10.0
        assert rois[0].center_y == 20.0
        assert rois[0].rotation == 45.0
        assert rois[1].center_x == 50.0
        assert rois[1].center_y == 60.0
        assert rois[1].rotation == 90.0

    def test_roundtrip_concatenated_ellipses(self):
        """Roundtrip: dict → concatenated DataArray → dict."""
        original = {
            0: models.EllipseROI(
                center_x=10.0,
                center_y=20.0,
                radius_x=5.0,
                radius_y=3.0,
                rotation=45.0,
                unit='mm',
            ),
            2: models.EllipseROI(
                center_x=50.0,
                center_y=60.0,
                radius_x=8.0,
                radius_y=4.0,
                rotation=90.0,
                unit='mm',
            ),
        }

        da = models.EllipseROI.to_concatenated_data_array(original)
        restored = models.EllipseROI.from_concatenated_data_array(da)

        assert original == restored


class TestROIWithDa00:
    """Test ROI roundtrip through da00 serialization."""

    def test_rectangle_through_da00(self):
        from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00

        # Create ROI
        original_roi = models.RectangleROI(
            x=models.Interval(min=10.0, max=20.0, unit='mm'),
            y=models.Interval(min=5.0, max=15.0, unit='mm'),
        )

        # Convert to DataArray
        da = original_roi.to_data_array()

        # Serialize through da00
        da00_vars = scipp_to_da00(da)
        da_restored = da00_to_scipp(da00_vars)

        # Convert back to ROI
        restored_roi = models.ROI.from_data_array(da_restored)

        assert original_roi == restored_roi

    def test_concatenated_rectangles_through_da00(self):
        """Test that concatenated rectangles roundtrip through da00."""
        from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00

        original_rois = {
            0: models.RectangleROI(
                x=models.Interval(min=10.0, max=20.0, unit='mm'),
                y=models.Interval(min=30.0, max=40.0, unit='mm'),
            ),
            1: models.RectangleROI(
                x=models.Interval(min=50.0, max=60.0, unit='mm'),
                y=models.Interval(min=70.0, max=80.0, unit='mm'),
            ),
        }

        # Convert to concatenated DataArray
        da = models.RectangleROI.to_concatenated_data_array(original_rois)

        # Serialize through da00
        da00_vars = scipp_to_da00(da)
        da_restored = da00_to_scipp(da00_vars)

        # Convert back to ROIs
        restored_rois = models.RectangleROI.from_concatenated_data_array(da_restored)

        assert original_rois == restored_rois

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
