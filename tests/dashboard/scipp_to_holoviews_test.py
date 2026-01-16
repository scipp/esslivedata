# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import holoviews as hv
import numpy as np
import pytest
import scipp as sc

from ess.livedata.dashboard import scipp_to_holoviews


class TestHelperFunctions:
    @pytest.mark.parametrize("unit", ['m', 'dimensionless', None])
    def test_coord_to_dimension(self, unit: str | None):
        coord = sc.array(dims=['x'], values=[1, 2, 3], unit=unit)
        dim = scipp_to_holoviews.coord_to_dimension(coord)

        assert dim.name == 'x'
        assert dim.label == 'x'
        assert dim.unit == unit

    @pytest.mark.parametrize("unit", ['m', 'dimensionless', None])
    def test_value_dimension_with_name_and_unit(self, unit: str | None):
        data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1, 2, 3], unit=unit),
            coords={'x': sc.array(dims=['x'], values=[0, 1, 2])},
            name='intensity',
        )
        dim = scipp_to_holoviews.create_value_dimension(data)

        assert dim.name == 'values'
        assert dim.label == 'intensity'
        assert dim.unit == unit

    def test_value_dimension_without_name(self):
        data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1, 2, 3], unit='m/s'),
            coords={'x': sc.array(dims=['x'], values=[0, 1, 2])},
        )
        dim = scipp_to_holoviews.create_value_dimension(data)

        assert dim.name == 'values'
        assert dim.label == 'values'
        assert dim.unit == 'm/s'


class TestConvertHistogram1d:
    def test_basic_conversion(self):
        # Create histogram data with bin edges
        edges = sc.array(dims=['x'], values=[0, 1, 2, 3], unit='m')
        values = sc.array(dims=['x'], values=[10, 20, 30], unit='counts')
        data = sc.DataArray(data=values, coords={'x': edges})

        result = scipp_to_holoviews.convert_histogram_1d(data)

        assert isinstance(result, hv.Histogram)
        assert len(result.kdims) == 1
        assert len(result.vdims) == 1
        assert result.kdims[0].name == 'x'
        assert result.kdims[0].unit == 'm'
        assert result.vdims[0].unit == 'counts'

        # Check data - holoviews data is accessed by column names
        hist_data = result.data
        np.testing.assert_array_equal(hist_data['x'], [0, 1, 2, 3])
        np.testing.assert_array_equal(hist_data['values'], [10, 20, 30])

    def test_with_named_data(self):
        edges = sc.array(dims=['energy'], values=[0, 1, 2, 3], unit='eV')
        values = sc.array(dims=['energy'], values=[100, 200, 300], unit='Hz')
        data = sc.DataArray(data=values, coords={'energy': edges}, name='spectrum')

        result = scipp_to_holoviews.convert_histogram_1d(data)

        assert result.vdims[0].label == 'spectrum'
        assert result.kdims[0].name == 'energy'

    def test_with_missing_coord_raises(self):
        values = sc.array(dims=['x'], values=[10, 20, 30], unit='counts')
        data = sc.DataArray(data=values)

        with pytest.raises(KeyError, match="Expected 'x'"):
            scipp_to_holoviews.convert_histogram_1d(data)


class TestConvertCurve1d:
    def test_basic_conversion(self):
        x_coord = sc.array(dims=['x'], values=[1, 2, 3, 4], unit='s')
        y_values = sc.array(dims=['x'], values=[10, 15, 20, 25], unit='V')
        data = sc.DataArray(data=y_values, coords={'x': x_coord})

        result = scipp_to_holoviews.convert_curve_1d(data)

        assert isinstance(result, hv.Curve)
        assert len(result.kdims) == 1
        assert len(result.vdims) == 1
        assert result.kdims[0].name == 'x'
        assert result.kdims[0].unit == 's'
        assert result.vdims[0].unit == 'V'

        # Check data - holoviews data is accessed by column names
        curve_data = result.data
        np.testing.assert_array_equal(curve_data['x'], [1, 2, 3, 4])
        np.testing.assert_array_equal(curve_data['values'], [10, 15, 20, 25])

    def test_with_named_data(self):
        coord = sc.array(dims=['time'], values=[0, 1, 2])
        values = sc.array(dims=['time'], values=[5, 10, 15])
        data = sc.DataArray(data=values, coords={'time': coord}, name='signal')

        result = scipp_to_holoviews.convert_curve_1d(data)

        assert result.vdims[0].label == 'signal'

    def test_with_missing_coord(self):
        values = sc.array(dims=['time'], values=[5, 10, 15], unit='V')
        data = sc.DataArray(data=values, name='signal')

        result = scipp_to_holoviews.convert_curve_1d(data)

        assert isinstance(result, hv.Curve)
        assert result.kdims[0].name == 'time'
        assert result.kdims[0].unit is None  # Dummy coord has no unit
        assert result.vdims[0].label == 'signal'

        # Check data contains dummy coordinates
        curve_data = result.data
        np.testing.assert_array_equal(curve_data['time'], [0, 1, 2])  # dummy coords
        np.testing.assert_array_equal(curve_data['values'], [5, 10, 15])


class TestConvertErrorBars1d:
    def test_basic_conversion(self):
        x_coord = sc.array(dims=['x'], values=[1, 2, 3, 4], unit='s')
        y_values = sc.array(dims=['x'], values=[10.0, 15.0, 20.0, 25.0], unit='V')
        y_variances = sc.array(dims=['x'], values=[1.0, 2.0, 1.5, 3.0], unit='V^2')
        data = sc.DataArray(
            data=sc.array(
                dims=['x'],
                values=y_values.values,
                variances=y_variances.values,
                unit='V',
            ),
            coords={'x': x_coord},
        )

        result = scipp_to_holoviews.convert_error_bars_1d(data)

        assert isinstance(result, hv.ErrorBars)
        assert len(result.kdims) == 1
        assert len(result.vdims) == 2  # values and yerr
        assert result.kdims[0].name == 'x'
        assert result.kdims[0].unit == 's'
        assert result.vdims[0].unit == 'V'

        # Check data
        error_data = result.data
        np.testing.assert_array_equal(error_data['x'], [1, 2, 3, 4])
        np.testing.assert_array_equal(error_data['values'], [10.0, 15.0, 20.0, 25.0])
        np.testing.assert_array_almost_equal(
            error_data['yerr'], [1, np.sqrt(2), np.sqrt(1.5), np.sqrt(3)]
        )

    def test_with_named_data(self):
        coord = sc.array(dims=['time'], values=[0, 1, 2])
        values = sc.array(dims=['time'], values=[5.0, 10.0, 15.0])
        variances = sc.array(dims=['time'], values=[0.25, 1.0, 2.25])
        data = sc.DataArray(
            data=sc.array(
                dims=['time'], values=values.values, variances=variances.values
            ),
            coords={'time': coord},
            name='signal',
        )

        result = scipp_to_holoviews.convert_error_bars_1d(data)

        assert result.vdims[0].label == 'signal'

    def test_with_missing_coord(self):
        values = sc.array(dims=['time'], values=[5.0, 10.0, 15.0], unit='V')
        variances = sc.array(dims=['time'], values=[1.0, 4.0, 9.0], unit='V^2')
        data = sc.DataArray(
            data=sc.array(
                dims=['time'],
                values=values.values,
                variances=variances.values,
                unit='V',
            ),
            name='signal',
        )

        result = scipp_to_holoviews.convert_error_bars_1d(data)

        assert isinstance(result, hv.ErrorBars)
        assert result.kdims[0].name == 'time'
        assert result.kdims[0].unit is None  # Dummy coord has no unit
        assert result.vdims[0].label == 'signal'

        # Check data contains dummy coordinates
        error_data = result.data
        np.testing.assert_array_equal(error_data['time'], [0, 1, 2])  # dummy coords
        np.testing.assert_array_equal(error_data['values'], [5.0, 10.0, 15.0])
        np.testing.assert_array_equal(
            error_data['yerr'], [1, 2, 3]
        )  # sqrt of variances

    def test_zero_variances(self):
        x_coord = sc.array(dims=['x'], values=[1, 2, 3])
        y_values = sc.array(dims=['x'], values=[10.0, 20.0, 30.0])
        y_variances = sc.array(dims=['x'], values=[0.0, 0.0, 0.0])  # Zero variances
        data = sc.DataArray(
            data=sc.array(
                dims=['x'], values=y_values.values, variances=y_variances.values
            ),
            coords={'x': x_coord},
        )

        result = scipp_to_holoviews.convert_error_bars_1d(data)

        assert isinstance(result, hv.ErrorBars)
        error_data = result.data
        np.testing.assert_array_equal(error_data['yerr'], [0, 0, 0])


class TestConvertQuadMesh2d:
    def test_basic_conversion(self):
        x_coord = sc.array(dims=['x'], values=[1, 2, 3], unit='m')
        y_coord = sc.array(dims=['y'], values=[4, 5], unit='s')
        values = sc.array(
            dims=['y', 'x'], values=[[10, 20, 30], [40, 50, 60]], unit='K'
        )
        data = sc.DataArray(data=values, coords={'x': x_coord, 'y': y_coord})

        result = scipp_to_holoviews.convert_quadmesh_2d(data)

        assert isinstance(result, hv.QuadMesh)
        assert len(result.kdims) == 2
        assert len(result.vdims) == 1

        # Dimensions should be reversed (x first, then y)
        assert result.kdims[0].name == 'x'
        assert result.kdims[0].unit == 'm'
        assert result.kdims[1].name == 'y'
        assert result.kdims[1].unit == 's'
        assert result.vdims[0].unit == 'K'

    def test_irregular_coordinates(self):
        # Non-evenly spaced coordinates
        x_coord = sc.array(dims=['x'], values=[1, 3, 10], unit='mm')
        y_coord = sc.array(dims=['y'], values=[0, 2, 5, 20], unit='Hz')
        values = sc.array(
            dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        )
        data = sc.DataArray(data=values, coords={'x': x_coord, 'y': y_coord})

        result = scipp_to_holoviews.convert_quadmesh_2d(data)

        assert isinstance(result, hv.QuadMesh)
        mesh_data = result.data
        np.testing.assert_array_equal(mesh_data['x'], [1, 3, 10])  # x coords
        np.testing.assert_array_equal(mesh_data['y'], [0, 2, 5, 20])  # y coords

    def test_with_bin_edges(self):
        # Test with bin edges (N+1 coordinates for N bins)
        x_edges = sc.array(dims=['x'], values=[0, 1, 2, 3], unit='m')
        y_edges = sc.array(dims=['y'], values=[0, 10, 20], unit='s')
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]], unit='counts')
        data = sc.DataArray(data=values, coords={'x': x_edges, 'y': y_edges})

        result = scipp_to_holoviews.convert_quadmesh_2d(data)

        assert isinstance(result, hv.QuadMesh)
        mesh_data = result.data
        # QuadMesh should preserve the bin edges
        np.testing.assert_array_equal(mesh_data['x'], [0, 1, 2, 3])  # x edges
        np.testing.assert_array_equal(mesh_data['y'], [0, 10, 20])  # y edges
        np.testing.assert_array_equal(mesh_data['values'], [[1, 2, 3], [4, 5, 6]])

    def test_with_mixed_midpoints_and_bin_edges(self):
        # Test with bin edges (N+1 coordinates for N bins) only along y
        x_coord = sc.array(dims=['x'], values=[0, 1, 2], unit='m')
        y_edges = sc.array(dims=['y'], values=[0, 10, 20], unit='s')
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]], unit='counts')
        data = sc.DataArray(data=values, coords={'x': x_coord, 'y': y_edges})

        result = scipp_to_holoviews.convert_quadmesh_2d(data)

        assert isinstance(result, hv.QuadMesh)
        mesh_data = result.data
        # QuadMesh should preserve the bin edges
        np.testing.assert_array_equal(mesh_data['x'], [0, 1, 2])  # x preserved
        np.testing.assert_array_equal(mesh_data['y'], [0, 10, 20])  # y edges
        np.testing.assert_array_equal(mesh_data['values'], [[1, 2, 3], [4, 5, 6]])

    def test_with_missing_x_coord(self):
        y_coord = sc.array(dims=['y'], values=[4, 5], unit='s')
        values = sc.array(
            dims=['y', 'x'], values=[[10, 20, 30], [40, 50, 60]], unit='K'
        )
        data = sc.DataArray(data=values, coords={'y': y_coord})

        result = scipp_to_holoviews.convert_quadmesh_2d(data)

        assert isinstance(result, hv.QuadMesh)
        assert len(result.kdims) == 2
        assert len(result.vdims) == 1

        # Dimensions should be reversed (x first, then y)
        assert result.kdims[0].name == 'x'
        assert result.kdims[0].unit is None  # Dummy coord has no unit
        assert result.kdims[1].name == 'y'
        assert result.kdims[1].unit == 's'
        assert result.vdims[0].unit == 'K'

        # Check data contains dummy x coordinates
        mesh_data = result.data
        np.testing.assert_array_equal(mesh_data['x'], [0, 1, 2])  # dummy x coords
        np.testing.assert_array_equal(mesh_data['y'], [4, 5])  # real y coords

    def test_with_missing_y_coord(self):
        x_coord = sc.array(dims=['x'], values=[1, 2, 3], unit='m')
        values = sc.array(
            dims=['y', 'x'], values=[[10, 20, 30], [40, 50, 60]], unit='K'
        )
        data = sc.DataArray(data=values, coords={'x': x_coord})

        result = scipp_to_holoviews.convert_quadmesh_2d(data)

        assert isinstance(result, hv.QuadMesh)
        mesh_data = result.data
        np.testing.assert_array_equal(mesh_data['x'], [1, 2, 3])  # real x coords
        np.testing.assert_array_equal(mesh_data['y'], [0, 1])  # dummy y coords

    def test_with_no_coords(self):
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]], unit='counts')
        data = sc.DataArray(data=values)

        result = scipp_to_holoviews.convert_quadmesh_2d(data)

        assert isinstance(result, hv.QuadMesh)
        mesh_data = result.data
        np.testing.assert_array_equal(mesh_data['x'], [0, 1, 2])  # dummy x coords
        np.testing.assert_array_equal(mesh_data['y'], [0, 1])  # dummy y coords


class TestConvertImage2d:
    def test_basic_conversion(self):
        x_coord = sc.array(dims=['x'], values=[0, 1, 2], unit='m')
        y_coord = sc.array(dims=['y'], values=[10, 20], unit='s')
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]], unit='counts')
        data = sc.DataArray(data=values, coords={'x': x_coord, 'y': y_coord})

        result = scipp_to_holoviews.convert_image_2d(data)

        assert isinstance(result, hv.Image)
        assert len(result.kdims) == 2
        assert len(result.vdims) == 1

        # Check dimension order (reversed)
        assert result.kdims[0].name == 'x'
        assert result.kdims[1].name == 'y'

    def test_with_bin_edges(self):
        # Test with bin edges
        x_edges = sc.array(dims=['x'], values=[0, 1, 2, 3], unit='m')
        y_edges = sc.array(dims=['y'], values=[0, 10, 20], unit='s')
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]], unit='counts')
        data = sc.DataArray(data=values, coords={'x': x_edges, 'y': y_edges})

        result = scipp_to_holoviews.convert_image_2d(data)

        assert isinstance(result, hv.Image)
        # Should use midpoints for coordinates
        image_data = result.data
        np.testing.assert_array_equal(image_data['x'], [0.5, 1.5, 2.5])  # x midpoints
        np.testing.assert_array_equal(image_data['y'], [5, 15])  # y midpoints

    def test_with_mixed_midpoints_and_bin_edges(self):
        # Test with bin edges only along y
        x_coord = sc.array(dims=['x'], values=[0, 1, 2], unit='m')
        y_edges = sc.array(dims=['y'], values=[0, 10, 20], unit='s')
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]], unit='counts')
        data = sc.DataArray(data=values, coords={'x': x_coord, 'y': y_edges})

        result = scipp_to_holoviews.convert_image_2d(data)

        assert isinstance(result, hv.Image)
        # Should use midpoints for coordinates
        image_data = result.data
        np.testing.assert_array_equal(image_data['x'], [0, 1, 2])  # x preserved
        np.testing.assert_array_equal(image_data['y'], [5, 15])  # y midpoints

    def test_with_missing_x_coord(self):
        y_coord = sc.array(dims=['y'], values=[10, 20], unit='s')
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]], unit='counts')
        data = sc.DataArray(data=values, coords={'y': y_coord})

        result = scipp_to_holoviews.convert_image_2d(data)

        assert isinstance(result, hv.Image)
        assert result.kdims[0].name == 'x'
        assert result.kdims[0].unit is None  # Dummy coord has no unit
        assert result.kdims[1].name == 'y'
        assert result.kdims[1].unit == 's'

        # Check data contains dummy x coordinates
        image_data = result.data
        np.testing.assert_array_equal(image_data['x'], [0, 1, 2])  # dummy x coords
        np.testing.assert_array_equal(image_data['y'], [10, 20])  # real y coords

    def test_with_missing_y_coord(self):
        x_coord = sc.array(dims=['x'], values=[0, 1, 2], unit='m')
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]], unit='counts')
        data = sc.DataArray(data=values, coords={'x': x_coord})

        result = scipp_to_holoviews.convert_image_2d(data)

        assert isinstance(result, hv.Image)
        image_data = result.data
        np.testing.assert_array_equal(image_data['x'], [0, 1, 2])  # real x coords
        np.testing.assert_array_equal(image_data['y'], [0, 1])  # dummy y coords

    def test_with_no_coords(self):
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]], unit='counts')
        data = sc.DataArray(data=values)

        result = scipp_to_holoviews.convert_image_2d(data)

        assert isinstance(result, hv.Image)
        image_data = result.data
        np.testing.assert_array_equal(image_data['x'], [0, 1, 2])  # dummy x coords
        np.testing.assert_array_equal(image_data['y'], [0, 1])  # dummy y coords


class TestToHoloviews:
    def test_1d_histogram(self):
        edges = sc.array(dims=['x'], values=[0, 1, 2, 3])
        values = sc.array(dims=['x'], values=[10, 20, 30])
        data = sc.DataArray(data=values, coords={'x': edges})

        result = scipp_to_holoviews.to_holoviews(data)

        assert isinstance(result, hv.Histogram)

    def test_1d_curve(self):
        coord = sc.array(dims=['x'], values=[0, 1, 2])
        values = sc.array(dims=['x'], values=[10, 20, 30])
        data = sc.DataArray(data=values, coords={'x': coord})

        result = scipp_to_holoviews.to_holoviews(data)

        assert isinstance(result, hv.Curve)

    def test_1d_error_bars(self):
        coord = sc.array(dims=['x'], values=[0, 1, 2])
        values = sc.array(dims=['x'], values=[10.0, 20.0, 30.0])
        variances = sc.array(dims=['x'], values=[1.0, 4.0, 9.0])
        data = sc.DataArray(
            data=sc.array(dims=['x'], values=values.values, variances=variances.values),
            coords={'x': coord},
        )

        result = scipp_to_holoviews.to_holoviews(data)

        assert isinstance(result, hv.ErrorBars)

    def test_1d_error_bars_missing_coord(self):
        values = sc.array(dims=['x'], values=[10.0, 20.0, 30.0])
        variances = sc.array(dims=['x'], values=[1.0, 4.0, 9.0])
        data = sc.DataArray(
            data=sc.array(dims=['x'], values=values.values, variances=variances.values)
        )

        result = scipp_to_holoviews.to_holoviews(data)

        assert isinstance(result, hv.ErrorBars)
        error_data = result.data
        np.testing.assert_array_equal(error_data['x'], [0, 1, 2])  # dummy coords
        np.testing.assert_array_equal(error_data['values'], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(
            error_data['yerr'], [1, 2, 3]
        )  # sqrt of variances

    def test_2d_image_evenly_spaced(self):
        x_coord = sc.linspace('x', 0, 2, 3)
        y_coord = sc.linspace('y', 0, 10, 2)
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]])
        data = sc.DataArray(data=values, coords={'x': x_coord, 'y': y_coord})

        result = scipp_to_holoviews.to_holoviews(data)

        assert isinstance(result, hv.Image)

    def test_2d_quadmesh_irregular(self):
        x_coord = sc.array(dims=['x'], values=[0, 1, 5])  # Non-evenly spaced
        y_coord = sc.array(dims=['y'], values=[0, 10])
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]])
        data = sc.DataArray(data=values, coords={'x': x_coord, 'y': y_coord})

        result = scipp_to_holoviews.to_holoviews(data)

        assert isinstance(result, hv.QuadMesh)

    def test_2d_with_bin_edges_defaults_to_image(self):
        x_edges = sc.array(dims=['x'], values=[0, 1, 2, 3], unit='m')
        y_edges = sc.array(dims=['y'], values=[0, 10, 20], unit='s')
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]], unit='counts')
        data = sc.DataArray(data=values, coords={'x': x_edges, 'y': y_edges})

        result = scipp_to_holoviews.to_holoviews(data)

        # Should default to Image even with bin edges
        assert isinstance(result, hv.Image)

    def test_2d_with_bin_edges_preserve_edges(self):
        x_edges = sc.array(dims=['x'], values=[0, 1, 2, 3], unit='m')
        y_edges = sc.array(dims=['y'], values=[0, 10, 20], unit='s')
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]], unit='counts')
        data = sc.DataArray(data=values, coords={'x': x_edges, 'y': y_edges})

        result = scipp_to_holoviews.to_holoviews(data, preserve_edges=True)

        # Should use QuadMesh when explicitly requested
        assert isinstance(result, hv.QuadMesh)

    def test_2d_no_bin_edges_ignores_preserve_edges(self):
        x_coord = sc.array(dims=['x'], values=[0, 1, 5])  # Non-evenly spaced, no edges
        y_coord = sc.array(dims=['y'], values=[0, 10])
        values = sc.array(dims=['y', 'x'], values=[[1, 2, 3], [4, 5, 6]])
        data = sc.DataArray(data=values, coords={'x': x_coord, 'y': y_coord})

        result = scipp_to_holoviews.to_holoviews(data, preserve_edges=True)

        # Should use QuadMesh due to irregular spacing, regardless of preserve_edges
        assert isinstance(result, hv.QuadMesh)

    def test_0d_raises_error(self):
        data = sc.DataArray(data=sc.scalar(42.0))

        with pytest.raises(
            ValueError, match="Input DataArray must have at least one dimension"
        ):
            scipp_to_holoviews.to_holoviews(data)

    def test_3d_raises_error(self):
        x_coord = sc.array(dims=['x'], values=[0, 1])
        y_coord = sc.array(dims=['y'], values=[0, 1])
        z_coord = sc.array(dims=['z'], values=[0, 1])
        values = sc.array(
            dims=['z', 'y', 'x'], values=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        )
        data = sc.DataArray(
            data=values, coords={'x': x_coord, 'y': y_coord, 'z': z_coord}
        )

        with pytest.raises(ValueError, match="Only 1D and 2D data are supported"):
            scipp_to_holoviews.to_holoviews(data)

    def test_1d_curve_missing_coord(self):
        values = sc.array(dims=['x'], values=[10, 20, 30])
        data = sc.DataArray(data=values)

        result = scipp_to_holoviews.to_holoviews(data)

        assert isinstance(result, hv.Curve)
        curve_data = result.data
        np.testing.assert_array_equal(curve_data['x'], [0, 1, 2])  # dummy coords
        np.testing.assert_array_equal(curve_data['values'], [10, 20, 30])

    def test_2d_single_row_returns_image(self):
        """hv.Image with 1xN data requires explicit bounds computation."""
        x_coord = sc.linspace('x', 0, 9, 10)
        y_coord = sc.linspace('y', 0, 0, 1)
        values = sc.arange('x', 10.0).fold(dim='x', sizes={'y': 1, 'x': 10})
        data = sc.DataArray(data=values, coords={'x': x_coord, 'y': y_coord})

        result = scipp_to_holoviews.to_holoviews(data)

        assert isinstance(result, hv.Image)
        np.testing.assert_array_equal(result.data['values'].flatten(), np.arange(10))

    def test_2d_single_column_returns_image(self):
        """hv.Image with Nx1 data requires explicit bounds computation."""
        x_coord = sc.linspace('x', 0, 0, 1)
        y_coord = sc.linspace('y', 0, 7, 8)
        values = sc.arange('y', 8.0).fold(dim='y', sizes={'y': 8, 'x': 1})
        data = sc.DataArray(data=values, coords={'x': x_coord, 'y': y_coord})

        result = scipp_to_holoviews.to_holoviews(data)

        assert isinstance(result, hv.Image)
        np.testing.assert_array_equal(result.data['values'].flatten(), np.arange(8))

    def test_2d_single_element_returns_image(self):
        """hv.Image with 1x1 data requires explicit bounds computation."""
        x_coord = sc.linspace('x', 0, 0, 1)
        y_coord = sc.linspace('y', 0, 0, 1)
        values = sc.array(dims=['y', 'x'], values=[[42.0]])
        data = sc.DataArray(data=values, coords={'x': x_coord, 'y': y_coord})

        result = scipp_to_holoviews.to_holoviews(data)

        assert isinstance(result, hv.Image)
        np.testing.assert_array_equal(result.data['values'], [[42.0]])

    def test_2d_single_row_no_coords_returns_image(self):
        """hv.Image with 1xN data works with dummy coords and bounds."""
        values = sc.arange('x', 10.0).fold(dim='x', sizes={'y': 1, 'x': 10})
        data = sc.DataArray(data=values)

        result = scipp_to_holoviews.to_holoviews(data)

        assert isinstance(result, hv.Image)
        np.testing.assert_array_equal(result.data['values'].flatten(), np.arange(10))

    def test_2d_single_row_image_renders_to_bokeh(self):
        """Verify that 1xN Image can be rendered to Bokeh."""
        from holoviews.plotting.bokeh import BokehRenderer

        values = sc.arange('x', 10.0).fold(dim='x', sizes={'y': 1, 'x': 10})
        data = sc.DataArray(data=values)

        result = scipp_to_holoviews.to_holoviews(data)
        assert isinstance(result, hv.Image)

        renderer = BokehRenderer.instance()
        bokeh_plot = renderer.get_plot(result)
        assert bokeh_plot is not None

    def test_2d_single_column_image_renders_to_bokeh(self):
        """Verify that Nx1 Image can be rendered to Bokeh."""
        from holoviews.plotting.bokeh import BokehRenderer

        values = sc.arange('y', 8.0).fold(dim='y', sizes={'y': 8, 'x': 1})
        data = sc.DataArray(data=values)

        result = scipp_to_holoviews.to_holoviews(data)
        assert isinstance(result, hv.Image)

        renderer = BokehRenderer.instance()
        bokeh_plot = renderer.get_plot(result)
        assert bokeh_plot is not None

    def test_2d_single_element_image_renders_to_bokeh(self):
        """Verify that 1x1 Image can be rendered to Bokeh."""
        from holoviews.plotting.bokeh import BokehRenderer

        values = sc.array(dims=['y', 'x'], values=[[42.0]])
        data = sc.DataArray(data=values)

        result = scipp_to_holoviews.to_holoviews(data)
        assert isinstance(result, hv.Image)

        renderer = BokehRenderer.instance()
        bokeh_plot = renderer.get_plot(result)
        assert bokeh_plot is not None


class TestAllCoordsEvenlySpaced:
    def test_returns_true_for_evenly_spaced_coords(self):
        data = sc.DataArray(
            sc.zeros(dims=['x', 'y'], shape=[3, 4]),
            coords={
                'x': sc.linspace('x', 0, 10, 3),
                'y': sc.linspace('y', 0, 20, 4),
            },
        )
        assert scipp_to_holoviews._all_coords_evenly_spaced(data) is True

    def test_returns_false_for_unevenly_spaced_coords(self):
        data = sc.DataArray(
            sc.zeros(dims=['x', 'y'], shape=[3, 4]),
            coords={
                'x': sc.array(dims=['x'], values=[0, 1, 5]),  # not evenly spaced
                'y': sc.linspace('y', 0, 20, 4),
            },
        )
        assert scipp_to_holoviews._all_coords_evenly_spaced(data) is False

    def test_returns_true_for_missing_coords(self):
        data = sc.DataArray(sc.zeros(dims=['x', 'y'], shape=[3, 4]))
        assert scipp_to_holoviews._all_coords_evenly_spaced(data) is True

    def test_returns_true_for_single_element_coord(self):
        data = sc.DataArray(
            sc.zeros(dims=['x', 'y'], shape=[1, 4]),
            coords={
                'x': sc.array(dims=['x'], values=[5.0]),
                'y': sc.linspace('y', 0, 20, 4),
            },
        )
        assert scipp_to_holoviews._all_coords_evenly_spaced(data) is True

    def test_returns_true_for_empty_coord(self):
        """Empty coordinates should be treated as trivially evenly spaced."""
        data = sc.DataArray(
            sc.zeros(dims=['x', 'y', 'z'], shape=[0, 0, 0]),
            coords={'z': sc.arange('z', 0, unit='ms')},
        )
        assert scipp_to_holoviews._all_coords_evenly_spaced(data) is True
