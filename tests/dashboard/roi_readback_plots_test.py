# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for ROI readback plotters."""

import uuid

import holoviews as hv
import pytest
import scipp as sc

from ess.livedata.config.models import Interval, PolygonROI, RectangleROI
from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.plotter_registry import DataRequirements, plotter_registry
from ess.livedata.dashboard.roi_readback_plots import (
    PolygonsReadbackParams,
    PolygonsReadbackPlotter,
    RectanglesReadbackParams,
    RectanglesReadbackPlotter,
)

hv.extension('bokeh')


@pytest.fixture
def result_key() -> ResultKey:
    workflow_id = WorkflowId(
        instrument='test_instrument',
        namespace='test_namespace',
        name='test_workflow',
        version=1,
    )
    job_id = JobId(source_name='test_source', job_number=uuid.uuid4())
    return ResultKey(
        workflow_id=workflow_id,
        job_id=job_id,
        output_name='roi_rectangle',
    )


@pytest.fixture
def rectangle_rois() -> dict[int, RectangleROI]:
    """Create sample rectangle ROIs for testing."""
    return {
        0: RectangleROI(
            x=Interval(min=0.0, max=10.0, unit='m'),
            y=Interval(min=0.0, max=5.0, unit='m'),
        ),
        2: RectangleROI(
            x=Interval(min=20.0, max=30.0, unit='m'),
            y=Interval(min=10.0, max=15.0, unit='m'),
        ),
    }


@pytest.fixture
def polygon_rois() -> dict[int, PolygonROI]:
    """Create sample polygon ROIs for testing."""
    return {
        4: PolygonROI(
            x=[0.0, 10.0, 5.0],
            y=[0.0, 0.0, 10.0],
            x_unit='m',
            y_unit='m',
        ),
        6: PolygonROI(
            x=[20.0, 30.0, 30.0, 20.0],
            y=[20.0, 20.0, 30.0, 30.0],
            x_unit='m',
            y_unit='m',
        ),
    }


@pytest.fixture
def rectangle_data_array(rectangle_rois: dict[int, RectangleROI]) -> sc.DataArray:
    """Create ROI readback DataArray for rectangles."""
    return RectangleROI.to_concatenated_data_array(rectangle_rois)


@pytest.fixture
def polygon_data_array(polygon_rois: dict[int, PolygonROI]) -> sc.DataArray:
    """Create ROI readback DataArray for polygons."""
    return PolygonROI.to_concatenated_data_array(polygon_rois)


class TestRectanglesReadbackPlotter:
    def test_plot_creates_rectangles_element(
        self, rectangle_data_array: sc.DataArray, result_key: ResultKey
    ):
        params = RectanglesReadbackParams()
        plotter = RectanglesReadbackPlotter.from_params(params)

        result = plotter.plot(rectangle_data_array, result_key)

        assert isinstance(result, hv.Rectangles)

    def test_plot_contains_correct_number_of_rectangles(
        self, rectangle_data_array: sc.DataArray, result_key: ResultKey
    ):
        params = RectanglesReadbackParams()
        plotter = RectanglesReadbackPlotter.from_params(params)

        result = plotter.plot(rectangle_data_array, result_key)

        # Should have 2 rectangles (indices 0 and 2)
        assert len(result.data) == 2

    def test_plot_assigns_colors_by_index(
        self, rectangle_data_array: sc.DataArray, result_key: ResultKey
    ):
        params = RectanglesReadbackParams()
        plotter = RectanglesReadbackPlotter.from_params(params)

        result = plotter.plot(rectangle_data_array, result_key)

        # Check that colors are assigned by index
        colors = hv.Cycle.default_cycles["default_colors"]
        rect_data = result.data.values.tolist()
        # First rectangle (index 0) should have color[0]
        assert rect_data[0][4] == colors[0]
        # Second rectangle (index 2) should have color[2]
        assert rect_data[1][4] == colors[2]

    def test_plot_with_empty_data_returns_empty_rectangles(self, result_key: ResultKey):
        params = RectanglesReadbackParams()
        plotter = RectanglesReadbackPlotter.from_params(params)
        empty_data = RectangleROI.to_concatenated_data_array({})

        result = plotter.plot(empty_data, result_key)

        assert isinstance(result, hv.Rectangles)
        assert len(result.data) == 0

    def test_plot_rectangle_coordinates_are_correct(
        self,
        rectangle_data_array: sc.DataArray,
        rectangle_rois: dict[int, RectangleROI],
        result_key: ResultKey,
    ):
        params = RectanglesReadbackParams()
        plotter = RectanglesReadbackPlotter.from_params(params)

        result = plotter.plot(rectangle_data_array, result_key)

        # Check first rectangle (index 0)
        roi_0 = rectangle_rois[0]
        rect_data = result.data.values.tolist()
        rect_0 = rect_data[0]
        assert rect_0[0] == roi_0.x.min
        assert rect_0[1] == roi_0.y.min
        assert rect_0[2] == roi_0.x.max
        assert rect_0[3] == roi_0.y.max

    def test_plot_applies_style_params(
        self, rectangle_data_array: sc.DataArray, result_key: ResultKey
    ):
        from ess.livedata.dashboard.roi_readback_plots import ROIReadbackStyle

        params = RectanglesReadbackParams(
            style=ROIReadbackStyle(fill_alpha=0.5, line_width=3.0)
        )
        plotter = RectanglesReadbackPlotter.from_params(params)

        result = plotter.plot(rectangle_data_array, result_key)

        # Style is applied via opts, which we can check on the element
        opts = result.opts.get().kwargs
        assert opts.get('fill_alpha') == 0.5
        assert opts.get('line_width') == 3.0

    def test_renders_to_bokeh(
        self, rectangle_data_array: sc.DataArray, result_key: ResultKey
    ):
        params = RectanglesReadbackParams()
        plotter = RectanglesReadbackPlotter.from_params(params)

        result = plotter.plot(rectangle_data_array, result_key)

        # Should render without error
        hv.render(result, backend='bokeh')


class TestPolygonsReadbackPlotter:
    def test_plot_creates_polygons_element(
        self, polygon_data_array: sc.DataArray, result_key: ResultKey
    ):
        params = PolygonsReadbackParams()
        plotter = PolygonsReadbackPlotter.from_params(params)

        result = plotter.plot(polygon_data_array, result_key)

        assert isinstance(result, hv.Polygons)

    def test_plot_contains_correct_number_of_polygons(
        self, polygon_data_array: sc.DataArray, result_key: ResultKey
    ):
        params = PolygonsReadbackParams()
        plotter = PolygonsReadbackPlotter.from_params(params)

        result = plotter.plot(polygon_data_array, result_key)

        # Should have 2 polygons (indices 4 and 6)
        assert len(result.data) == 2

    def test_plot_assigns_colors_by_index(
        self, polygon_data_array: sc.DataArray, result_key: ResultKey
    ):
        params = PolygonsReadbackParams()
        plotter = PolygonsReadbackPlotter.from_params(params)

        result = plotter.plot(polygon_data_array, result_key)

        # Check that colors are assigned by index
        colors = hv.Cycle.default_cycles["default_colors"]
        poly_data = result.data
        # First polygon (index 4) should have color[4]
        assert poly_data[0]['color'] == colors[4]
        # Second polygon (index 6) should have color[6]
        assert poly_data[1]['color'] == colors[6]

    def test_plot_with_empty_data_returns_empty_polygons(self, result_key: ResultKey):
        params = PolygonsReadbackParams()
        plotter = PolygonsReadbackPlotter.from_params(params)
        empty_data = PolygonROI.to_concatenated_data_array({})

        result = plotter.plot(empty_data, result_key)

        assert isinstance(result, hv.Polygons)
        assert len(result.data) == 0

    def test_plot_polygon_coordinates_are_correct(
        self,
        polygon_data_array: sc.DataArray,
        polygon_rois: dict[int, PolygonROI],
        result_key: ResultKey,
    ):
        params = PolygonsReadbackParams()
        plotter = PolygonsReadbackPlotter.from_params(params)

        result = plotter.plot(polygon_data_array, result_key)

        # Check first polygon (index 4)
        roi_4 = polygon_rois[4]
        poly_0 = result.data[0]
        # HoloViews stores coordinates as numpy arrays
        assert list(poly_0['x']) == [float(v) for v in roi_4.x]
        assert list(poly_0['y']) == [float(v) for v in roi_4.y]

    def test_renders_to_bokeh(
        self, polygon_data_array: sc.DataArray, result_key: ResultKey
    ):
        params = PolygonsReadbackParams()
        plotter = PolygonsReadbackPlotter.from_params(params)

        result = plotter.plot(polygon_data_array, result_key)

        # Should render without error
        hv.render(result, backend='bokeh')


class TestDataRequirementsDenyCoords:
    def test_deny_coords_rejects_data_with_coord(self):
        requirements = DataRequirements(
            min_dims=1,
            max_dims=1,
            deny_coords=['roi_index'],
        )
        # Create 1D data with roi_index coordinate
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
            coords={'roi_index': sc.array(dims=['x'], values=[0, 0, 1], dtype='int32')},
        )

        assert not requirements.validate_data({'key': data})

    def test_deny_coords_accepts_data_without_coord(self):
        requirements = DataRequirements(
            min_dims=1,
            max_dims=1,
            deny_coords=['roi_index'],
        )
        # Create 1D data without roi_index coordinate
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
        )

        assert requirements.validate_data({'key': data})

    def test_deny_coords_with_multiple_coords(self):
        requirements = DataRequirements(
            min_dims=1,
            max_dims=1,
            deny_coords=['roi_index', 'forbidden'],
        )
        # Data with 'forbidden' coord should be rejected
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0]),
            coords={'forbidden': sc.array(dims=['x'], values=[1.0, 2.0])},
        )

        assert not requirements.validate_data({'key': data})


class TestDataRequirementsRequiredDimNames:
    def test_required_dim_names_accepts_matching_dim(self):
        requirements = DataRequirements(
            min_dims=1,
            max_dims=1,
            required_dim_names=['bounds'],
        )
        data = sc.DataArray(
            sc.array(dims=['bounds'], values=[1.0, 2.0]),
        )

        assert requirements.validate_data({'key': data})

    def test_required_dim_names_rejects_different_dim(self):
        requirements = DataRequirements(
            min_dims=1,
            max_dims=1,
            required_dim_names=['bounds'],
        )
        data = sc.DataArray(
            sc.array(dims=['vertex'], values=[1.0, 2.0]),
        )

        assert not requirements.validate_data({'key': data})

    def test_required_dim_names_with_multiple_dims(self):
        requirements = DataRequirements(
            min_dims=2,
            max_dims=2,
            required_dim_names=['x', 'y'],
        )
        data = sc.DataArray(
            sc.array(dims=['x', 'y'], values=[[1.0, 2.0], [3.0, 4.0]]),
        )

        assert requirements.validate_data({'key': data})


class TestPlotterRegistration:
    def test_rectangles_readback_registered(self):
        assert 'rectangles_readback' in plotter_registry

    def test_polygons_readback_registered(self):
        assert 'polygons_readback' in plotter_registry

    def test_rectangles_readback_has_correct_requirements(self):
        spec = plotter_registry.get_spec('rectangles_readback')
        reqs = spec.data_requirements

        assert reqs.min_dims == 1
        assert reqs.max_dims == 1
        assert 'roi_index' in reqs.required_coords
        assert 'x' in reqs.required_coords
        assert 'y' in reqs.required_coords
        assert 'bounds' in reqs.required_dim_names

    def test_polygons_readback_has_correct_requirements(self):
        spec = plotter_registry.get_spec('polygons_readback')
        reqs = spec.data_requirements

        assert reqs.min_dims == 1
        assert reqs.max_dims == 1
        assert 'roi_index' in reqs.required_coords
        assert 'x' in reqs.required_coords
        assert 'y' in reqs.required_coords
        assert 'vertex' in reqs.required_dim_names

    def test_lines_plotter_denies_roi_index(self):
        spec = plotter_registry.get_spec('lines')
        reqs = spec.data_requirements

        assert 'roi_index' in reqs.deny_coords


class TestPlotterCompatibility:
    def test_rectangles_readback_compatible_with_rectangle_data(
        self, rectangle_data_array: sc.DataArray
    ):
        compatible = plotter_registry.get_compatible_plotters(
            {'key': rectangle_data_array}
        )
        assert 'rectangles_readback' in compatible

    def test_rectangles_readback_not_compatible_with_polygon_data(
        self, polygon_data_array: sc.DataArray
    ):
        compatible = plotter_registry.get_compatible_plotters(
            {'key': polygon_data_array}
        )
        assert 'rectangles_readback' not in compatible

    def test_polygons_readback_compatible_with_polygon_data(
        self, polygon_data_array: sc.DataArray
    ):
        compatible = plotter_registry.get_compatible_plotters(
            {'key': polygon_data_array}
        )
        assert 'polygons_readback' in compatible

    def test_polygons_readback_not_compatible_with_rectangle_data(
        self, rectangle_data_array: sc.DataArray
    ):
        compatible = plotter_registry.get_compatible_plotters(
            {'key': rectangle_data_array}
        )
        assert 'polygons_readback' not in compatible

    def test_lines_plotter_not_compatible_with_roi_data(
        self, rectangle_data_array: sc.DataArray
    ):
        compatible = plotter_registry.get_compatible_plotters(
            {'key': rectangle_data_array}
        )
        assert 'lines' not in compatible

    def test_lines_plotter_compatible_with_regular_1d_data(self):
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
        )
        compatible = plotter_registry.get_compatible_plotters({'key': data})
        assert 'lines' in compatible
