# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from ess.livedata.config.roi_names import ROIGeometry, ROIStreamMapper, get_roi_mapper


class TestROIGeometry:
    def test_readback_key_for_rectangle(self):
        geom = ROIGeometry(geometry_type="rectangle", num_rois=4, index_offset=0)
        assert geom.readback_key == "roi_rectangle"

    def test_readback_key_for_polygon(self):
        geom = ROIGeometry(geometry_type="polygon", num_rois=3, index_offset=4)
        assert geom.readback_key == "roi_polygon"

    def test_index_range_without_offset(self):
        geom = ROIGeometry(geometry_type="rectangle", num_rois=4, index_offset=0)
        assert list(geom.index_range) == [0, 1, 2, 3]

    def test_index_range_with_offset(self):
        geom = ROIGeometry(geometry_type="polygon", num_rois=3, index_offset=4)
        assert list(geom.index_range) == [4, 5, 6]

    def test_default_offset_is_zero(self):
        geom = ROIGeometry(geometry_type="rectangle", num_rois=2)
        assert geom.index_offset == 0


class TestROIStreamMapper:
    def test_single_geometry_total_rois(self):
        mapper = ROIStreamMapper([ROIGeometry("rectangle", num_rois=4)])
        assert mapper.total_rois == 4

    def test_multi_geometry_total_rois(self):
        mapper = ROIStreamMapper(
            [
                ROIGeometry("rectangle", num_rois=4, index_offset=0),
                ROIGeometry("polygon", num_rois=3, index_offset=4),
            ]
        )
        assert mapper.total_rois == 7

    def test_readback_keys_single_geometry(self):
        mapper = ROIStreamMapper([ROIGeometry("rectangle", num_rois=4)])
        assert mapper.readback_keys == ["roi_rectangle"]

    def test_readback_keys_multi_geometry(self):
        mapper = ROIStreamMapper(
            [
                ROIGeometry("rectangle", num_rois=4, index_offset=0),
                ROIGeometry("polygon", num_rois=3, index_offset=4),
            ]
        )
        assert mapper.readback_keys == ["roi_rectangle", "roi_polygon"]

    def test_current_key_generation(self):
        mapper = ROIStreamMapper([ROIGeometry("rectangle", num_rois=4)])
        assert mapper.current_key(0) == "roi_current_0"
        assert mapper.current_key(2) == "roi_current_2"

    def test_cumulative_key_generation(self):
        mapper = ROIStreamMapper([ROIGeometry("rectangle", num_rois=4)])
        assert mapper.cumulative_key(0) == "roi_cumulative_0"
        assert mapper.cumulative_key(3) == "roi_cumulative_3"

    def test_all_current_keys(self):
        mapper = ROIStreamMapper([ROIGeometry("rectangle", num_rois=3)])
        assert mapper.all_current_keys() == [
            "roi_current_0",
            "roi_current_1",
            "roi_current_2",
        ]

    def test_all_cumulative_keys(self):
        mapper = ROIStreamMapper([ROIGeometry("rectangle", num_rois=3)])
        assert mapper.all_cumulative_keys() == [
            "roi_cumulative_0",
            "roi_cumulative_1",
            "roi_cumulative_2",
        ]

    def test_all_histogram_keys(self):
        mapper = ROIStreamMapper([ROIGeometry("rectangle", num_rois=2)])
        assert mapper.all_histogram_keys() == [
            "roi_current_0",
            "roi_current_1",
            "roi_cumulative_0",
            "roi_cumulative_1",
        ]

    def test_all_keys_span_all_geometries(self):
        mapper = ROIStreamMapper(
            [
                ROIGeometry("rectangle", num_rois=2, index_offset=0),
                ROIGeometry("polygon", num_rois=2, index_offset=2),
            ]
        )
        assert mapper.all_current_keys() == [
            "roi_current_0",
            "roi_current_1",
            "roi_current_2",
            "roi_current_3",
        ]

    def test_parse_roi_index_from_current_key(self):
        mapper = ROIStreamMapper([ROIGeometry("rectangle", num_rois=4)])
        assert mapper.parse_roi_index("roi_current_0") == 0
        assert mapper.parse_roi_index("roi_current_3") == 3

    def test_parse_roi_index_from_cumulative_key(self):
        mapper = ROIStreamMapper([ROIGeometry("rectangle", num_rois=4)])
        assert mapper.parse_roi_index("roi_cumulative_0") == 0
        assert mapper.parse_roi_index("roi_cumulative_2") == 2

    def test_parse_roi_index_returns_none_for_invalid_key(self):
        mapper = ROIStreamMapper([ROIGeometry("rectangle", num_rois=4)])
        assert mapper.parse_roi_index("roi_rectangle") is None
        assert mapper.parse_roi_index("something_else") is None
        assert mapper.parse_roi_index("roi_other_0") is None

    def test_geometry_for_index_first_geometry(self):
        geom1 = ROIGeometry("rectangle", num_rois=4, index_offset=0)
        geom2 = ROIGeometry("polygon", num_rois=3, index_offset=4)
        mapper = ROIStreamMapper([geom1, geom2])

        assert mapper.geometry_for_index(0) == geom1
        assert mapper.geometry_for_index(2) == geom1

    def test_geometry_for_index_second_geometry(self):
        geom1 = ROIGeometry("rectangle", num_rois=4, index_offset=0)
        geom2 = ROIGeometry("polygon", num_rois=3, index_offset=4)
        mapper = ROIStreamMapper([geom1, geom2])

        assert mapper.geometry_for_index(4) == geom2
        assert mapper.geometry_for_index(6) == geom2

    def test_geometry_for_index_out_of_range(self):
        mapper = ROIStreamMapper([ROIGeometry("rectangle", num_rois=4, index_offset=0)])
        assert mapper.geometry_for_index(10) is None
        assert mapper.geometry_for_index(-1) is None


class TestGetROIMapper:
    def test_returns_mapper_with_default_configuration(self):
        mapper = get_roi_mapper()
        assert isinstance(mapper, ROIStreamMapper)
        assert mapper.total_rois > 0

    def test_default_uses_rectangle_geometry(self):
        mapper = get_roi_mapper()
        assert "roi_rectangle" in mapper.readback_keys

    def test_default_uses_polygon_geometry(self):
        mapper = get_roi_mapper()
        assert "roi_polygon" in mapper.readback_keys

    def test_default_has_eight_rois(self):
        # 4 rectangle ROIs (indices 0-3) + 4 polygon ROIs (indices 4-7)
        mapper = get_roi_mapper()
        assert mapper.total_rois == 8

    def test_accepts_instrument_parameter(self):
        # For future extensibility
        mapper = get_roi_mapper(instrument="dummy")
        assert isinstance(mapper, ROIStreamMapper)
