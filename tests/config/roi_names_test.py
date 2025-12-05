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
