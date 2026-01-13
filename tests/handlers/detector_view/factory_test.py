# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for DetectorViewScilineFactory."""

import scipp as sc

from ess.livedata.handlers.detector_view import (
    DetectorNumberSource,
    DetectorViewScilineFactory,
    GeometricViewConfig,
    LogicalViewConfig,
)

from .utils import make_fake_detector_number


class TestDetectorViewScilineFactory:
    """Tests for DetectorViewScilineFactory."""

    def test_factory_initialization_with_logical_config(self):
        """Test that factory can be initialized with logical view config."""
        detector_number = make_fake_detector_number(4, 4)

        def transform(da: sc.DataArray, source_name: str) -> sc.DataArray:
            return da.fold(dim='detector_number', sizes={'y': 4, 'x': 4})

        factory = DetectorViewScilineFactory(
            data_source=DetectorNumberSource(detector_number),
            tof_bins=sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns'),
            view_config=LogicalViewConfig(transform=transform),
        )

        # Just verify the factory was created - don't try to instantiate the full
        # workflow as that requires all Sciline dependencies to be satisfied
        assert factory is not None
        assert isinstance(factory._view_config, LogicalViewConfig)
        assert factory._view_config.transform is not None

    def test_factory_initialization_with_geometric_config(self):
        """Test that factory can be initialized with geometric view config."""
        detector_number = make_fake_detector_number(4, 4)
        factory = DetectorViewScilineFactory(
            data_source=DetectorNumberSource(detector_number),
            tof_bins=sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns'),
            view_config=GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'x': 100, 'y': 100},
            ),
        )

        assert factory is not None
        assert isinstance(factory._view_config, GeometricViewConfig)
        assert factory._view_config.projection_type == 'xy_plane'

    def test_factory_initialization_with_per_source_configs(self):
        """Test that factory can be initialized with per-source configs."""
        detector_number = make_fake_detector_number(4, 4)

        def transform(da: sc.DataArray, source_name: str) -> sc.DataArray:
            return da.fold(dim='detector_number', sizes={'y': 4, 'x': 4})

        factory = DetectorViewScilineFactory(
            data_source=DetectorNumberSource(detector_number),
            tof_bins=sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns'),
            view_config={
                'source_a': LogicalViewConfig(transform=transform),
                'source_b': GeometricViewConfig(
                    projection_type='xy_plane',
                    resolution={'x': 100, 'y': 100},
                ),
            },
        )

        assert factory is not None
        assert isinstance(factory._get_config('source_a'), LogicalViewConfig)
        assert isinstance(factory._get_config('source_b'), GeometricViewConfig)
