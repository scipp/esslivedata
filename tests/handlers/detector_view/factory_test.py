# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for DetectorViewScilineFactory."""

import scipp as sc

from ess.livedata.handlers.detector_view.data_source import DetectorNumberSource
from ess.livedata.handlers.detector_view.factory import DetectorViewFactory
from ess.livedata.handlers.detector_view.types import (
    GeometricViewConfig,
    LogicalViewConfig,
)

from .utils import make_fake_detector_number


class TestDetectorViewScilineFactory:
    """Tests for DetectorViewScilineFactory."""

    def test_factory_initialization_with_logical_and_geometric_configs(self):
        """Test factory initialization with logical and geometric view configs."""
        detector_number = make_fake_detector_number(4, 4)

        # Logical config
        def transform(da: sc.DataArray, source_name: str) -> sc.DataArray:
            return da.fold(dim='detector_number', sizes={'y': 4, 'x': 4})

        logical_factory = DetectorViewFactory(
            data_source=DetectorNumberSource(detector_number),
            view_config=LogicalViewConfig(transform=transform),
        )
        assert logical_factory is not None
        assert isinstance(logical_factory._view_config, LogicalViewConfig)
        assert logical_factory._view_config.transform is not None

        # Geometric config
        geometric_factory = DetectorViewFactory(
            data_source=DetectorNumberSource(detector_number),
            view_config=GeometricViewConfig(
                projection_type='xy_plane',
                resolution={'x': 100, 'y': 100},
            ),
        )
        assert geometric_factory is not None
        assert isinstance(geometric_factory._view_config, GeometricViewConfig)
        assert geometric_factory._view_config.projection_type == 'xy_plane'

    def test_factory_initialization_with_per_source_configs(self):
        """Test that factory can be initialized with per-source configs."""
        detector_number = make_fake_detector_number(4, 4)

        def transform(da: sc.DataArray, source_name: str) -> sc.DataArray:
            return da.fold(dim='detector_number', sizes={'y': 4, 'x': 4})

        factory = DetectorViewFactory(
            data_source=DetectorNumberSource(detector_number),
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
