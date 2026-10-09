# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for DetectorViewScilineFactory."""

import numpy as np
import pytest
import scipp as sc
from ess.reduce.nexus.types import RawDetector, SampleRun

from ess.livedata.config.models import PixelWeighting
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.workflows.detector_view.data_source import DetectorNumberSource
from ess.livedata.workflows.detector_view.factory import DetectorViewFactory
from ess.livedata.workflows.detector_view.types import (
    GeometricViewConfig,
    LogicalViewConfig,
)
from ess.livedata.workflows.detector_view_specs import DetectorViewParams

from .utils import make_fake_detector_number, make_fake_nexus_detector_data


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


@pytest.mark.parametrize(
    ('reduction_dim', 'image_sizes', 'pixels_per_image_pixel'),
    [(None, {'y': 4, 'x': 4}, 1), ('y', {'x': 4}, 4)],
)
@pytest.mark.parametrize('enabled', [True, False])
def test_logical_view_image_with_pixel_weighting(
    reduction_dim: str | None,
    image_sizes: dict[str, int],
    pixels_per_image_pixel: int,
    enabled: bool,
) -> None:
    n_events_per_pixel = 10

    def transform(da: sc.DataArray, source_name: str) -> sc.DataArray:
        return da.fold(dim='detector_number', sizes={'y': 4, 'x': 4})

    factory = DetectorViewFactory(
        data_source=DetectorNumberSource(make_fake_detector_number(4, 4)),
        view_config=LogicalViewConfig(
            transform=transform, reduction_dim=reduction_dim, roi_support=False
        ),
    )
    params = DetectorViewParams(pixel_weighting=PixelWeighting(enabled=enabled))
    workflow = factory.make_workflow('detector', params=params)
    workflow.build()
    events = make_fake_nexus_detector_data(
        y_size=4, x_size=4, n_events_per_pixel=n_events_per_pixel
    )
    workflow.accumulate(
        {'detector': RawDetector[SampleRun](events)},
        start_time=Timestamp.from_ns(1000),
        end_time=Timestamp.from_ns(2000),
    )

    image = workflow.finalize()['cumulative']

    assert image.sizes == image_sizes
    expected = n_events_per_pixel * (1 if enabled else pixels_per_image_pixel)
    np.testing.assert_array_equal(image.values, expected)
