# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for DetectorViewScilineFactory."""

import scipp as sc

from ess.livedata.handlers.detector_view_sciline import (
    DetectorNumberSource,
    DetectorViewScilineFactory,
)

from .utils import make_fake_detector_number


class TestDetectorViewScilineFactory:
    """Tests for DetectorViewScilineFactory."""

    def test_factory_initialization_with_logical_transform(self):
        """Test that factory can be initialized with logical transform."""
        detector_number = make_fake_detector_number(4, 4)

        def transform(da: sc.DataArray, source_name: str) -> sc.DataArray:
            return da.fold(dim='detector_number', sizes={'y': 4, 'x': 4})

        factory = DetectorViewScilineFactory(
            data_source=DetectorNumberSource(detector_number),
            tof_bins=sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns'),
            logical_transform=transform,
        )

        # Just verify the factory was created - don't try to instantiate the full
        # workflow as that requires all Sciline dependencies to be satisfied
        assert factory is not None
        assert factory._logical_transform is not None

    def test_factory_initialization_without_transform(self):
        """Test that factory can be initialized without transform."""
        detector_number = make_fake_detector_number(4, 4)
        factory = DetectorViewScilineFactory(
            data_source=DetectorNumberSource(detector_number),
            tof_bins=sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns'),
            logical_transform=None,
        )

        assert factory is not None
        assert factory._logical_transform is None
