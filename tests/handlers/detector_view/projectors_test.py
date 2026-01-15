# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for GeometricProjector class."""

import numpy as np
import scipp as sc

from ess.livedata.handlers.detector_view.projectors import GeometricProjector

from .utils import make_fake_nexus_detector_data


class TestGeometricProjector:
    """Tests for GeometricProjector class."""

    @staticmethod
    def make_screen_coords_and_edges(n_pixels: int, screen_shape: tuple[int, int]):
        """Create test coordinates and edges for projection."""
        n_replicas = 2
        det_side = int(np.sqrt(n_pixels))
        scale_x = screen_shape[0] / det_side
        scale_y = screen_shape[1] / det_side

        pixel_y = np.arange(n_pixels) // det_side
        pixel_x = np.arange(n_pixels) % det_side

        rng = np.random.default_rng(42)
        coords_x = []
        coords_y = []
        for _ in range(n_replicas):
            noise = rng.normal(0, 0.1, n_pixels)
            coords_x.append(pixel_x * scale_x + noise)
            coords_y.append(pixel_y * scale_y + noise)

        coords = sc.DataGroup(
            screen_x=sc.array(
                dims=['replica', 'detector_number'], values=np.array(coords_x), unit='m'
            ),
            screen_y=sc.array(
                dims=['replica', 'detector_number'], values=np.array(coords_y), unit='m'
            ),
        )
        edges = sc.DataGroup(
            screen_x=sc.linspace(
                'screen_x', 0, screen_shape[0], screen_shape[0] + 1, unit='m'
            ),
            screen_y=sc.linspace(
                'screen_y', 0, screen_shape[1], screen_shape[1] + 1, unit='m'
            ),
        )
        return coords, edges

    def test_project_events_preserves_total_counts(self):
        """Test that event projection preserves total event count."""
        n_pixels = 16
        screen_shape = (4, 4)
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        coords, edges = self.make_screen_coords_and_edges(n_pixels, screen_shape)

        # Make edges wider to ensure all events are captured despite noise
        edges = sc.DataGroup(
            screen_x=sc.linspace('screen_x', -1, screen_shape[0] + 1, 10, unit='m'),
            screen_y=sc.linspace('screen_y', -1, screen_shape[1] + 1, 10, unit='m'),
        )

        projector = GeometricProjector(coords, edges)
        result = projector.project_events(data)

        # Total events should be preserved when edges are wide enough
        original_events = data.bins.size().sum().value
        projected_events = result.bins.size().sum().value
        assert projected_events == original_events

    def test_project_events_returns_binned_data(self):
        """Test that result is binned data with screen coordinate dims."""
        n_pixels = 16
        screen_shape = (4, 4)
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        coords, edges = self.make_screen_coords_and_edges(n_pixels, screen_shape)

        projector = GeometricProjector(coords, edges)
        result = projector.project_events(data)

        assert result.bins is not None
        assert 'screen_x' in result.dims
        assert 'screen_y' in result.dims
        assert result.sizes['screen_x'] == screen_shape[0]
        assert result.sizes['screen_y'] == screen_shape[1]

    def test_project_events_preserves_event_time_offset(self):
        """Test that event_time_offset is preserved in projected events."""
        n_pixels = 16
        screen_shape = (4, 4)
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        coords, edges = self.make_screen_coords_and_edges(n_pixels, screen_shape)

        projector = GeometricProjector(coords, edges)
        result = projector.project_events(data)

        # Check that events have event_time_offset coordinate
        event_data = result.bins.constituents['data']
        assert 'event_time_offset' in event_data.coords

    def test_project_events_cycles_replicas(self):
        """Test that projector cycles through replicas."""
        n_pixels = 16
        screen_shape = (4, 4)
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        coords, edges = self.make_screen_coords_and_edges(n_pixels, screen_shape)

        projector = GeometricProjector(coords, edges)

        # Project twice with same data
        result1 = projector.project_events(data)
        result2 = projector.project_events(data)

        # Results should differ (different replicas with different noise)
        assert not sc.identical(result1.bins.size(), result2.bins.size())
