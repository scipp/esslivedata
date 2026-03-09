# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for GeometricProjector and make_geometric_projector."""

import numpy as np
import scipp as sc

from ess.livedata.handlers.detector_view.projectors import (
    GeometricProjector,
    make_geometric_projector,
)

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


def _make_positions(*, z_sign: float = 1.0) -> sc.Variable:
    """Create a 2x3 grid of detector positions at z_sign * 2.0 m."""
    x = np.array([-0.1, 0.0, 0.1, -0.1, 0.0, 0.1])
    y = np.array([-0.05, -0.05, -0.05, 0.05, 0.05, 0.05])
    z = np.full(6, z_sign * 2.0)
    positions = sc.vectors(
        dims=['replica', 'detector_number'],
        values=np.stack([x, y, z], axis=-1).reshape(1, 6, 3),
        unit='m',
    )
    return positions


class TestMakeGeometricProjectorFlipX:
    """Tests for the flip_x parameter in make_geometric_projector."""

    def test_flip_x_negates_x_coordinates(self):
        positions = _make_positions()
        resolution = {'x': 3, 'y': 2}

        proj_normal = make_geometric_projector(
            positions, 'xy_plane', resolution, flip_x=False
        )
        proj_flipped = make_geometric_projector(
            positions, 'xy_plane', resolution, flip_x=True
        )

        normal_x = sc.midpoints(proj_normal._edges['x'])
        flipped_x = sc.midpoints(proj_flipped._edges['x'])

        # Flipped edges should be the negation of normal edges (reversed order)
        assert sc.allclose(flipped_x, -sc.sort(normal_x, 'x', order='descending'))

    def test_flip_x_mirrors_projected_image(self):
        """Projecting events with flip_x mirrors the image along x."""
        positions = _make_positions()
        resolution = {'x': 3, 'y': 2}

        proj_normal = make_geometric_projector(
            positions, 'xy_plane', resolution, flip_x=False
        )
        proj_flipped = make_geometric_projector(
            positions, 'xy_plane', resolution, flip_x=True
        )

        data = make_fake_nexus_detector_data(y_size=2, x_size=3, n_events_per_pixel=50)

        normal_counts = proj_normal.project_events(data).bins.size()
        flipped_counts = proj_flipped.project_events(data).bins.size()

        # Flipped image should be mirrored along x
        n_x = normal_counts.sizes['x']
        for i in range(n_x):
            assert sc.identical(
                flipped_counts['x', i].data, normal_counts['x', n_x - 1 - i].data
            )

    def test_flip_x_false_is_default_behavior(self):
        positions = _make_positions()
        resolution = {'x': 3, 'y': 2}

        proj = make_geometric_projector(positions, 'xy_plane', resolution, flip_x=False)

        # x edges should span the original positive range
        x_edges = proj._edges['x']
        assert x_edges.min().value >= -0.15
        assert x_edges.max().value <= 0.15

    def test_flip_x_does_not_affect_y(self):
        positions = _make_positions()
        resolution = {'x': 3, 'y': 2}

        proj_normal = make_geometric_projector(
            positions, 'xy_plane', resolution, flip_x=False
        )
        proj_flipped = make_geometric_projector(
            positions, 'xy_plane', resolution, flip_x=True
        )

        assert sc.identical(proj_normal._edges['y'], proj_flipped._edges['y'])
