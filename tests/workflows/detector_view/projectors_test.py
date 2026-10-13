# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for GeometricProjector and make_geometric_projector."""

import numpy as np
import pytest
import scipp as sc

from ess.livedata.workflows.detector_view.projectors import (
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


def _make_events(counts: list[int]) -> sc.DataArray:
    """Events binned by detector pixel, with the given number of events per pixel."""
    n_events = sum(counts)
    table = sc.DataArray(
        sc.ones(dims=['event'], shape=[n_events], unit='counts'),
        coords={
            'event_time_offset': sc.zeros(dims=['event'], shape=[n_events], unit='ns')
        },
    )
    begin = sc.array(
        dims=['detector_number'], values=np.cumsum([0, *counts[:-1]]), unit=None
    )
    return sc.DataArray(sc.bins(begin=begin, dim='event', data=table))


class TestGeometricProjectorReplicas:
    """Each event uses one position-noise replica of its detector pixel.

    The image is one row of four 1 m wide image pixels. Four replicas place the
    detector pixels as follows (x in m, y = 0.5 m):

    =====  ==================  ============
    pixel  replica 0, 1, 2, 3  image pixels
    =====  ==================  ============
    0      0.5, 1.5, 2.5, 3.5  0, 1, 2, 3
    1      3.5, 3.5, 3.5, 3.5  3, 3, 3, 3
    =====  ==================  ============
    """

    @pytest.fixture
    def projector(self) -> GeometricProjector:
        x = sc.array(
            dims=['replica', 'detector_number'],
            values=[[0.5, 3.5], [1.5, 3.5], [2.5, 3.5], [3.5, 3.5]],
            unit='m',
        )
        coords = sc.DataGroup(y=sc.full_like(x, 0.5), x=x)
        edges = sc.DataGroup(
            y=sc.linspace('y', 0.0, 1.0, 2, unit='m'),
            x=sc.linspace('x', 0.0, 4.0, 5, unit='m'),
        )
        return GeometricProjector(coords, edges)

    @staticmethod
    def image(projector: GeometricProjector, counts: list[int]) -> sc.DataArray:
        return projector.project_events(_make_events(counts)).bins.sum()['y', 0]

    def test_single_batch_divided_by_weights_gives_counts_per_detector_pixel(
        self, projector
    ):
        weights = projector.compute_weights()['y', 0]
        assert weights.values.tolist() == [0.25, 0.25, 0.25, 1.25]

        image = self.image(projector, [4, 4]) / weights

        assert image.values.tolist() == [4.0, 4.0, 4.0, 4.0]

    def test_events_of_a_pixel_are_spread_evenly_over_its_replicas(self, projector):
        image = self.image(projector, [6, 0])

        assert sorted(image.values.tolist()) == [1.0, 1.0, 2.0, 2.0]

    def test_first_replica_rotates_from_batch_to_batch(self, projector):
        images = [self.image(projector, [1, 0]).values for _ in range(4)]

        # One event per batch visits every replica of its pixel in four batches
        assert np.sum(images, axis=0).tolist() == [1.0, 1.0, 1.0, 1.0]


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

    def test_flip_x_does_not_affect_y(self):
        positions = _make_positions()
        resolution = {'x': 3, 'y': 2}

        proj_normal = make_geometric_projector(
            positions, 'xy_plane', resolution, flip_x=False
        )
        proj_flipped = make_geometric_projector(
            positions, 'xy_plane', resolution, flip_x=True
        )

        assert sc.identical(
            proj_normal.screen_metadata.coords['y'],
            proj_flipped.screen_metadata.coords['y'],
        )
