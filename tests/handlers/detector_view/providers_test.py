# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for Sciline provider functions."""

import pytest
import scipp as sc

from ess.livedata.handlers.detector_view.projectors import make_logical_projector
from ess.livedata.handlers.detector_view.providers import (
    accumulated_histogram,
    compute_detector_histogram,
    counts_in_range,
    counts_total,
    detector_image,
)
from ess.livedata.handlers.detector_view.types import (
    AccumulatedHistogram,
    Cumulative,
    Current,
    DetectorHistogram,
    PixelWeights,
    UsePixelWeighting,
)
from ess.livedata.handlers.detector_view.workflow import (
    NoCopyWindowAccumulator,
    add_logical_projection,
    create_base_workflow,
)

from .utils import (
    make_fake_empty_detector,
    make_fake_nexus_detector_data,
    make_logical_transform,
)


class TestNoCopyWindowAccumulator:
    """Tests for NoCopyWindowAccumulator that clears after finalize."""

    def test_clears_after_on_finalize(self):
        """Test that NoCopyWindowAccumulator clears after on_finalize."""
        acc = NoCopyWindowAccumulator()
        acc.push(sc.scalar(10))
        assert acc.value == sc.scalar(10)

        # Simulate what happens during finalize cycle
        acc.on_finalize()
        assert acc.is_empty

    def test_accumulates_multiple_values(self):
        """Test that values are accumulated before on_finalize."""
        acc = NoCopyWindowAccumulator()
        acc.push(sc.scalar(10))
        acc.push(sc.scalar(20))
        assert acc.value == sc.scalar(30)


class TestCreateBaseWorkflow:
    """Tests for create_base_workflow function."""

    def test_creates_workflow_with_various_configurations(self):
        """Test workflow creation with different parameter combinations."""
        bins = sc.linspace('event_time_offset', 0, 100000, 11, unit='ns')

        # With required params only
        wf1 = create_base_workflow(bins=bins)
        assert wf1 is not None

        # With histogram slice
        histogram_slice = (sc.scalar(10000, unit='ns'), sc.scalar(50000, unit='ns'))
        wf2 = create_base_workflow(bins=bins, histogram_slice=histogram_slice)
        assert wf2 is not None

        # With logical projection added
        def identity(da: sc.DataArray) -> sc.DataArray:
            return da

        wf3 = create_base_workflow(bins=bins)
        add_logical_projection(wf3, transform=identity)
        assert wf3 is not None


class TestComputeDetectorHistogram:
    """Tests for compute_detector_histogram function."""

    def test_histogram_from_screen_binned_events(self):
        """Test histogramming screen-binned events."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4)
        bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')

        # Use LogicalProjector with a fold transform
        transform = make_logical_transform(4, 4)
        projector = make_logical_projector(transform=transform, reduction_dim=None)
        screen_binned = projector.project_events(sc.values(data))

        result = compute_detector_histogram(
            screen_binned_events=screen_binned,
            bins=bins,
            event_coord='event_time_offset',
        )

        assert 'event_time_offset' in result.dims
        assert result.sizes['event_time_offset'] == 10
        assert result.sizes['y'] == 4
        assert result.sizes['x'] == 4

    def test_histogram_with_reduction_dim(self):
        """Test histogramming with reduction dimension."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')
        transform = make_logical_transform(4, 4)

        projector = make_logical_projector(transform=transform, reduction_dim='y')
        screen_binned = projector.project_events(sc.values(data))

        result = compute_detector_histogram(
            screen_binned_events=screen_binned,
            bins=bins,
            event_coord='event_time_offset',
        )

        # Should have x and event_time_offset dims, y was reduced
        assert 'event_time_offset' in result.dims
        assert 'x' in result.dims
        assert 'y' not in result.dims

    def test_histogram_preserves_user_dim_and_unit(self):
        """Test that output has user's dimension name and unit."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4)
        # Bins with user's dimension name (time_of_arrival) and unit (ms)
        bins = sc.linspace('time_of_arrival', 0, 71, 11, unit='ms')

        transform = make_logical_transform(4, 4)
        projector = make_logical_projector(transform=transform, reduction_dim=None)
        screen_binned = projector.project_events(sc.values(data))

        result = compute_detector_histogram(
            screen_binned_events=screen_binned,
            bins=bins,
            event_coord='event_time_offset',
        )

        # Output should have user's dimension name and unit
        assert 'time_of_arrival' in result.dims
        assert 'event_time_offset' not in result.dims
        assert result.coords['time_of_arrival'].unit == 'ms'
        assert sc.allclose(result.coords['time_of_arrival'], bins)


class TestAccumulatedHistogramProvider:
    """Tests for the accumulated_histogram generic provider."""

    def test_accumulated_histogram_identity(self):
        """Test that accumulated_histogram returns input unchanged."""
        data = sc.DataArray(
            sc.ones(dims=['y', 'x', 'tof'], shape=[4, 4, 10], unit='counts')
        )

        result = accumulated_histogram(DetectorHistogram(data))

        assert sc.identical(result, data)


class TestLogicalProjector:
    """Tests for LogicalProjector.project_events() and get_screen_metadata()."""

    @pytest.mark.parametrize(
        ('reduction_dim', 'expected_coords', 'expected_sizes'),
        [
            (None, ['y', 'x'], {'y': 4, 'x': 4}),
            ('y', ['x'], {'x': 4}),
            (['y', 'x'], [], {}),
        ],
        ids=['no_reduction', 'single_reduction', 'multiple_reduction'],
    )
    def test_get_screen_metadata_excludes_reduced_dims(
        self, reduction_dim, expected_coords, expected_sizes
    ):
        """Test that get_screen_metadata excludes reduced dimensions."""
        empty_detector = make_fake_empty_detector(y_size=4, x_size=4)
        transform = make_logical_transform(4, 4)

        projector = make_logical_projector(
            transform=transform, reduction_dim=reduction_dim
        )

        metadata = projector.get_screen_metadata(empty_detector)
        assert list(metadata.coords.keys()) == expected_coords
        assert metadata.sizes == expected_sizes
        # Logical projections typically don't have edges
        for coord in expected_coords:
            assert metadata.coords[coord] is None

    def test_fold_transform(self):
        """Test that fold transform reshapes detector data."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4)
        transform = make_logical_transform(4, 4)

        projector = make_logical_projector(transform=transform, reduction_dim=None)
        result = projector.project_events(sc.values(data))

        assert 'y' in result.dims
        assert 'x' in result.dims
        assert 'detector_number' not in result.dims
        assert result.sizes['y'] == 4
        assert result.sizes['x'] == 4

    def test_reduction_dim_concatenates_events(self):
        """Test that reduction_dim concatenates events along specified dim."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        transform = make_logical_transform(4, 4)

        projector = make_logical_projector(transform=transform, reduction_dim='y')
        result = projector.project_events(sc.values(data))

        # y dimension should be reduced
        assert 'y' not in result.dims
        assert 'x' in result.dims
        assert result.sizes['x'] == 4
        # Each x bin should have all events from that column
        assert result.bins.size().sum().value == 4 * 4 * 10

    def test_multiple_reduction_dims(self):
        """Test reduction over multiple dimensions."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=10)
        transform = make_logical_transform(4, 4)

        projector = make_logical_projector(
            transform=transform, reduction_dim=['y', 'x']
        )
        result = projector.project_events(sc.values(data))

        # Both dimensions should be reduced - result is scalar with all events
        assert 'y' not in result.dims
        assert 'x' not in result.dims
        assert result.dims == ()
        assert result.bins.size().sum().value == 4 * 4 * 10


class TestDetectorImageProviders:
    """Tests for detector_image generic provider function."""

    def test_detector_image_sums_over_spectral_dim(self):
        """Test that detector_image produces 2D output."""
        # Create histogram - spectral dim is last
        data = sc.DataArray(
            sc.ones(
                dims=['y', 'x', 'event_time_offset'], shape=[4, 4, 10], unit='counts'
            )
        )
        weights = PixelWeights(sc.ones(dims=['y', 'x'], shape=[4, 4], dtype='float32'))

        result = detector_image(
            histogram=AccumulatedHistogram[Cumulative](data),
            histogram_slice=None,
            weights=weights,
            use_weighting=UsePixelWeighting(False),
        )

        assert result.dims == ('y', 'x')
        assert result.sizes == {'y': 4, 'x': 4}
        # Each pixel should have sum of 10 spectral bins
        expected = sc.full(dims=['y', 'x'], shape=[4, 4], value=10.0, unit='counts')
        assert sc.allclose(result.data, expected)

    def test_detector_image_with_histogram_slice(self):
        """Test that histogram slicing is applied correctly."""
        # Create histogram with coordinate
        coord = sc.linspace('event_time_offset', 0, 100000, 11, unit='ns')
        data = sc.DataArray(
            sc.ones(
                dims=['y', 'x', 'event_time_offset'], shape=[4, 4, 10], unit='counts'
            ),
            coords={'event_time_offset': coord},
        )
        weights = PixelWeights(sc.ones(dims=['y', 'x'], shape=[4, 4], dtype='float32'))

        # Slice to first half
        histogram_slice = (sc.scalar(0, unit='ns'), sc.scalar(50000, unit='ns'))

        result = detector_image(
            histogram=AccumulatedHistogram[Current](data),
            histogram_slice=histogram_slice,
            weights=weights,
            use_weighting=UsePixelWeighting(False),
        )

        # Should only sum ~5 bins (0-50000 ns from 0-100000 ns range)
        assert result.dims == ('y', 'x')


class TestCountProviders:
    """Tests for count provider functions."""

    def test_counts_total(self):
        """Test that counts_total sums all counts."""
        data = sc.DataArray(
            sc.ones(
                dims=['y', 'x', 'event_time_offset'], shape=[4, 4, 10], unit='counts'
            )
        )

        result = counts_total(histogram=AccumulatedHistogram[Current](data))

        expected = sc.scalar(4 * 4 * 10, unit='counts', dtype='float64')
        assert sc.identical(result.data, expected)

    def test_counts_in_range_with_and_without_slice(self):
        """Test counts_in_range with and without histogram slice."""
        coord = sc.linspace('event_time_offset', 0, 100000, 11, unit='ns')
        data = sc.DataArray(
            sc.ones(
                dims=['y', 'x', 'event_time_offset'], shape=[4, 4, 10], unit='counts'
            ),
            coords={'event_time_offset': coord},
        )

        # Without slice - should count all bins
        result_no_slice = counts_in_range(
            histogram=AccumulatedHistogram[Current](data), histogram_slice=None
        )
        expected_all = sc.scalar(4 * 4 * 10, unit='counts', dtype='float64')
        assert sc.identical(result_no_slice.data, expected_all)

        # With slice to first half - should count approximately half
        histogram_slice = (sc.scalar(0, unit='ns'), sc.scalar(50000, unit='ns'))
        result_sliced = counts_in_range(
            histogram=AccumulatedHistogram[Current](data),
            histogram_slice=histogram_slice,
        )
        assert result_sliced.dims == ()
        assert result_sliced.data < expected_all
