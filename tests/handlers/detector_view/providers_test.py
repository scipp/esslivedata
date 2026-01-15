# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for Sciline provider functions."""

import scipp as sc

from ess.livedata.handlers.detector_view import (
    Cumulative,
    Current,
    DetectorHistogram3D,
    Histogram3D,
    NoCopyAccumulator,
    NoCopyWindowAccumulator,
    PixelWeights,
    UsePixelWeighting,
    add_logical_projection,
    compute_detector_histogram_3d,
    counts_in_range,
    counts_total,
    create_accumulators,
    create_base_workflow,
    detector_image,
    histogram_3d,
    make_logical_projector,
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

    def test_creates_workflow_with_required_params(self):
        """Test workflow creation with required parameters."""
        bins = sc.linspace('event_time_offset', 0, 100000, 11, unit='ns')
        wf = create_base_workflow(bins=bins)
        assert wf is not None

    def test_creates_workflow_with_histogram_slice(self):
        """Test workflow creation with histogram slice parameter."""
        bins = sc.linspace('event_time_offset', 0, 100000, 11, unit='ns')
        histogram_slice = (sc.scalar(10000, unit='ns'), sc.scalar(50000, unit='ns'))
        wf = create_base_workflow(bins=bins, histogram_slice=histogram_slice)
        assert wf is not None

    def test_creates_workflow_and_add_logical_projection(self):
        """Test workflow creation with logical projection added separately."""
        bins = sc.linspace('event_time_offset', 0, 100000, 11, unit='ns')

        def identity(da: sc.DataArray) -> sc.DataArray:
            return da

        wf = create_base_workflow(bins=bins)
        add_logical_projection(wf, transform=identity)
        assert wf is not None


class TestCreateAccumulators:
    """Tests for create_accumulators function."""

    def test_creates_correct_accumulator_types(self):
        """Test that correct accumulator types are created."""
        accumulators = create_accumulators()

        assert Histogram3D[Cumulative] in accumulators
        assert Histogram3D[Current] in accumulators
        # Uses NoCopy variants for performance (~2x faster with large histograms)
        assert isinstance(accumulators[Histogram3D[Cumulative]], NoCopyAccumulator)
        assert isinstance(accumulators[Histogram3D[Current]], NoCopyWindowAccumulator)


class TestComputeDetectorHistogram3D:
    """Tests for compute_detector_histogram_3d function."""

    def test_histogram_from_screen_binned_events(self):
        """Test histogramming screen-binned events to 3D."""
        data = make_fake_nexus_detector_data(y_size=4, x_size=4)
        bins = sc.linspace('event_time_offset', 0, 71_000_000, 11, unit='ns')

        # Use LogicalProjector with a fold transform
        transform = make_logical_transform(4, 4)
        projector = make_logical_projector(transform=transform, reduction_dim=None)
        screen_binned = projector.project_events(sc.values(data))

        result = compute_detector_histogram_3d(
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

        result = compute_detector_histogram_3d(
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

        result = compute_detector_histogram_3d(
            screen_binned_events=screen_binned,
            bins=bins,
            event_coord='event_time_offset',
        )

        # Output should have user's dimension name and unit
        assert 'time_of_arrival' in result.dims
        assert 'event_time_offset' not in result.dims
        assert result.coords['time_of_arrival'].unit == 'ms'
        assert sc.allclose(result.coords['time_of_arrival'], bins)


class TestHistogram3DProvider:
    """Tests for the histogram_3d generic provider."""

    def test_histogram_3d_identity(self):
        """Test that histogram_3d returns input unchanged."""
        data_3d = sc.DataArray(
            sc.ones(dims=['y', 'x', 'tof'], shape=[4, 4, 10], unit='counts')
        )

        result = histogram_3d(DetectorHistogram3D(data_3d))

        assert sc.identical(result, data_3d)


class TestLogicalProjector:
    """Tests for LogicalProjector.project_events() and get_screen_metadata()."""

    def test_get_screen_metadata_returns_all_dims_without_reduction(self):
        """Test that get_screen_metadata contains all dims when no reduction."""
        empty_detector = make_fake_empty_detector(y_size=4, x_size=4)
        transform = make_logical_transform(4, 4)

        projector = make_logical_projector(transform=transform, reduction_dim=None)

        metadata = projector.get_screen_metadata(empty_detector)
        assert list(metadata.coords.keys()) == ['y', 'x']
        # Logical projections typically don't have edges
        assert metadata.coords['y'] is None
        assert metadata.coords['x'] is None
        assert metadata.sizes == {'y': 4, 'x': 4}

    def test_get_screen_metadata_excludes_reduced_dims(self):
        """Test that get_screen_metadata excludes dimensions that will be reduced."""
        empty_detector = make_fake_empty_detector(y_size=4, x_size=4)
        transform = make_logical_transform(4, 4)

        projector = make_logical_projector(transform=transform, reduction_dim='y')

        metadata = projector.get_screen_metadata(empty_detector)
        assert list(metadata.coords.keys()) == ['x']
        assert 'y' not in metadata.coords
        assert metadata.sizes == {'x': 4}

    def test_get_screen_metadata_excludes_multiple_reduced_dims(self):
        """Test that get_screen_metadata excludes multiple reduced dimensions."""
        empty_detector = make_fake_empty_detector(y_size=4, x_size=4)
        transform = make_logical_transform(4, 4)

        projector = make_logical_projector(
            transform=transform, reduction_dim=['y', 'x']
        )

        metadata = projector.get_screen_metadata(empty_detector)
        assert list(metadata.coords.keys()) == []
        assert metadata.sizes == {}

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
        # Create 3D histogram - spectral dim is last
        data_3d = sc.DataArray(
            sc.ones(
                dims=['y', 'x', 'event_time_offset'], shape=[4, 4, 10], unit='counts'
            )
        )
        weights = PixelWeights(sc.ones(dims=['y', 'x'], shape=[4, 4], dtype='float32'))

        result = detector_image(
            data_3d=Histogram3D[Cumulative](data_3d),
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
        # Create 3D histogram with coordinate
        coord = sc.linspace('event_time_offset', 0, 100000, 11, unit='ns')
        data_3d = sc.DataArray(
            sc.ones(
                dims=['y', 'x', 'event_time_offset'], shape=[4, 4, 10], unit='counts'
            ),
            coords={'event_time_offset': coord},
        )
        weights = PixelWeights(sc.ones(dims=['y', 'x'], shape=[4, 4], dtype='float32'))

        # Slice to first half
        histogram_slice = (sc.scalar(0, unit='ns'), sc.scalar(50000, unit='ns'))

        result = detector_image(
            data_3d=Histogram3D[Current](data_3d),
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
        data_3d = sc.DataArray(
            sc.ones(
                dims=['y', 'x', 'event_time_offset'], shape=[4, 4, 10], unit='counts'
            )
        )

        result = counts_total(data_3d=Histogram3D[Current](data_3d))

        expected = sc.scalar(4 * 4 * 10, unit='counts', dtype='float64')
        assert sc.identical(result.data, expected)

    def test_counts_in_range_no_slice(self):
        """Test counts_in_range with no slice."""
        data_3d = sc.DataArray(
            sc.ones(
                dims=['y', 'x', 'event_time_offset'], shape=[4, 4, 10], unit='counts'
            )
        )

        result = counts_in_range(
            data_3d=Histogram3D[Current](data_3d), histogram_slice=None
        )

        expected = sc.scalar(4 * 4 * 10, unit='counts', dtype='float64')
        assert sc.identical(result.data, expected)

    def test_counts_in_range_with_slice(self):
        """Test counts_in_range with histogram slice."""
        coord = sc.linspace('event_time_offset', 0, 100000, 11, unit='ns')
        data_3d = sc.DataArray(
            sc.ones(
                dims=['y', 'x', 'event_time_offset'], shape=[4, 4, 10], unit='counts'
            ),
            coords={'event_time_offset': coord},
        )

        # Slice to first half
        histogram_slice = (sc.scalar(0, unit='ns'), sc.scalar(50000, unit='ns'))

        result = counts_in_range(
            data_3d=Histogram3D[Current](data_3d), histogram_slice=histogram_slice
        )

        # Should count approximately half the bins
        assert result.dims == ()
