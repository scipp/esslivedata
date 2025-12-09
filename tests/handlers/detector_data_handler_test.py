# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.livedata import StreamKind
from ess.livedata.config import instrument_registry
from ess.livedata.config.instrument import Instrument
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.core.handler import StreamId
from ess.livedata.handlers.accumulators import LatestValue
from ess.livedata.handlers.detector_data_handler import (
    DetectorHandlerFactory,
    DetectorLogicalView,
    get_nexus_geometry_filename,
)
from ess.livedata.handlers.detector_view_specs import DetectorViewParams


def get_instrument(instrument_name: str) -> Instrument:
    _ = get_config(instrument_name)  # Load the module to register the instrument
    instrument = instrument_registry[instrument_name]
    instrument.load_factories()
    return instrument


@pytest.mark.parametrize('instrument', ['dream', 'loki'])
def test_get_nexus_filename_returns_file_for_given_date(instrument: str) -> None:
    filename = get_nexus_geometry_filename(
        instrument, date=sc.datetime('2025-01-02T00:00:00')
    )
    assert str(filename).endswith(f'geometry-{instrument}-2025-01-01.nxs')


def test_get_nexus_filename_uses_current_date_by_default() -> None:
    auto = get_nexus_geometry_filename('dream')
    explicit = get_nexus_geometry_filename('dream', date=sc.datetime('now'))
    assert auto == explicit


def test_get_nexus_filename_raises_if_instrument_unknown() -> None:
    with pytest.raises(ValueError, match='No geometry files found for instrument'):
        get_nexus_geometry_filename('abcde', date=sc.datetime('2025-01-01T00:00:00'))


def test_get_nexus_filename_raises_if_datetime_out_of_range() -> None:
    with pytest.raises(ValueError, match='No geometry file found for given date'):
        get_nexus_geometry_filename('dream', date=sc.datetime('2020-01-01T00:00:00'))


@pytest.mark.parametrize('instrument_name', available_instruments())
def test_factory_can_create_preprocessor(instrument_name: str) -> None:
    instrument = get_instrument(instrument_name)
    factory = DetectorHandlerFactory(instrument=instrument)
    for name in instrument.detector_names:
        # Try to get detector_number to determine if this is an event detector
        try:
            _ = instrument.get_detector_number(name)
            kind = StreamKind.DETECTOR_EVENTS
        except KeyError:
            # No detector_number means this is an area detector
            kind = StreamKind.AREA_DETECTOR
        _ = factory.make_preprocessor(StreamId(kind=kind, name=name))


def test_factory_creates_latest_value_accumulator_for_roi_messages() -> None:
    """Test that DetectorHandlerFactory creates a LatestValue accumulator for ROI."""
    instrument = get_instrument('dummy')
    factory = DetectorHandlerFactory(instrument=instrument)

    # Create a stream ID for an ROI message
    roi_stream_id = StreamId(
        kind=StreamKind.LIVEDATA_ROI, name='test-job-123/roi_rectangle'
    )

    preprocessor = factory.make_preprocessor(roi_stream_id)

    # Should return a LatestValue accumulator
    assert preprocessor is not None
    assert isinstance(preprocessor, LatestValue)


def test_factory_returns_none_for_unknown_stream_kinds() -> None:
    """Test that DetectorHandlerFactory returns None for unknown stream kinds."""
    instrument = get_instrument('dummy')
    factory = DetectorHandlerFactory(instrument=instrument)

    # Try various stream kinds that should not be handled
    unknown_stream_id = StreamId(kind=StreamKind.LOG, name='some_log')
    preprocessor = factory.make_preprocessor(unknown_stream_id)
    assert preprocessor is None

    config_stream_id = StreamId(kind=StreamKind.LIVEDATA_COMMANDS, name='config')
    preprocessor = factory.make_preprocessor(config_stream_id)
    assert preprocessor is None


def test_factory_returns_none_for_unconfigured_detectors() -> None:
    """Test that DetectorHandlerFactory returns None for unconfigured detectors.

    This ensures that messages from detectors not in the instrument's detector_names
    list are gracefully skipped instead of causing a KeyError.
    """
    instrument = get_instrument('dummy')
    factory = DetectorHandlerFactory(instrument=instrument)

    # dummy is configured with 'panel_0_detector' but not 'unknown_detector'
    unconfigured_detector = StreamId(
        kind=StreamKind.DETECTOR_EVENTS, name='unknown_detector'
    )
    preprocessor = factory.make_preprocessor(unconfigured_detector)

    # Should return None to indicate this detector should be skipped
    assert preprocessor is None


class FakeInstrument:
    """Minimal fake instrument for testing detector view factories."""

    def __init__(self, detector_number: sc.Variable) -> None:
        self._detector_number = detector_number

    def get_detector_number(self, source_name: str) -> sc.Variable:
        return self._detector_number


class TestDetectorLogicalView:
    """Tests for DetectorLogicalView factory."""

    def test_make_view_creates_detector_view(self) -> None:
        """Test that make_view creates a DetectorView."""
        detector_number = sc.arange('pixel', 16, dtype='int64')
        instrument = FakeInstrument(detector_number)
        factory = DetectorLogicalView(instrument=instrument)

        view = factory.make_view('test_detector', DetectorViewParams())

        from ess.livedata.handlers.detector_view import DetectorView

        assert isinstance(view, DetectorView)

    def test_make_view_with_transform(self) -> None:
        """Test that make_view applies the transform."""
        detector_number = sc.arange('x', 16, dtype='int64').fold(
            dim='x', sizes={'x': 4, 'y': 4}
        )
        instrument = FakeInstrument(detector_number)

        # A transform that sums over y
        def sum_y(da: sc.DataArray) -> sc.DataArray:
            return da.sum('y')

        factory = DetectorLogicalView(instrument=instrument, transform=sum_y)
        view = factory.make_view('test_detector', DetectorViewParams())

        # Add some events and check shape
        view._view.add_counts(list(range(16)))
        result = view._view.get()
        assert result.sizes == {'x': 4}

    def test_make_view_without_transform_preserves_shape(self) -> None:
        """Test that make_view without transform preserves detector shape."""
        detector_number = sc.arange('x', 16, dtype='int64').fold(
            dim='x', sizes={'x': 4, 'y': 4}
        )
        instrument = FakeInstrument(detector_number)
        factory = DetectorLogicalView(instrument=instrument)

        view = factory.make_view('test_detector', DetectorViewParams())

        view._view.add_counts(list(range(16)))
        result = view._view.get()
        assert result.sizes == {'x': 4, 'y': 4}


class TestDetectorLogicalViewWithReduction:
    """Tests for DetectorLogicalView with reduction_dim parameter.

    When reduction_dim is specified, LogicalView from ess.reduce.live provides
    proper index mapping for ROI support.
    """

    @pytest.fixture
    def detector_number_2d(self) -> sc.Variable:
        """8x8 detector number array for downsampling tests."""
        return sc.arange('x', 64, dtype='int64').fold(dim='x', sizes={'x': 8, 'y': 8})

    @pytest.fixture
    def fold_transform(self):
        """Transform that folds 8x8 to 4x4 with 2x2 bins."""

        def transform(da: sc.DataArray) -> sc.DataArray:
            da = da.fold(dim='x', sizes={'x': 4, 'x_bin': 2})
            da = da.fold(dim='y', sizes={'y': 4, 'y_bin': 2})
            return da

        return transform

    def test_make_view_creates_detector_view(
        self, detector_number_2d: sc.Variable, fold_transform
    ) -> None:
        """Test that make_view creates a DetectorView."""
        instrument = FakeInstrument(detector_number_2d)
        factory = DetectorLogicalView(
            instrument=instrument,
            transform=fold_transform,
            reduction_dim=['x_bin', 'y_bin'],
        )

        view = factory.make_view('test_detector', DetectorViewParams())

        from ess.livedata.handlers.detector_view import DetectorView

        assert isinstance(view, DetectorView)

    def test_make_view_applies_downsampling(
        self, detector_number_2d: sc.Variable, fold_transform
    ) -> None:
        """Test that the view correctly downsamples from 8x8 to 4x4."""
        instrument = FakeInstrument(detector_number_2d)
        factory = DetectorLogicalView(
            instrument=instrument,
            transform=fold_transform,
            reduction_dim=['x_bin', 'y_bin'],
        )

        view = factory.make_view('test_detector', DetectorViewParams())

        # Add events to all pixels
        view._view.add_counts(list(range(64)))
        result = view._view.get()

        # Should be downsampled to 4x4
        assert result.sizes == {'x': 4, 'y': 4}
        # Each output pixel should have events from 4 input pixels (2x2)
        assert result.values[0, 0] == 4

    def test_make_roi_filter_returns_binned_indices(
        self, detector_number_2d: sc.Variable, fold_transform
    ) -> None:
        """Test that ROI filter has binned indices for proper ROI support."""
        instrument = FakeInstrument(detector_number_2d)
        factory = DetectorLogicalView(
            instrument=instrument,
            transform=fold_transform,
            reduction_dim=['x_bin', 'y_bin'],
        )

        view = factory.make_view('test_detector', DetectorViewParams())
        roi_filter = view._view.make_roi_filter()

        # The indices should be binned (each output pixel contains a bin of
        # input indices)
        assert roi_filter._indices.bins is not None
        # Shape should match downsampled output
        assert roi_filter._indices.sizes == {'x': 4, 'y': 4}

    def test_roi_filter_indices_contain_correct_input_pixels(
        self, detector_number_2d: sc.Variable, fold_transform
    ) -> None:
        """Test that each output bin contains the correct 4 input pixel indices."""
        instrument = FakeInstrument(detector_number_2d)
        factory = DetectorLogicalView(
            instrument=instrument,
            transform=fold_transform,
            reduction_dim=['x_bin', 'y_bin'],
        )

        view = factory.make_view('test_detector', DetectorViewParams())
        roi_filter = view._view.make_roi_filter()

        # Check first output pixel (0,0) contains 4 input indices
        first_bin = roi_filter._indices['x', 0]['y', 0]
        assert first_bin.bins.size().value == 4

    def test_single_reduction_dim(self, detector_number_2d: sc.Variable) -> None:
        """Test with a single reduction dimension (string instead of list)."""
        instrument = FakeInstrument(detector_number_2d)

        # Fold only x, sum only over x_bin
        def fold_x_only(da: sc.DataArray) -> sc.DataArray:
            return da.fold(dim='x', sizes={'x': 4, 'x_bin': 2})

        factory = DetectorLogicalView(
            instrument=instrument,
            transform=fold_x_only,
            reduction_dim='x_bin',  # Single string
        )

        view = factory.make_view('test_detector', DetectorViewParams())
        view._view.add_counts(list(range(64)))
        result = view._view.get()

        # Should have shape (4, 8) - x downsampled, y preserved
        assert result.sizes == {'x': 4, 'y': 8}


class TestDetectorProjectionMixedProjections:
    """Tests for DetectorProjection with mixed projection types."""

    def test_get_projection_with_single_projection_type(self) -> None:
        """Test that single projection type is returned for all sources."""
        from ess.livedata.handlers.detector_data_handler import DetectorProjection

        instrument = FakeInstrument(sc.arange('pixel', 16, dtype='int64'))
        projection = DetectorProjection(
            instrument=instrument,
            projection='xy_plane',
            resolution={'detector1': {'x': 10, 'y': 10}},
        )

        assert projection._get_projection('detector1') == 'xy_plane'
        assert projection._get_projection('detector2') == 'xy_plane'

    def test_get_projection_with_dict_projection_type(self) -> None:
        """Test that dict projection returns per-source projection types."""
        from ess.livedata.handlers.detector_data_handler import DetectorProjection

        instrument = FakeInstrument(sc.arange('pixel', 16, dtype='int64'))
        projections = {
            'mantle': 'cylinder_mantle_z',
            'endcap': 'xy_plane',
        }
        projection = DetectorProjection(
            instrument=instrument,
            projection=projections,
            resolution={
                'mantle': {'arc_length': 10, 'z': 40},
                'endcap': {'x': 10, 'y': 20},
            },
        )

        assert projection._get_projection('mantle') == 'cylinder_mantle_z'
        assert projection._get_projection('endcap') == 'xy_plane'

    def test_get_projection_raises_for_unknown_source_with_dict(self) -> None:
        """Test that accessing unknown source raises KeyError with dict projections."""
        from ess.livedata.handlers.detector_data_handler import DetectorProjection

        instrument = FakeInstrument(sc.arange('pixel', 16, dtype='int64'))
        projections = {
            'mantle': 'cylinder_mantle_z',
        }
        projection = DetectorProjection(
            instrument=instrument,
            projection=projections,
            resolution={'mantle': {'arc_length': 10, 'z': 40}},
        )

        with pytest.raises(KeyError):
            projection._get_projection('unknown_detector')


class TestDetectorLogicalViewROISupport:
    """Tests for ROI support in DetectorLogicalView without reduction."""

    def test_roi_filter_works_without_transform(self) -> None:
        """Test that ROI filter works for simple view without transform."""
        detector_number = sc.arange('x', 16, dtype='int64').fold(
            dim='x', sizes={'x': 4, 'y': 4}
        )
        instrument = FakeInstrument(detector_number)
        factory = DetectorLogicalView(instrument=instrument)

        view = factory.make_view('test_detector', DetectorViewParams())
        roi_filter = view._view.make_roi_filter()

        # ROI filter should work and have correct shape
        assert roi_filter._indices.sizes == {'x': 4, 'y': 4}

    def test_roi_filter_works_with_transform_no_reduction(self) -> None:
        """Test ROI filter with transform that doesn't reduce dimensions."""
        detector_number = sc.arange('x', 16, dtype='int64').fold(
            dim='x', sizes={'x': 4, 'y': 4}
        )
        instrument = FakeInstrument(detector_number)

        # Transform that just transposes (no reduction)
        def transpose_dims(da: sc.DataArray) -> sc.DataArray:
            return da.transpose(('y', 'x'))

        factory = DetectorLogicalView(instrument=instrument, transform=transpose_dims)

        view = factory.make_view('test_detector', DetectorViewParams())
        roi_filter = view._view.make_roi_filter()

        # ROI filter should work with transposed shape
        assert roi_filter._indices.sizes == {'y': 4, 'x': 4}
