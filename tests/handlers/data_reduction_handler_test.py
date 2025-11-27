# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata import StreamId, StreamKind
from ess.livedata.config.instrument import Instrument
from ess.livedata.handlers.data_reduction_handler import ReductionHandlerFactory
from ess.livedata.handlers.to_nxlog import ToNXlog


class TestReductionHandlerFactory:
    @pytest.fixture
    def instrument_with_logs(self):
        """Create an instrument with log attribute configuration."""
        instrument = Instrument(name='test_instrument', detector_names=['det1'])
        instrument.f144_attribute_registry = {
            'temp_sensor': {
                'value': {'shape': [], 'dtype': 'float'},
                'timestamp': {'shape': [], 'dtype': 'uint64'},
                'offset': 0.0,
            },
            'pressure_sensor': {
                'value': {'shape': [], 'dtype': 'float'},
                'timestamp': {'shape': [], 'dtype': 'uint64'},
                'offset': 0.0,
            },
        }
        return instrument

    @pytest.fixture
    def factory(self, instrument_with_logs):
        """Create ReductionHandlerFactory instance."""
        return ReductionHandlerFactory(instrument=instrument_with_logs)

    def test_make_preprocessor_monitor_counts(self, factory):
        """Test that MONITOR_COUNTS returns Cumulative accumulator."""
        stream_id = StreamId(kind=StreamKind.MONITOR_COUNTS, name='test')
        preprocessor = factory.make_preprocessor(stream_id)

        from ess.livedata.handlers.accumulators import Cumulative

        assert isinstance(preprocessor, Cumulative)
        assert preprocessor._clear_on_get is True

    def test_make_preprocessor_log_configured(self, factory):
        """Test that LOG returns ToNXlog for configured sources."""
        stream_id = StreamId(kind=StreamKind.LOG, name='temp_sensor')
        preprocessor = factory.make_preprocessor(stream_id)

        assert isinstance(preprocessor, ToNXlog)

    def test_make_preprocessor_log_unconfigured(self, factory):
        """Test that LOG returns None for unconfigured sources.

        This ensures that log data for sources not in the attribute registry
        are gracefully skipped instead of causing a KeyError.
        """
        stream_id = StreamId(kind=StreamKind.LOG, name='unknown_sensor')
        preprocessor = factory.make_preprocessor(stream_id)

        # Should return None to indicate this log should be skipped
        assert preprocessor is None

    def test_make_preprocessor_monitor_events(self, factory):
        """Test that MONITOR_EVENTS returns ToNXevent_data."""
        stream_id = StreamId(kind=StreamKind.MONITOR_EVENTS, name='test')
        preprocessor = factory.make_preprocessor(stream_id)

        from ess.livedata.handlers.to_nxevent_data import ToNXevent_data

        assert isinstance(preprocessor, ToNXevent_data)

    def test_make_preprocessor_detector_events(self, factory):
        """Test that DETECTOR_EVENTS returns ToNXevent_data."""
        stream_id = StreamId(kind=StreamKind.DETECTOR_EVENTS, name='test')
        preprocessor = factory.make_preprocessor(stream_id)

        from ess.livedata.handlers.to_nxevent_data import ToNXevent_data

        assert isinstance(preprocessor, ToNXevent_data)

    def test_make_preprocessor_area_detector(self, factory):
        """Test that AREA_DETECTOR returns Cumulative accumulator."""
        stream_id = StreamId(kind=StreamKind.AREA_DETECTOR, name='test')
        preprocessor = factory.make_preprocessor(stream_id)

        from ess.livedata.handlers.accumulators import Cumulative

        assert isinstance(preprocessor, Cumulative)
        assert preprocessor._clear_on_get is True

    def test_make_preprocessor_unknown_kind(self, factory):
        """Test that unknown stream kinds return None."""
        stream_id = StreamId(kind=StreamKind.LIVEDATA_ROI, name='test')
        preprocessor = factory.make_preprocessor(stream_id)

        assert preprocessor is None
