# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import pytest
import scipp as sc

from ess.livedata.dashboard.extractors import (
    FullHistoryExtractor,
    LatestValueExtractor,
    WindowAggregatingExtractor,
)
from ess.livedata.dashboard.temporal_buffer_manager import TemporalBufferManager
from ess.livedata.dashboard.temporal_buffers import SingleValueBuffer, TemporalBuffer


class TestTemporalBufferManager:
    """Tests for TemporalBufferManager."""

    def test_create_buffer_with_only_latest_extractors_uses_single_value_buffer(self):
        """
        Test that SingleValueBuffer is used with all LatestValueExtractor.
        """
        manager = TemporalBufferManager()
        extractors = [LatestValueExtractor(), LatestValueExtractor()]

        manager.create_buffer('test', extractors)

        assert isinstance(manager._states['test'].buffer, SingleValueBuffer)

    def test_create_buffer_with_mixed_extractors_uses_temporal_buffer(self):
        """
        Test that TemporalBuffer is used with mixed extractors.
        """
        manager = TemporalBufferManager()
        extractors = [LatestValueExtractor(), FullHistoryExtractor()]

        manager.create_buffer('test', extractors)

        assert isinstance(manager._states['test'].buffer, TemporalBuffer)

    def test_create_buffer_with_window_extractor_uses_temporal_buffer(self):
        """Test that TemporalBuffer is used with WindowAggregatingExtractor."""
        manager = TemporalBufferManager()
        extractors = [WindowAggregatingExtractor(window_duration_seconds=1.0)]

        manager.create_buffer('test', extractors)

        assert isinstance(manager._states['test'].buffer, TemporalBuffer)

    def test_create_buffer_with_no_extractors_uses_single_value_buffer(self):
        """
        Test that SingleValueBuffer is used by default with no extractors.
        """
        manager = TemporalBufferManager()

        manager.create_buffer('test', [])

        assert isinstance(manager._states['test'].buffer, SingleValueBuffer)

    def test_create_buffer_raises_error_for_duplicate_key(self):
        """Test that creating a buffer with existing key raises ValueError."""
        manager = TemporalBufferManager()
        extractors = [LatestValueExtractor()]

        manager.create_buffer('test', extractors)

        with pytest.raises(ValueError, match="already exists"):
            manager.create_buffer('test', extractors)

    def test_update_buffer_adds_data(self):
        """Test that update_buffer adds data to the buffer."""
        manager = TemporalBufferManager()
        extractors = [LatestValueExtractor()]
        data = sc.scalar(42, unit='counts')

        manager.create_buffer('test', extractors)
        manager.update_buffer('test', data)

        result = manager.get_buffered_data('test')
        assert result == data

    def test_update_buffer_raises_error_for_missing_key(self):
        """Test that updating non-existent buffer raises KeyError."""
        manager = TemporalBufferManager()
        data = sc.scalar(42, unit='counts')

        with pytest.raises(KeyError, match="No buffer found"):
            manager.update_buffer('test', data)

    def test_add_extractor_keeps_same_buffer_type(self):
        """Test that adding compatible extractor keeps same buffer type."""
        manager = TemporalBufferManager()
        extractors = [LatestValueExtractor()]

        manager.create_buffer('test', extractors)
        original_buffer = manager._states['test'].buffer

        manager.add_extractor('test', LatestValueExtractor())

        assert manager._states['test'].buffer is original_buffer
        assert isinstance(manager._states['test'].buffer, SingleValueBuffer)

    def test_add_extractor_switches_to_temporal_buffer(self):
        """Test that switching buffer types preserves existing data."""
        manager = TemporalBufferManager()
        extractors = [LatestValueExtractor()]
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.scalar(1.0, unit='s'),
            },
        )

        manager.create_buffer('test', extractors)
        manager.update_buffer('test', data)

        # Add full history extractor - should trigger buffer type switch
        manager.add_extractor('test', FullHistoryExtractor())

        # Data should be preserved when switching
        result = manager.get_buffered_data('test')
        assert result is not None
        # After switching, buffer transforms scalar time coord to time dimension
        assert 'time' in result.dims
        assert result.sizes['time'] == 1
        # Verify the data values are preserved
        assert sc.allclose(result['time', 0].data, data.data)

    def test_add_extractor_switches_to_single_value_buffer(self):
        """Test that switching buffer types preserves latest data."""
        manager = TemporalBufferManager()
        extractors = [WindowAggregatingExtractor(window_duration_seconds=1.0)]

        manager.create_buffer('test', extractors)

        # Add multiple time slices
        for i in range(3):
            data = sc.DataArray(
                sc.array(dims=['x'], values=[float(i)] * 2, unit='counts'),
                coords={
                    'x': sc.arange('x', 2, unit='m'),
                    'time': sc.scalar(float(i), unit='s'),
                },
            )
            manager.update_buffer('test', data)

        # Verify we have temporal data with 3 time points
        result = manager.get_buffered_data('test')
        assert result is not None
        assert 'time' in result.dims
        assert result.sizes['time'] == 3

        # Manually clear extractors to simulate reconfiguration
        state = manager._states['test']
        state.extractors.clear()

        # Add LatestValueExtractor - this should trigger buffer type switch
        manager.add_extractor('test', LatestValueExtractor())

        # Verify the latest time slice is preserved after transition
        result = manager.get_buffered_data('test')
        assert result is not None
        # The last slice should have values [2.0, 2.0] and time=2.0
        expected_data = sc.array(dims=['x'], values=[2.0, 2.0], unit='counts')
        assert sc.allclose(result.data, expected_data)
        assert result.coords['time'].value == 2.0

    def test_add_extractor_raises_error_for_missing_key(self):
        """Test that adding extractor to non-existent buffer raises KeyError."""
        manager = TemporalBufferManager()

        with pytest.raises(KeyError, match="No buffer found"):
            manager.add_extractor('test', LatestValueExtractor())

    def test_delete_buffer_removes_buffer(self):
        """Test that delete_buffer removes the buffer."""
        manager = TemporalBufferManager()
        extractors = [LatestValueExtractor()]

        manager.create_buffer('test', extractors)
        assert 'test' in manager

        manager.delete_buffer('test')
        assert 'test' not in manager

    def test_delete_buffer_nonexistent_key_does_nothing(self):
        """Test that deleting non-existent buffer doesn't raise error."""
        manager = TemporalBufferManager()
        manager.delete_buffer('nonexistent')  # Should not raise

    def test_mapping_interface(self):
        """Test that manager implements Mapping interface correctly."""
        manager = TemporalBufferManager()
        extractors = [LatestValueExtractor()]

        manager.create_buffer('key1', extractors)
        manager.create_buffer('key2', extractors)

        assert len(manager) == 2
        assert 'key1' in manager
        assert 'key2' in manager
        assert list(manager) == ['key1', 'key2']


class TestTemporalBufferManagerTimespanPropagation:
    """Tests for timespan requirement propagation."""

    def test_window_extractor_sets_timespan_on_buffer(self):
        """Test that WindowAggregatingExtractor's timespan is set on buffer."""
        manager = TemporalBufferManager()
        window_duration = 5.0
        extractors = [
            WindowAggregatingExtractor(window_duration_seconds=window_duration)
        ]

        manager.create_buffer('test', extractors)

        buffer = manager._states['test'].buffer
        assert isinstance(buffer, TemporalBuffer)
        assert buffer._required_timespan == window_duration

    def test_multiple_window_extractors_use_max_timespan(self):
        """Test that maximum timespan from multiple extractors is used."""
        manager = TemporalBufferManager()
        extractors = [
            WindowAggregatingExtractor(window_duration_seconds=3.0),
            WindowAggregatingExtractor(window_duration_seconds=5.0),
            WindowAggregatingExtractor(window_duration_seconds=2.0),
        ]

        manager.create_buffer('test', extractors)

        buffer = manager._states['test'].buffer
        assert buffer._required_timespan == 5.0

    def test_latest_extractor_does_not_set_timespan(self):
        """Test that LatestValueExtractor doesn't set a timespan."""
        manager = TemporalBufferManager()
        extractors = [LatestValueExtractor()]

        manager.create_buffer('test', extractors)

        buffer = manager._states['test'].buffer
        assert isinstance(buffer, SingleValueBuffer)
        assert buffer._required_timespan == 0.0

    def test_mixed_extractors_use_window_timespan(self):
        """Test that timespan is set when mixing Latest and Window extractors."""
        manager = TemporalBufferManager()
        extractors = [
            LatestValueExtractor(),
            WindowAggregatingExtractor(window_duration_seconds=4.0),
        ]

        manager.create_buffer('test', extractors)

        buffer = manager._states['test'].buffer
        assert isinstance(buffer, TemporalBuffer)
        assert buffer._required_timespan == 4.0

    def test_adding_extractor_updates_timespan(self):
        """Test that adding an extractor updates the buffer's timespan."""
        manager = TemporalBufferManager()
        extractors = [WindowAggregatingExtractor(window_duration_seconds=2.0)]

        manager.create_buffer('test', extractors)
        buffer = manager._states['test'].buffer
        assert buffer._required_timespan == 2.0

        # Add extractor with larger timespan
        manager.add_extractor(
            'test', WindowAggregatingExtractor(window_duration_seconds=10.0)
        )

        assert buffer._required_timespan == 10.0

    def test_full_history_extractor_infinite_timespan(self):
        """Test that FullHistoryExtractor sets infinite timespan."""
        manager = TemporalBufferManager()
        extractors = [FullHistoryExtractor()]

        manager.create_buffer('test', extractors)

        buffer = manager._states['test'].buffer
        assert isinstance(buffer, TemporalBuffer)
        assert buffer._required_timespan == float('inf')

    def test_full_history_with_window_uses_infinite(self):
        """Test that mixing FullHistory with Window uses infinite timespan."""
        manager = TemporalBufferManager()
        extractors = [
            WindowAggregatingExtractor(window_duration_seconds=5.0),
            FullHistoryExtractor(),
        ]

        manager.create_buffer('test', extractors)

        buffer = manager._states['test'].buffer
        assert isinstance(buffer, TemporalBuffer)
        # max(5.0, inf) = inf
        assert buffer._required_timespan == float('inf')


class TestTemporalBufferManagerWithRealData:
    """Integration tests with real scipp data."""

    def test_single_value_buffer_workflow(self):
        """Test complete workflow with SingleValueBuffer."""
        manager = TemporalBufferManager()
        extractors = [LatestValueExtractor()]

        manager.create_buffer('stream', extractors)

        # Add multiple values
        for i in range(3):
            data = sc.DataArray(
                sc.array(dims=['x'], values=[float(i)] * 2, unit='counts'),
                coords={
                    'x': sc.arange('x', 2, unit='m'),
                    'time': sc.scalar(float(i), unit='s'),
                },
            )
            manager.update_buffer('stream', data)

        # Should only have latest value
        result = manager.get_buffered_data('stream')
        assert result is not None
        # Extract using the extractor
        extracted = extractors[0].extract(result)
        expected = sc.array(dims=['x'], values=[2.0, 2.0], unit='counts')
        assert sc.allclose(extracted.data, expected)

    def test_temporal_buffer_workflow(self):
        """Test complete workflow with TemporalBuffer."""
        manager = TemporalBufferManager()
        extractors = [WindowAggregatingExtractor(window_duration_seconds=5.0)]

        manager.create_buffer('stream', extractors)

        # Add multiple time slices
        for i in range(3):
            data = sc.DataArray(
                sc.array(dims=['x'], values=[float(i)] * 2, unit='counts'),
                coords={
                    'x': sc.arange('x', 2, unit='m'),
                    'time': sc.scalar(float(i), unit='s'),
                },
            )
            manager.update_buffer('stream', data)

        # Should have all data concatenated
        result = manager.get_buffered_data('stream')
        assert result is not None
        assert 'time' in result.dims
        assert result.sizes['time'] == 3
