# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for BufferManager."""

from __future__ import annotations

import pytest
import scipp as sc

from ess.livedata.dashboard.buffer import BufferFactory
from ess.livedata.dashboard.buffer_manager import BufferManager
from ess.livedata.dashboard.extractors import (
    FullHistoryExtractor,
    LatestValueExtractor,
    WindowAggregatingExtractor,
)


@pytest.fixture
def buffer_factory() -> BufferFactory:
    """Create a buffer factory for testing."""
    return BufferFactory()


@pytest.fixture
def buffer_manager(buffer_factory: BufferFactory) -> BufferManager:
    """Create a buffer manager for testing."""
    return BufferManager(buffer_factory)


class TestBufferManagerCreation:
    """Tests for buffer creation."""

    def test_create_buffer_with_latest_value_extractor(
        self, buffer_manager: BufferManager
    ):
        """Test creating buffer with LatestValueExtractor."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [LatestValueExtractor()])

        # Buffer should be created (no data initially)
        buffer = buffer_manager[key]
        assert buffer.get_all() is None

    def test_create_buffer_with_window_aggregating_extractor(
        self, buffer_manager: BufferManager
    ):
        """Test creating buffer with WindowAggregatingExtractor."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(
            key, template, [WindowAggregatingExtractor(window_duration_seconds=5.0)]
        )

        # Buffer should be created (no data initially)
        buffer = buffer_manager[key]
        assert buffer.get_all() is None

    def test_create_buffer_with_full_history_extractor(
        self, buffer_manager: BufferManager
    ):
        """Test creating buffer with FullHistoryExtractor."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [FullHistoryExtractor()])

        # Buffer should be created (no data initially)
        buffer = buffer_manager[key]
        assert buffer.get_all() is None

    def test_create_buffer_with_multiple_extractors(
        self, buffer_manager: BufferManager
    ):
        """Test creating buffer with multiple extractors."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(
            key,
            template,
            [
                LatestValueExtractor(),
                WindowAggregatingExtractor(window_duration_seconds=2.0),
                FullHistoryExtractor(),
            ],
        )

        # Buffer should be created (no data initially)
        buffer = buffer_manager[key]
        assert buffer.get_all() is None


class TestBufferManagerUpdateAndResize:
    """Tests for buffer updates and automatic resizing."""

    def test_update_buffer_appends_data(self, buffer_manager: BufferManager):
        """Test that update_buffer appends data to buffer."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        extractor = LatestValueExtractor()
        buffer_manager.create_buffer(key, template, [extractor])

        data = sc.scalar(42, unit='counts')
        buffer_manager.update_buffer(key, data)

        buffer = buffer_manager[key]
        assert buffer.get_all() is not None
        result = extractor.extract(buffer.get_all())
        assert result.value == 42

    def test_buffer_grows_for_full_history(self, buffer_manager: BufferManager):
        """Test that buffer grows when FullHistoryExtractor is added."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [LatestValueExtractor()])

        # Add data
        buffer_manager.update_buffer(key, sc.scalar(1, unit='counts'))
        buffer = buffer_manager[key]
        assert buffer.get_all() is not None

        # Add FullHistoryExtractor
        buffer_manager.add_extractor(key, FullHistoryExtractor())

        # Buffer should grow (or be ready to grow)
        # Add more data to trigger growth
        for i in range(2, 5):
            buffer_manager.update_buffer(key, sc.scalar(i, unit='counts'))

        # Buffer should have data
        buffer = buffer_manager[key]
        data = buffer.get_all()
        assert data is not None
        # Memory usage should be non-zero
        assert buffer.get_memory_usage() > 0

    def test_buffer_grows_for_time_window_with_time_coord(
        self, buffer_manager: BufferManager
    ):
        """Test buffer grows to satisfy WindowAggregatingExtractor with time."""
        # Create data with time coordinates
        template = sc.DataArray(
            sc.scalar(1.0, unit='counts'),
            coords={'time': sc.scalar(0.0, unit='s')},
        )
        key = 'test_key'
        extractor = WindowAggregatingExtractor(window_duration_seconds=1.0)
        buffer_manager.create_buffer(key, template, [extractor])

        # Add data points spaced 0.1 seconds apart
        for i in range(5):
            data = sc.DataArray(
                sc.scalar(float(i), unit='counts'),
                coords={'time': sc.scalar(i * 0.1, unit='s')},
            )
            buffer_manager.update_buffer(key, data)

        # After 5 points at 0.1s spacing, coverage should be 0.4s
        buffer = buffer_manager[key]
        buffered_data = buffer.get_all()
        assert buffered_data is not None
        time_span = buffered_data.coords['time'][-1] - buffered_data.coords['time'][0]
        coverage = float(time_span.to(unit='s').value)
        assert coverage == pytest.approx(0.4, abs=0.01)

        # Add more points to reach 1.0s coverage
        for i in range(5, 15):
            data = sc.DataArray(
                sc.scalar(float(i), unit='counts'),
                coords={'time': sc.scalar(i * 0.1, unit='s')},
            )
            buffer_manager.update_buffer(key, data)

        # Coverage should now be >= 1.0s
        buffer = buffer_manager[key]
        buffered_data = buffer.get_all()
        assert buffered_data is not None
        time_span = buffered_data.coords['time'][-1] - buffered_data.coords['time'][0]
        coverage = float(time_span.to(unit='s').value)
        assert coverage >= 1.0


class TestBufferManagerValidation:
    """Tests for extractor requirement validation."""

    def test_validate_latest_value_extractor(self, buffer_manager: BufferManager):
        """Test validation for LatestValueExtractor."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [LatestValueExtractor()])

        # Empty buffer should have no data
        buffer = buffer_manager[key]
        assert buffer.get_all() is None

        # Add data
        buffer_manager.update_buffer(key, sc.scalar(1, unit='counts'))

        # Now should have data
        buffer = buffer_manager[key]
        assert buffer.get_all() is not None

    def test_validate_window_extractor_without_time_coord(
        self, buffer_manager: BufferManager
    ):
        """Test that WindowAggregatingExtractor returns False for data without time."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        extractor = WindowAggregatingExtractor(window_duration_seconds=1.0)
        buffer_manager.create_buffer(key, template, [extractor])

        # Adding data without time coordinate is allowed, but requirements not fulfilled
        buffer_manager.update_buffer(key, sc.scalar(1, unit='counts'))

        # Check that requirement is not fulfilled
        buffer = buffer_manager[key]
        data = buffer.get_all()
        assert not extractor.is_requirement_fulfilled(data)

    def test_validate_window_extractor_with_insufficient_coverage(
        self, buffer_manager: BufferManager
    ):
        """Test validation fails when temporal coverage is insufficient."""
        template = sc.DataArray(
            sc.scalar(1.0, unit='counts'),
            coords={'time': sc.scalar(0.0, unit='s')},
        )
        key = 'test_key'
        extractor = WindowAggregatingExtractor(window_duration_seconds=2.0)
        buffer_manager.create_buffer(key, template, [extractor])

        # Add points covering only 0.5 seconds
        for i in range(6):
            data = sc.DataArray(
                sc.scalar(float(i), unit='counts'),
                coords={'time': sc.scalar(i * 0.1, unit='s')},
            )
            buffer_manager.update_buffer(key, data)

        # Check coverage is insufficient
        buffer = buffer_manager[key]
        buffered_data = buffer.get_all()
        assert buffered_data is not None
        time_span = buffered_data.coords['time'][-1] - buffered_data.coords['time'][0]
        coverage = float(time_span.to(unit='s').value)
        assert coverage < 2.0

    def test_validate_full_history_extractor(self, buffer_manager: BufferManager):
        """Test validation for FullHistoryExtractor."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [FullHistoryExtractor()])

        # Add some data
        for i in range(10):
            buffer_manager.update_buffer(key, sc.scalar(i, unit='counts'))

        # Buffer should have grown (FullHistory is never satisfied, keeps growing)
        buffer = buffer_manager[key]
        assert buffer.get_all() is not None
        # Should have non-zero memory usage
        assert buffer.get_memory_usage() > 0


class TestBufferManagerAddExtractor:
    """Tests for adding extractors to existing buffers."""

    def test_add_extractor_triggers_resize(self, buffer_manager: BufferManager):
        """Test that adding extractor triggers immediate growth if needed."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [LatestValueExtractor()])

        # Add some data
        buffer = buffer_manager[key]
        for i in range(5):
            buffer.append(sc.scalar(i, unit='counts'))

        initial_memory = buffer.get_memory_usage()

        # Add FullHistoryExtractor (should trigger growth preparation)
        buffer_manager.add_extractor(key, FullHistoryExtractor())

        # Data should still be present
        assert buffer.get_all() is not None
        # Memory shouldn't decrease
        assert buffer.get_memory_usage() >= initial_memory
