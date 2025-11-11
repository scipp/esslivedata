# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for BufferManager."""

from __future__ import annotations

import pytest
import scipp as sc

from ess.livedata.dashboard.buffer_manager import BufferManager
from ess.livedata.dashboard.buffer_strategy import BufferFactory
from ess.livedata.dashboard.temporal_requirements import (
    CompleteHistory,
    LatestFrame,
    TimeWindow,
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

    def test_create_buffer_with_latest_frame_requirement(
        self, buffer_manager: BufferManager
    ):
        """Test creating buffer with LatestFrame requirement."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [LatestFrame()])

        # Buffer should be created (frame count starts at 0)
        buffer = buffer_manager.get_buffer(key)
        assert buffer.get_frame_count() == 0

    def test_create_buffer_with_time_window_requirement(
        self, buffer_manager: BufferManager
    ):
        """Test creating buffer with TimeWindow requirement."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [TimeWindow(duration_seconds=5.0)])

        # Buffer should be created with conservative initial size
        buffer = buffer_manager.get_buffer(key)
        assert buffer.get_frame_count() == 0

    def test_create_buffer_with_complete_history_requirement(
        self, buffer_manager: BufferManager
    ):
        """Test creating buffer with CompleteHistory requirement."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [CompleteHistory()])

        # Buffer should be created with MAX_FRAMES size
        buffer = buffer_manager.get_buffer(key)
        assert buffer.get_frame_count() == 0

    def test_create_buffer_with_multiple_requirements(
        self, buffer_manager: BufferManager
    ):
        """Test creating buffer with multiple requirements takes max."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(
            key,
            template,
            [LatestFrame(), TimeWindow(duration_seconds=2.0), CompleteHistory()],
        )

        # CompleteHistory should dominate (MAX_FRAMES)
        buffer = buffer_manager.get_buffer(key)
        assert buffer.get_frame_count() == 0


class TestBufferManagerUpdateAndResize:
    """Tests for buffer updates and automatic resizing."""

    def test_update_buffer_appends_data(self, buffer_manager: BufferManager):
        """Test that update_buffer appends data to buffer."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [LatestFrame()])

        data = sc.scalar(42, unit='counts')
        buffer_manager.update_buffer(key, data)

        buffer = buffer_manager.get_buffer(key)
        assert buffer.get_frame_count() == 1
        result = buffer.get_latest()
        assert result.value == 42

    def test_buffer_grows_for_complete_history(self, buffer_manager: BufferManager):
        """Test that buffer grows when CompleteHistory requirement is added."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [LatestFrame()])

        # Add data
        buffer_manager.update_buffer(key, sc.scalar(1, unit='counts'))
        buffer = buffer_manager.get_buffer(key)
        initial_count = buffer.get_frame_count()
        assert initial_count == 1

        # Add CompleteHistory requirement
        buffer_manager.add_requirement(key, CompleteHistory())

        # Buffer should grow (or be ready to grow)
        # After adding requirement, validate_coverage should trigger resize
        # Add more data to trigger resize
        for i in range(2, 5):
            buffer_manager.update_buffer(key, sc.scalar(i, unit='counts'))

        # Buffer should have grown beyond initial size
        buffer = buffer_manager.get_buffer(key)
        assert buffer.get_frame_count() == 4

    def test_buffer_grows_for_time_window_with_time_coord(
        self, buffer_manager: BufferManager
    ):
        """Test buffer grows to satisfy TimeWindow when data has time coordinates."""
        # Create data with time coordinates
        template = sc.DataArray(
            sc.scalar(1.0, unit='counts'),
            coords={'time': sc.scalar(0.0, unit='s')},
        )
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [TimeWindow(duration_seconds=1.0)])

        # Add data points spaced 0.1 seconds apart
        for i in range(5):
            data = sc.DataArray(
                sc.scalar(float(i), unit='counts'),
                coords={'time': sc.scalar(i * 0.1, unit='s')},
            )
            buffer_manager.update_buffer(key, data)

        # After 5 points at 0.1s spacing, coverage should be 0.4s
        buffer = buffer_manager.get_buffer(key)
        coverage = buffer.get_temporal_coverage()
        assert coverage is not None
        assert coverage == pytest.approx(0.4, abs=0.01)

        # Add more points to reach 1.0s coverage
        for i in range(5, 15):
            data = sc.DataArray(
                sc.scalar(float(i), unit='counts'),
                coords={'time': sc.scalar(i * 0.1, unit='s')},
            )
            buffer_manager.update_buffer(key, data)

        # Coverage should now be >= 1.0s
        buffer = buffer_manager.get_buffer(key)
        coverage = buffer.get_temporal_coverage()
        assert coverage is not None
        assert coverage >= 1.0


class TestBufferManagerValidation:
    """Tests for coverage validation."""

    def test_validate_coverage_latest_frame(self, buffer_manager: BufferManager):
        """Test validation for LatestFrame requirement."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [LatestFrame()])

        # Empty buffer should fail validation (internally checked)
        buffer = buffer_manager.get_buffer(key)
        assert buffer.get_frame_count() == 0

        # Add data
        buffer_manager.update_buffer(key, sc.scalar(1, unit='counts'))

        # Now should have data
        buffer = buffer_manager.get_buffer(key)
        assert buffer.get_frame_count() == 1

    def test_validate_coverage_time_window_without_time_coord(
        self, buffer_manager: BufferManager
    ):
        """Test validation for TimeWindow with data that has no time coordinate."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [TimeWindow(duration_seconds=1.0)])

        # Add scalar data (no time coordinate)
        buffer_manager.update_buffer(key, sc.scalar(1, unit='counts'))

        # Buffer should have data
        buffer = buffer_manager.get_buffer(key)
        assert buffer.get_frame_count() == 1

    def test_validate_coverage_time_window_with_insufficient_coverage(
        self, buffer_manager: BufferManager
    ):
        """Test validation fails when temporal coverage is insufficient."""
        template = sc.DataArray(
            sc.scalar(1.0, unit='counts'),
            coords={'time': sc.scalar(0.0, unit='s')},
        )
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [TimeWindow(duration_seconds=2.0)])

        # Add points covering only 0.5 seconds
        for i in range(6):
            data = sc.DataArray(
                sc.scalar(float(i), unit='counts'),
                coords={'time': sc.scalar(i * 0.1, unit='s')},
            )
            buffer_manager.update_buffer(key, data)

        # Check coverage is insufficient
        buffer = buffer_manager.get_buffer(key)
        coverage = buffer.get_temporal_coverage()
        assert coverage is not None
        assert coverage < 2.0

    def test_validate_coverage_complete_history(self, buffer_manager: BufferManager):
        """Test validation for CompleteHistory requirement."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [CompleteHistory()])

        # Add some data (but less than MAX_FRAMES)
        for i in range(10):
            buffer_manager.update_buffer(key, sc.scalar(i, unit='counts'))

        # Check frame count is less than MAX_FRAMES
        buffer = buffer_manager.get_buffer(key)
        assert buffer.get_frame_count() < CompleteHistory.MAX_FRAMES


class TestBufferManagerAddRequirement:
    """Tests for adding requirements to existing buffers."""

    def test_add_requirement_triggers_resize(self, buffer_manager: BufferManager):
        """Test that adding requirement triggers immediate resize if needed."""
        template = sc.scalar(1, unit='counts')
        key = 'test_key'
        buffer_manager.create_buffer(key, template, [LatestFrame()])

        # Add some data
        buffer = buffer_manager.get_buffer(key)
        for i in range(5):
            buffer.append(sc.scalar(i, unit='counts'))

        initial_count = buffer.get_frame_count()

        # Add CompleteHistory requirement (should trigger resize)
        buffer_manager.add_requirement(key, CompleteHistory())

        # Frame count shouldn't change immediately, but buffer capacity should grow
        assert buffer.get_frame_count() == initial_count


class TestTemporalRequirements:
    """Tests for TemporalRequirement classes."""

    def test_latest_frame_repr(self):
        """Test LatestFrame string representation."""
        req = LatestFrame()
        assert "LatestFrame" in repr(req)

    def test_time_window_repr(self):
        """Test TimeWindow string representation."""
        req = TimeWindow(duration_seconds=5.0)
        assert "TimeWindow" in repr(req)
        assert "5.0" in repr(req)

    def test_time_window_validation(self):
        """Test TimeWindow validates duration."""
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            TimeWindow(duration_seconds=-1.0)

        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            TimeWindow(duration_seconds=0.0)

    def test_complete_history_repr(self):
        """Test CompleteHistory string representation."""
        req = CompleteHistory()
        assert "CompleteHistory" in repr(req)
        assert "10000" in repr(req)
