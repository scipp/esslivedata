# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass

import scipp as sc

from ess.livedata.dashboard.buffer_config import BufferConfig, BufferStrategyType
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.history_buffer_service import (
    BufferViewType,
    HistoryBufferService,
    SimpleBufferSubscriber,
)


@dataclass(frozen=True)
class SimpleKey:
    """Simple key for testing."""

    name: str


class FakePipe:
    """Fake pipe for testing subscribers."""

    def __init__(self):
        self.received_data = []

    def send(self, data):
        self.received_data.append(data)


class TestHistoryBufferService:
    def test_initialization(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        assert len(buffer_service.keys) == 0

    def test_register_key_with_explicit_config(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key = SimpleKey("test1")
        config = BufferConfig(
            strategy_type=BufferStrategyType.FIXED_SIZE, max_points=100
        )

        buffer_service.register_key(key, config=config)

        assert key in buffer_service.keys

    def test_register_key_with_initial_data(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key = SimpleKey("timeseries1")
        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0]),
            coords={'time': sc.array(dims=['time'], values=[0.0, 1.0, 2.0], unit='s')},
        )

        buffer_service.register_key(key, initial_data=data)

        assert key in buffer_service.keys
        result = buffer_service.get_buffer(key)
        assert result is not None
        assert sc.identical(result, data)

    def test_lazy_initialization_on_first_data(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key = SimpleKey("test1")
        # Register key without config or initial data
        buffer_service.register_key(key)

        # Simulate data update from DataService
        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0]),
            coords={'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s')},
        )

        # Trigger via DataService (which will notify buffer service)
        with data_service.transaction():
            data_service[key] = data

        # Buffer should now be initialized
        assert key in buffer_service.keys
        result = buffer_service.get_buffer(key)
        assert result is not None

    def test_data_service_integration(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key = SimpleKey("test1")
        config = BufferConfig(
            strategy_type=BufferStrategyType.FIXED_SIZE, max_points=100
        )
        buffer_service.register_key(key, config=config)

        # Update DataService (which should trigger buffer service)
        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2])},
        )

        with data_service.transaction():
            data_service[key] = data

        # Buffer should have received the data
        result = buffer_service.get_buffer(key)
        assert result is not None
        assert len(result) == 3

    def test_buffer_accumulation(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key = SimpleKey("test1")
        config = BufferConfig(
            strategy_type=BufferStrategyType.FIXED_SIZE, max_points=100
        )
        buffer_service.register_key(key, config=config)

        # Add data in multiple updates
        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1])},
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['time'], values=[3.0, 4.0]),
            coords={'time': sc.array(dims=['time'], values=[2, 3])},
        )

        with data_service.transaction():
            data_service[key] = data1

        with data_service.transaction():
            data_service[key] = data2

        # Buffer should have accumulated both
        result = buffer_service.get_buffer(key)
        assert result is not None
        assert len(result) == 4

    def test_get_window(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key = SimpleKey("test1")
        config = BufferConfig(
            strategy_type=BufferStrategyType.FIXED_SIZE, max_points=100
        )
        buffer_service.register_key(key, config=config)

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0, 4.0, 5.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2, 3, 4])},
        )

        with data_service.transaction():
            data_service[key] = data

        window = buffer_service.get_window(key, size=3)
        assert window is not None
        assert len(window) == 3
        assert sc.identical(
            window.data, sc.array(dims=['time'], values=[3.0, 4.0, 5.0])
        )

    def test_subscriber_notification(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key = SimpleKey("test1")
        config = BufferConfig(
            strategy_type=BufferStrategyType.FIXED_SIZE, max_points=100
        )
        buffer_service.register_key(key, config=config)

        # Create a subscriber
        pipe = FakePipe()
        subscriber = SimpleBufferSubscriber(
            keys={key}, pipe=pipe, view_type=BufferViewType.FULL
        )
        buffer_service.register_subscriber(subscriber)

        # Update data
        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2])},
        )

        with data_service.transaction():
            data_service[key] = data

        # Subscriber should have received notification
        assert len(pipe.received_data) == 1
        received = pipe.received_data[0]
        assert key in received
        assert received[key] is not None

    def test_subscriber_window_view(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key = SimpleKey("test1")
        config = BufferConfig(
            strategy_type=BufferStrategyType.FIXED_SIZE, max_points=100
        )
        buffer_service.register_key(key, config=config)

        # Create a subscriber with window view
        pipe = FakePipe()
        subscriber = SimpleBufferSubscriber(
            keys={key}, pipe=pipe, view_type=BufferViewType.WINDOW, window_size=2
        )
        buffer_service.register_subscriber(subscriber)

        # Update data
        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0, 4.0, 5.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2, 3, 4])},
        )

        with data_service.transaction():
            data_service[key] = data

        # Subscriber should have received window of size 2
        assert len(pipe.received_data) == 1
        received = pipe.received_data[0]
        assert key in received
        windowed_data = received[key]
        assert len(windowed_data) == 2

    def test_clear_buffer(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key = SimpleKey("test1")
        config = BufferConfig(
            strategy_type=BufferStrategyType.FIXED_SIZE, max_points=100
        )
        buffer_service.register_key(key, config=config)

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2])},
        )

        with data_service.transaction():
            data_service[key] = data

        assert buffer_service.get_buffer(key) is not None

        buffer_service.clear_buffer(key)
        assert buffer_service.get_buffer(key) is None

    def test_memory_usage_tracking(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key = SimpleKey("test1")
        config = BufferConfig(
            strategy_type=BufferStrategyType.FIXED_SIZE, max_points=100
        )
        buffer_service.register_key(key, config=config)

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2])},
        )

        with data_service.transaction():
            data_service[key] = data

        memory_usage = buffer_service.get_memory_usage()
        assert key in memory_usage
        assert memory_usage[key] > 0

    def test_unregister_key(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key = SimpleKey("test1")
        config = BufferConfig(
            strategy_type=BufferStrategyType.FIXED_SIZE, max_points=100
        )
        buffer_service.register_key(key, config=config)

        assert key in buffer_service.keys

        buffer_service.unregister_key(key)
        assert key not in buffer_service.keys
        assert buffer_service.get_buffer(key) is None

    def test_multiple_keys(self):
        data_service = DataService[SimpleKey, sc.DataArray]()
        buffer_service = HistoryBufferService(data_service)

        key1 = SimpleKey("test1")
        key2 = SimpleKey("test2")

        config = BufferConfig(
            strategy_type=BufferStrategyType.FIXED_SIZE, max_points=100
        )
        buffer_service.register_key(key1, config=config)
        buffer_service.register_key(key2, config=config)

        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1])},
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['time'], values=[3.0, 4.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1])},
        )

        with data_service.transaction():
            data_service[key1] = data1
            data_service[key2] = data2

        assert buffer_service.get_buffer(key1) is not None
        assert buffer_service.get_buffer(key2) is not None
        assert len(buffer_service.get_buffer(key1)) == 2
        assert len(buffer_service.get_buffer(key2)) == 2
