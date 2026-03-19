# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from scipp.testing import assert_identical
from streaming_data_types import eventdata_ev44

from ess.livedata.handlers.to_nxevent_data import (
    DetectorEvents,
    MonitorEvents,
    ToNXevent_data,
)


def test_MonitorEvents_from_ev44() -> None:
    ev44 = eventdata_ev44.EventData(
        source_name='ignored',
        message_id=0,
        reference_time=[],
        reference_time_index=[0],
        pixel_id=[1, 1, 1],
        time_of_flight=[1, 2, 3],
    )
    monitor_events = MonitorEvents.from_ev44(ev44)
    assert monitor_events.time_of_arrival == [1, 2, 3]
    assert monitor_events.unit == 'ns'


@pytest.mark.parametrize('events_cls', [MonitorEvents, DetectorEvents])
def test_MonitorEvents_from_ev44_raises_with_multi_pulse_message(
    events_cls: MonitorEvents,
) -> None:
    ev44 = eventdata_ev44.EventData(
        source_name='ignored',
        message_id=0,
        reference_time=[1, 2],
        reference_time_index=[0, 1],
        pixel_id=[1, 1, 1],
        time_of_flight=[1, 2, 3],
    )
    with pytest.raises(NotImplementedError):
        events_cls.from_ev44(ev44)


def test_MonitorEvents_ToNXevent_data() -> None:
    to_nx = ToNXevent_data()
    to_nx.add(0, MonitorEvents(time_of_arrival=[1, 10], unit='ns'))
    to_nx.add(1000, MonitorEvents(time_of_arrival=[2], unit='ns'))
    events = to_nx.get()
    content = sc.DataArray(
        data=sc.ones(dims=['event'], shape=[3], unit='counts', dtype='float32'),
        coords={
            'event_time_offset': sc.array(
                dims=['event'], values=[1, 10, 2], unit='ns', dtype='int32'
            )
        },
    )
    assert_identical(
        events,
        sc.DataArray(
            data=sc.bins(
                begin=sc.array(dims=['event'], values=[0, 2], unit=None),
                dim='event',
                data=content,
            ),
            coords={
                'event_time_zero': sc.epoch(unit='ns')
                + sc.array(dims=['event_time_zero'], values=[0, 1000], unit='ns')
            },
        ),
    )
    to_nx.release_buffers()
    empty_events = to_nx.get()
    assert empty_events.sizes == {'event_time_zero': 0}
    to_nx.release_buffers()
    assert_identical(empty_events, events[0:0])


def test_DetectorEvents_ToNXevent_data() -> None:
    to_nx = ToNXevent_data()
    to_nx.add(0, DetectorEvents(time_of_arrival=[1, 10], pixel_id=[2, 1], unit='ns'))
    to_nx.add(1000, DetectorEvents(time_of_arrival=[2], pixel_id=[1], unit='ns'))
    events = to_nx.get()
    content = sc.DataArray(
        data=sc.ones(dims=['event'], shape=[3], unit='counts', dtype='float32'),
        coords={
            'event_time_offset': sc.array(
                dims=['event'], values=[1, 10, 2], unit='ns', dtype='int32'
            ),
            'event_id': sc.array(
                dims=['event'], values=[2, 1, 1], unit=None, dtype='int32'
            ),
        },
    )
    assert_identical(
        events,
        sc.DataArray(
            data=sc.bins(
                begin=sc.array(dims=['event'], values=[0, 2], unit=None),
                dim='event',
                data=content,
            ),
            coords={
                'event_time_zero': sc.epoch(unit='ns')
                + sc.array(dims=['event_time_zero'], values=[0, 1000], unit='ns')
            },
        ),
    )
    to_nx.release_buffers()
    empty_events = to_nx.get()
    assert empty_events.sizes == {'event_time_zero': 0}
    to_nx.release_buffers()
    assert_identical(empty_events, events[0:0])


def test_DetectorEvents_raises_on_array_length_mismatch() -> None:
    with pytest.raises(
        ValueError, match="pixel_id and time_of_arrival must have the same length"
    ):
        DetectorEvents(time_of_arrival=[1, 2, 3], pixel_id=[1, 2], unit='ns')


def test_ToNXevent_data_wrong_unit() -> None:
    to_nx = ToNXevent_data()
    with pytest.raises(ValueError, match="Expected unit 'ns'"):
        to_nx.add(0, MonitorEvents(time_of_arrival=[1, 2, 3], unit='s'))


def test_ToNXevent_data_mixing_event_types() -> None:
    to_nx = ToNXevent_data()
    to_nx.add(0, MonitorEvents(time_of_arrival=[1, 2, 3], unit='ns'))

    with pytest.raises(ValueError, match="Inconsistent event_id"):
        to_nx.add(
            1000, DetectorEvents(time_of_arrival=[4, 5], pixel_id=[1, 2], unit='ns')
        )


def test_ToNXevent_data_mixing_event_types_reversed() -> None:
    to_nx = ToNXevent_data()
    to_nx.add(0, DetectorEvents(time_of_arrival=[1, 2], pixel_id=[1, 2], unit='ns'))

    with pytest.raises(ValueError, match="Inconsistent event_id"):
        to_nx.add(1000, MonitorEvents(time_of_arrival=[3, 4, 5], unit='ns'))


def test_ToNXevent_data_get_raises_if_no_data_was_added() -> None:
    to_nx = ToNXevent_data()
    with pytest.raises(ValueError, match="No data has been added"):
        to_nx.get()


def test_ToNXevent_data_get_works_if_no_data_after_previous_get() -> None:
    to_nx = ToNXevent_data()
    to_nx.add(0, DetectorEvents(time_of_arrival=[1, 2], pixel_id=[1, 2], unit='ns'))
    ref = to_nx.get()
    to_nx.release_buffers()
    # Empty, but initial data allowed for full initialization
    empty = to_nx.get()
    assert sc.identical(empty, ref['event_time_zero', 0:0])


def test_ToNXevent_data_second_cycle_not_corrupted_by_buffer_reuse() -> None:
    """After release, a new add/get cycle must return fresh data, not stale values."""
    acc = ToNXevent_data()

    acc.add(0, DetectorEvents(time_of_arrival=[10, 20], pixel_id=[1, 2], unit='ns'))
    acc.get()
    acc.release_buffers()

    acc.add(
        1000,
        DetectorEvents(time_of_arrival=[30, 40, 50], pixel_id=[3, 4, 5], unit='ns'),
    )
    result2 = acc.get()
    acc.release_buffers()

    toa2 = result2.bins.coords['event_time_offset'].values[0].values
    assert list(toa2) == [30, 40, 50]
    pid2 = result2.bins.coords['event_id'].values[0].values
    assert list(pid2) == [3, 4, 5]


def test_ToNXevent_data_buffer_grows_across_cycles() -> None:
    """A cycle with more events than the previous must still be correct."""
    acc = ToNXevent_data()

    acc.add(0, MonitorEvents(time_of_arrival=[1, 2], unit='ns'))
    acc.get()
    acc.release_buffers()

    acc.add(1000, MonitorEvents(time_of_arrival=[10, 20, 30, 40, 50], unit='ns'))
    result = acc.get()
    acc.release_buffers()

    toa = result.bins.coords['event_time_offset'].values[0].values
    assert list(toa) == [10, 20, 30, 40, 50]


def test_ToNXevent_data_buffer_shrinks_across_cycles() -> None:
    """A cycle with fewer events than a previous one must not include stale data."""
    acc = ToNXevent_data()

    acc.add(0, MonitorEvents(time_of_arrival=[1, 2, 3, 4, 5], unit='ns'))
    acc.get()
    acc.release_buffers()

    acc.add(1000, MonitorEvents(time_of_arrival=[10, 20], unit='ns'))
    result = acc.get()
    acc.release_buffers()

    toa = result.bins.coords['event_time_offset'].values[0].values
    assert list(toa) == [10, 20]
    assert result.bins.size().values[0] == 2


def test_ToNXevent_data_get_raises_without_release() -> None:
    acc = ToNXevent_data()
    acc.add(0, MonitorEvents(time_of_arrival=[1], unit='ns'))
    acc.get()
    # No release_buffers() call
    acc.add(1000, MonitorEvents(time_of_arrival=[2], unit='ns'))
    with pytest.raises(RuntimeError, match="have not been released"):
        acc.get()


def test_ToNXevent_data_empty_chunks() -> None:
    to_nx = ToNXevent_data()
    to_nx.add(0, DetectorEvents(time_of_arrival=[], pixel_id=[], unit='ns'))
    to_nx.add(1000, DetectorEvents(time_of_arrival=[], pixel_id=[], unit='ns'))

    events = to_nx.get()
    assert events.sizes["event_time_zero"] == 2
