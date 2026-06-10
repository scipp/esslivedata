# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from ess.livedata.core.timestamp import Timestamp
from ess.livedata.handlers.group_by_pixel import GroupByPixel
from ess.livedata.handlers.to_nxevent_data import DetectorEvents, ToNXevent_data


def _make_detector_number(n_pixels: int) -> sc.Variable:
    return sc.arange('detector_number', 1, n_pixels + 1, unit=None)


def _make_events(pixel_ids: list[int], toa: list[int]) -> DetectorEvents:
    return DetectorEvents(
        pixel_id=np.array(pixel_ids, dtype='int32'),
        time_of_arrival=np.array(toa, dtype='int64'),
        unit='ns',
    )


class TestGroupByPixel:
    def test_get_returns_data_grouped_by_detector_number(self):
        detector_number = _make_detector_number(3)
        acc = GroupByPixel(ToNXevent_data(), detector_number)

        # Events for pixels 1, 2, 3
        events = _make_events(
            pixel_ids=[1, 2, 3, 1, 3],
            toa=[100, 200, 300, 400, 500],
        )
        acc.add(timestamp=Timestamp.from_ns(1000), data=events)
        result = acc.get()

        assert result.dims == ('detector_number',)
        assert result.sizes == {'detector_number': 3}
        assert 'detector_number' in result.coords
        assert sc.identical(result.coords['detector_number'], detector_number)

        # Pixel 1 has 2 events, pixel 2 has 1, pixel 3 has 2
        sizes = result.bins.size()
        assert sizes.values[0] == 2
        assert sizes.values[1] == 1
        assert sizes.values[2] == 2

    def test_multiple_messages_are_accumulated(self):
        detector_number = _make_detector_number(2)
        acc = GroupByPixel(ToNXevent_data(), detector_number)

        acc.add(
            timestamp=Timestamp.from_ns(1000), data=_make_events([1, 2], [100, 200])
        )
        acc.add(
            timestamp=Timestamp.from_ns(2000), data=_make_events([1, 1], [300, 400])
        )
        result = acc.get()

        # Pixel 1 has 3 events total (1 from first + 2 from second)
        # Pixel 2 has 1 event
        sizes = result.bins.size()
        assert sizes.values[0] == 3
        assert sizes.values[1] == 1

    def test_clear_resets_state(self):
        detector_number = _make_detector_number(2)
        acc = GroupByPixel(ToNXevent_data(), detector_number)

        acc.add(
            timestamp=Timestamp.from_ns(1000), data=_make_events([1, 2], [100, 200])
        )
        acc.clear()

        # After clear, inner accumulator has no data
        # get() on inner ToNXevent_data raises when empty
        inner_acc = GroupByPixel(ToNXevent_data(), detector_number)
        inner_acc.add(timestamp=Timestamp.from_ns(3000), data=_make_events([1], [500]))
        result = inner_acc.get()
        assert result.bins.size().sum().value == 1

    def test_get_clears_inner_accumulator(self):
        """ToNXevent_data clears on get(), so subsequent get() without add() fails."""
        detector_number = _make_detector_number(2)
        acc = GroupByPixel(ToNXevent_data(), detector_number)

        acc.add(timestamp=Timestamp.from_ns(1000), data=_make_events([1], [100]))
        acc.get()

        # After get(), adding new data and getting again should work
        acc.add(timestamp=Timestamp.from_ns(2000), data=_make_events([2], [200]))
        result = acc.get()
        assert result.bins.size().values[0] == 0  # pixel 1: no events
        assert result.bins.size().values[1] == 1  # pixel 2: 1 event

    def test_events_have_event_time_offset_coord(self):
        detector_number = _make_detector_number(2)
        acc = GroupByPixel(ToNXevent_data(), detector_number)

        acc.add(
            timestamp=Timestamp.from_ns(1000), data=_make_events([1, 2], [100, 200])
        )
        result = acc.get()

        event_data = result.bins.constituents['data']
        assert 'event_time_offset' in event_data.coords
