# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from ess.livedata.handlers.group_by_pixel import GroupByPixel
from ess.livedata.handlers.to_nxevent_data import DetectorEvents, ToNXevent_data


def _make_empty_detector(n_pixels: int) -> sc.DataArray:
    from ess.livedata.handlers.detector_view.data_source import create_empty_detector

    detector_number = sc.arange('detector_number', 1, n_pixels + 1, unit=None)
    return create_empty_detector(detector_number)


def _make_events(pixel_ids: list[int], toa: list[int]) -> DetectorEvents:
    return DetectorEvents(
        pixel_id=np.array(pixel_ids, dtype='int32'),
        time_of_arrival=np.array(toa, dtype='int64'),
        unit='ns',
    )


class TestGroupByPixel:
    def test_get_returns_data_grouped_by_detector_number(self):
        empty_det = _make_empty_detector(3)
        acc = GroupByPixel(ToNXevent_data(), empty_det)

        # Events for pixels 1, 2, 3
        events = _make_events(
            pixel_ids=[1, 2, 3, 1, 3],
            toa=[100, 200, 300, 400, 500],
        )
        acc.add(timestamp=1000, data=events)
        result = acc.get()

        assert result.dims == ('detector_number',)
        assert result.sizes == {'detector_number': 3}
        assert 'detector_number' in result.coords
        assert sc.identical(
            result.coords['detector_number'],
            empty_det.coords['detector_number'],
        )

        # Pixel 1 has 2 events, pixel 2 has 1, pixel 3 has 2
        sizes = result.bins.size()
        assert sizes.values[0] == 2
        assert sizes.values[1] == 1
        assert sizes.values[2] == 2

    def test_multiple_messages_are_accumulated(self):
        empty_det = _make_empty_detector(2)
        acc = GroupByPixel(ToNXevent_data(), empty_det)

        acc.add(timestamp=1000, data=_make_events([1, 2], [100, 200]))
        acc.add(timestamp=2000, data=_make_events([1, 1], [300, 400]))
        result = acc.get()

        # Pixel 1 has 3 events total (1 from first + 2 from second)
        # Pixel 2 has 1 event
        sizes = result.bins.size()
        assert sizes.values[0] == 3
        assert sizes.values[1] == 1

    def test_clear_resets_state(self):
        empty_det = _make_empty_detector(2)
        acc = GroupByPixel(ToNXevent_data(), empty_det)

        acc.add(timestamp=1000, data=_make_events([1, 2], [100, 200]))
        acc.clear()

        # After clear, inner accumulator has no data
        # get() on inner ToNXevent_data raises when empty
        inner_acc = GroupByPixel(ToNXevent_data(), empty_det)
        inner_acc.add(timestamp=3000, data=_make_events([1], [500]))
        result = inner_acc.get()
        assert result.bins.size().sum().value == 1

    def test_get_clears_inner_accumulator(self):
        """ToNXevent_data clears on get(), so subsequent get() without add() fails."""
        empty_det = _make_empty_detector(2)
        acc = GroupByPixel(ToNXevent_data(), empty_det)

        acc.add(timestamp=1000, data=_make_events([1], [100]))
        acc.get()

        # After get(), adding new data and getting again should work
        acc.add(timestamp=2000, data=_make_events([2], [200]))
        result = acc.get()
        assert result.bins.size().values[0] == 0  # pixel 1: no events
        assert result.bins.size().values[1] == 1  # pixel 2: 1 event

    def test_events_have_event_time_offset_coord(self):
        empty_det = _make_empty_detector(2)
        acc = GroupByPixel(ToNXevent_data(), empty_det)

        acc.add(timestamp=1000, data=_make_events([1, 2], [100, 200]))
        result = acc.get()

        event_data = result.bins.constituents['data']
        assert 'event_time_offset' in event_data.coords
